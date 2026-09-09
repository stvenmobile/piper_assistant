"""
Piper Geometry: shared layer-injection mechanics.

Originally lived inside concept_transfer_test.py, written for that one
script. Now used by two consumers - the qualitative four-condition test
and the trained-adapter script - so it's factored out here rather than
duplicated. Everything in this module is generic to "extract a hidden
state at a given layer, or override one during a forward pass" - nothing
here is specific to the qualitative test's four conditions or to the
adapter's training loop.

The rotation/extraction utilities are also autograd-safe by construction:
LayerInjectionHook only ever calls .to(dtype=..., device=...) on the
injected tensor, never .detach() - so when the trained adapter passes in
a tensor that requires_grad, gradients flow back through the hook, into
the adapter's own parameters, correctly. This wasn't a deliberate design
for training when it was first written for the (gradient-free) qualitative
test, but it didn't need to change to become gradient-safe either.
"""

import torch


def build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts):
    """Closed-form orthogonal Procrustes fit (same math as
    CongruenceOptimizer.run_trial's rotation step) between two layers'
    last-token concept summaries. Used both as the qualitative test's
    fixed rotation and as the trained adapter's warm-start."""
    source_vecs, target_vecs = [], []
    for concept in calibration_concepts:
        src_acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])
        tgt_acts = target_extractor.extract_activations(prompt=concept, target_layers=[receiver_layer])
        source_vecs.append(src_acts[source_layer])
        target_vecs.append(tgt_acts[receiver_layer])

    A = torch.stack(source_vecs).to(torch.float32)
    B = torch.stack(target_vecs).to(torch.float32)
    M = A.t() @ B
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh
    return W


def random_semi_orthogonal_like(W: torch.Tensor) -> torch.Tensor:
    """A random matrix of the same shape as W, with the same orthogonality
    property W actually has, for the negative control condition.

    W comes from an SVD (U @ Vh, U square orthogonal, Vh row-orthonormal),
    so W @ W.T = I on the source-dimension side - its ROWS are
    orthonormal, not its columns. That's the only orthogonality direction
    that's even possible here: source_dim (896) < target_dim (3072), and
    you cannot fit more mutually-orthogonal columns than there are
    dimensions to hold them in (QR on a "wide" matrix caps out at
    min(rows, cols) orthonormal columns - trying to get 3072 orthonormal
    columns out of 896-dimensional rows is a mathematical impossibility,
    not just an unlikely random draw). QR on the transpose (a "tall"
    matrix, cols > rows) gives orthonormal columns there instead;
    transposing back yields the row-orthonormal shape actually needed.
    """
    rows, cols = W.shape
    # QR runs on CPU regardless of W's own device, then the result moves
    # to W's device afterward - not just a style choice. This Jetson's
    # PyTorch build has a broken CUDA cusolver linkage for at least some
    # GPU linalg routines (torch.linalg.qr on CUDA fails here with
    # "undefined symbol: cusolverDnXsyevBatched_bufferSize"). CPU QR is
    # proven to work: build_rotation's SVD already runs entirely on CPU,
    # for the unrelated reason that ResidualExtractor.extract_activations()
    # always returns .cpu() tensors, and never hits this failure.
    Q, _ = torch.linalg.qr(torch.randn(cols, rows, dtype=W.dtype))
    return Q.t().to(device=W.device)


def extract_full_sequence(model, tokenizer, device, text: str, layer_idx: int) -> torch.Tensor:
    """Every token's hidden state at layer_idx for `text`, not just a
    last-token summary - generation needs the whole sequence to avoid
    stripping away the context a real passage carries."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_idx][0]  # (seq_len, hidden_dim), batch dim dropped


def norm_stats(states: torch.Tensor) -> dict:
    """Per-token L2 norm summary, to check whether translated vectors
    carry a systematically different scale than what the target layer's
    own native activations look like. A Procrustes rotation is provably
    norm-preserving (W @ W.T = I by construction), so rotated/random
    states are mathematically guaranteed to carry Qwen's original
    magnitudes unchanged - if that turns out to differ from Phi-4-mini's
    own native scale at the same layer, translation can't close that gap
    on its own, no matter how good the rotation's direction is."""
    norms = states.norm(dim=-1)
    return {
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
    }


def print_norm_stats(label: str, stats: dict) -> None:
    print(f"[layer_injection]   {label}: mean={stats['mean']:.2f}  "
          f"std={stats['std']:.2f}  min={stats['min']:.2f}  max={stats['max']:.2f}")


class LayerInjectionHook:
    """Overrides one transformer layer's output on its first invocation -
    the prefill pass over the full placeholder sequence - with externally
    supplied hidden states, then gets out of the way for every subsequent
    single-token decode step, which must run unmodified so newly
    generated tokens reflect the model's own real computation given
    whatever is now sitting in the stream at the injected positions.

    Gradient-safe: only ever calls .to(dtype=..., device=...) on the
    injected tensor, never .detach() - if the caller passes in a tensor
    that requires_grad (the trained adapter's output), gradients flow
    back through this hook correctly during backprop."""

    def __init__(self, injected_states: torch.Tensor):
        self.injected_states = injected_states  # (1, seq_len, hidden_dim)
        self.applied = False

    def __call__(self, module, inputs, output):
        if self.applied:
            return output
        self.applied = True
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.shape[1] != self.injected_states.shape[1]:
            # Not the prefill pass we expected - leave it alone rather
            # than silently apply an injection with mismatched shape.
            return output
        new_hidden = self.injected_states.to(dtype=hidden.dtype, device=hidden.device)
        return (new_hidden,) + output[1:] if is_tuple else new_hidden


class PartialLayerInjectionHook:
    """Overrides only the first inject_len positions of a layer's output,
    leaving every position after that - e.g. a fixed real-text suffix
    appended after translated content - to reflect the model's own actual
    computation over whatever now sits in the stream ahead of it.

    Unlike LayerInjectionHook, this isn't restricted to firing once. It's
    built for a single non-autoregressive forward pass (see train_adapter.py's
    forward_with_injection), which calls the hooked layer exactly once -
    there's no prefill-vs-decode-step distinction to guard against here.

    Still gradient-safe: torch.cat is differentiable, and injected_states
    is only ever moved with .to(dtype=..., device=...), never detached -
    so a loss computed downstream (e.g. at the final position's logits)
    backpropagates through the concatenated positions into whatever
    produced injected_states, exactly like LayerInjectionHook."""

    def __init__(self, injected_states: torch.Tensor):
        self.injected_states = injected_states  # (1, inject_len, hidden_dim)

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        inject_len = self.injected_states.shape[1]
        if inject_len > hidden.shape[1]:
            # Sequence shorter than what we'd need to inject into - leave
            # it alone rather than inject a mismatched/truncated span.
            return output
        injected = self.injected_states.to(dtype=hidden.dtype, device=hidden.device)
        new_hidden = torch.cat([injected, hidden[:, inject_len:, :]], dim=1)
        return (new_hidden,) + output[1:] if is_tuple else new_hidden


def generate_with_injection(model, tokenizer, device, layer_idx: int, injected_states: torch.Tensor,
                             max_new_tokens: int = 60) -> str:
    """Runs a neutral placeholder sequence through the model with
    layer_idx's output overridden (via LayerInjectionHook) on the
    prefill pass only, then decodes just the newly generated continuation
    - not the placeholder prefix, which carries no information itself.
    Inference-only (no_grad) - for training, see train_adapter.py's
    forward_with_injection, a single differentiable forward pass instead
    of this function's full autoregressive generate() loop."""
    seq_len = injected_states.shape[0]
    placeholder_id = tokenizer.eos_token_id
    input_ids = torch.full((1, seq_len), placeholder_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    hook = LayerInjectionHook(injected_states.unsqueeze(0))
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    if not hook.applied:
        print(f"[layer_injection] WARNING: injection hook never fired for layer {layer_idx}")
    return tokenizer.decode(output_ids[0, seq_len:], skip_special_tokens=True).strip()


def generate_normally(model, tokenizer, device, text: str, max_new_tokens: int = 60) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, seq_len:], skip_special_tokens=True).strip()
