"""
Piper Geometry: Orthogonal Procrustes Latent Alignment Engine.
Computes optimal rotation matrices to align vector spaces between disparate models or layers.
"""

from typing import Tuple
import torch


class LatentAligner:
    """Computes orthogonal rotations to map Source activations into Receiver coordinates."""

    @staticmethod
    def compute_procrustes(source_matrix: torch.Tensor, target_matrix: torch.Tensor) -> torch.Tensor:
        """
        Calculates orthogonal rotation matrix W that minimizes ||source @ W - target||_F.
        Expects shapes: [num_samples, hidden_dim].
        """
        A = source_matrix.to(torch.float32)
        B = target_matrix.to(torch.float32)

        # Cross-covariance matrix M = A^T @ B
        M = torch.matmul(A.t(), B)

        # Singular Value Decomposition: M = U @ S @ Vh
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)

        # Optimal orthogonal transformation matrix: W = U @ Vh
        W = torch.matmul(U, Vh)
        return W

    @staticmethod
    def align_and_measure(
        source_vec: torch.Tensor,
        target_vec: torch.Tensor,
        rotation_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Rotates source_vec into target coordinate space and computes cosine similarity.
        """
        # Apply orthogonal rotation: v_aligned = v_source @ W
        rotated_vec = torch.matmul(source_vec.to(torch.float32), rotation_matrix)
        rotated_norm = rotated_vec / torch.norm(rotated_vec, p=2)
        target_norm = target_vec.to(torch.float32) / torch.norm(target_vec.to(torch.float32), p=2)

        cosine_sim = torch.dot(rotated_norm, target_norm).item()
        return rotated_norm, cosine_sim


if __name__ == "__main__":
    print("[Aligner] Testing Orthogonal Procrustes calculation on synthetic latent spaces...")

    dim = 896
    num_calibration_samples = 1000

    # Simulate Source (A) and Target (B) with an arbitrary rigid rotation Q
    torch.manual_seed(42)
    source_calibration = torch.randn(num_calibration_samples, dim)
    source_calibration /= torch.norm(source_calibration, dim=1, keepdim=True)

    # Generate a ground-truth orthogonal rotation matrix Q
    random_matrix = torch.randn(dim, dim)
    q, _ = torch.linalg.qr(random_matrix)
    target_calibration = torch.matmul(source_calibration, q)

    # Compute learned rotation W
    learned_W = LatentAligner.compute_procrustes(source_calibration, target_calibration)

    # Test alignment on an unseen test vector
    unseen_source = torch.randn(dim)
    unseen_source /= torch.norm(unseen_source)
    unseen_target = torch.matmul(unseen_source, q)

    rotated_test, similarity = LatentAligner.align_and_measure(unseen_source, unseen_target, learned_W)
    print(f"Alignment verification on unseen concept -> Cosine Similarity: {similarity:.4f}")