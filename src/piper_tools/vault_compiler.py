"""
Piper Tools: Obsidian Vault Compiler & Experiment Knowledge Graph Generator.
Targets ~/piper_assistant/obsidian/ with subdirectories: Experiments, Concepts, Journals.
"""

from pathlib import Path
from datetime import datetime
import yaml

# Workspace path resolution: ~/piper_assistant/obsidian
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
OBSIDIAN_DIR = WORKSPACE_DIR / "obsidian"

EXPERIMENTS_DIR = OBSIDIAN_DIR / "Experiments"
CONCEPTS_DIR = OBSIDIAN_DIR / "Concepts"
JOURNALS_DIR = OBSIDIAN_DIR / "Journals"

for folder in [EXPERIMENTS_DIR, CONCEPTS_DIR, JOURNALS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


class VaultCompiler:
    def __init__(self, vault_path: Path = OBSIDIAN_DIR):
        self.vault_path = vault_path

    def record_experiment(
        self,
        experiment_id: str,
        target_concept: str,
        source_layer: int,
        receiver_layer: int,
        metrics: dict,
        findings_summary: str
    ) -> Path:
        """Compiles a Level 1 tensor experiment into an Obsidian note with frontmatter."""
        file_path = EXPERIMENTS_DIR / f"{experiment_id}.md"
        concept_link = f"[[{target_concept}]]"
        today_str = datetime.now().strftime("%Y-%m-%d")

        frontmatter = {
            "id": experiment_id,
            "type": "experiment",
            "date": datetime.now().isoformat(),
            "target_concept": target_concept,
            "source_layer": source_layer,
            "receiver_layer": receiver_layer,
            "cosine_similarity": round(float(metrics.get("cosine_sim", 0.0)), 4),
            "transfer_success": bool(metrics.get("success", False)),
            "tags": ["latent_transfer", "geometry_of_reasoning", "level1_autonomous"]
        }

        content = (
            f"---\n{yaml.dump(frontmatter, sort_keys=False)}---\n\n"
            f"# Experiment: {experiment_id}\n\n"
            f"**Target Concept**: {concept_link}\n"
            f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## 1. Transmission Parameters\n"
            f"- **Source Layer**: Layer `{source_layer}`\n"
            f"- **Receiver Layer**: Layer `{receiver_layer}`\n"
            f"- **Cosine Alignment**: `{frontmatter['cosine_similarity']}`\n"
            f"- **Transfer Result**: `{'SUCCESS' if frontmatter['transfer_success'] else 'FAILED'}`\n\n"
            f"## 2. Autonomous Findings (Level 2 Summary)\n"
            f"{findings_summary}\n\n"
            f"## 3. Related Links\n"
            f"- Target Concept: {concept_link}\n"
            f"- Daily Journal: [[{today_str}]]\n"
        )

        file_path.write_text(content, encoding="utf-8")
        self._update_concept_index(target_concept, experiment_id, metrics)
        self._append_daily_journal(experiment_id, target_concept, metrics)
        return file_path

    def _update_concept_index(self, concept_name: str, experiment_id: str, metrics: dict):
        """Maintains bidirectional links in the concept note."""
        concept_path = CONCEPTS_DIR / f"{concept_name}.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        entry = (
            f"- **{timestamp}** | [[{experiment_id}]] | "
            f"Sim: `{metrics.get('cosine_sim', 0.0):.3f}` | "
            f"Status: `{'Success' if metrics.get('success') else 'Fail'}`\n"
        )

        if not concept_path.exists():
            header = (
                f"# Concept: {concept_name}\n\n"
                f"Topological manifold profile and cross-AI latent transmission logs.\n\n"
                f"## Experiment History\n"
            )
            concept_path.write_text(header + entry, encoding="utf-8")
        else:
            with open(concept_path, "a", encoding="utf-8") as f:
                f.write(entry)

    def _append_daily_journal(self, experiment_id: str, target_concept: str, metrics: dict):
        """Appends a concise log entry to today's journal note."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        journal_path = JOURNALS_DIR / f"{today_str}.md"
        timestamp = datetime.now().strftime("%H:%M:%S")

        status = "SUCCESS" if metrics.get("success") else "FAILED"
        entry = f"- `{timestamp}`: Completed [[{experiment_id}]] testing [[{target_concept}]] (Result: `{status}`, Cosine Sim: `{metrics.get('cosine_sim', 0.0):.3f}`)\n"

        if not journal_path.exists():
            header = f"# Daily Journal: {today_str}\n\n## Autonomous Research Log\n"
            journal_path.write_text(header + entry, encoding="utf-8")
        else:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(entry)


if __name__ == "__main__":
    compiler = VaultCompiler()
    sample_metrics = {"cosine_sim": 0.892, "success": True}
    test_file = compiler.record_experiment(
        experiment_id="EXP-20260827-001",
        target_concept="Recursive Feedback",
        source_layer=12,
        receiver_layer=12,
        metrics=sample_metrics,
        findings_summary="Latent trajectory successfully induced semantic classification in Receiver without intermediate text tokenization."
    )
    print(f"[Vault Compiler] Generated experiment note at: {test_file}")