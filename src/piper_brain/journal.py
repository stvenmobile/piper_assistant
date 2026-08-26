"""
Piper Journal: Date-grouped activity logging to daily_journal.md.
"""

from pathlib import Path
from datetime import datetime

JOURNAL_FILE = Path(__file__).resolve().parent.parent.parent / "daily_journal.md"

class ActivityJournal:
    def __init__(self, journal_path: Path = JOURNAL_FILE):
        self.journal_path = journal_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.journal_path.exists():
            self.journal_path.write_text("# Piper Activity Journal\n\n", encoding="utf-8")

    def log(self, category: str, message: str, details: str | None = None):
        """Appends a timestamped activity entry under today's date header."""
        today_header = f"## {datetime.now().strftime('%Y-%m-%d')}"
        timestamp = datetime.now().strftime("%H:%M:%S")

        content = self.journal_path.read_text(encoding="utf-8") if self.journal_path.exists() else ""

        entry = f"- `[{timestamp}]` **[{category.upper()}]** {message}\n"
        if details:
            entry += f"  - *Details*: {details}\n"

        if today_header in content:
            # Find the section and append below it
            sections = content.split(today_header)
            header_part = sections[0] + today_header + "\n"
            rest = sections[1]
            
            # Insert entry at the start of the current date's section
            updated_content = header_part + entry + rest.lstrip("\n")
        else:
            # Add new date header and first entry
            updated_content = content.rstrip() + f"\n\n{today_header}\n{entry}"

        self.journal_path.write_text(updated_content.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    journal = ActivityJournal()
    journal.log("STATE", "System transitioned to ENGAGED via wake-word.")
    journal.log("INTENT", "Processed local intent.", "Text: 'what time is it'")
    print("Journal entry written.")