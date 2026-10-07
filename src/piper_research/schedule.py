"""
When research runs: windows of local time ("01:00-08:00"), optionally only on some days.

A window may cross midnight ("22:00-06:00"); it belongs to the day it STARTS on, so with
days ["fri"] that window runs Friday 22:00 to Saturday 06:00. "00:00-24:00" (or equal start and
end) is all day.
"""
from datetime import datetime, timedelta

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m):
        raise ValueError(f"bad time {hhmm!r} (use HH:MM)")
    return h * 60 + m


def parse_window(text: str) -> tuple[int, int]:
    """'01:00-08:00' -> (60, 480): minutes after midnight."""
    try:
        start, end = text.split("-")
        return _minutes(start), _minutes(end)
    except ValueError as e:
        raise ValueError(f"bad research window {text!r} (use HH:MM-HH:MM): {e}") from None


class Schedule:
    def __init__(self, windows, days="daily"):
        if isinstance(windows, str):                # from an environment variable: "a-b,c-d"
            windows = [w for w in windows.split(",") if w.strip()]
        self.windows = [parse_window(w) for w in windows]
        if days in (None, "daily", "all") or days == []:
            self.days = set(range(7))
        else:
            if isinstance(days, str):
                days = days.split(",")
            bad = [d for d in days if d.strip().lower()[:3] not in DAYS]
            if bad:
                raise ValueError(f"bad research days {bad} (use {', '.join(DAYS)})")
            self.days = {DAYS.index(d.strip().lower()[:3]) for d in days}
        self.text = ", ".join(windows) + ("" if len(self.days) == 7 else
                                          " on " + ", ".join(DAYS[d] for d in sorted(self.days)))

    def active(self, t: datetime) -> bool:
        """Is `t` (local time) inside a research window?"""
        minute = t.hour * 60 + t.minute
        today, yesterday = t.weekday(), (t - timedelta(days=1)).weekday()
        for start, end in self.windows:
            if start == end or (start == 0 and end == 24 * 60):
                if today in self.days:
                    return True
            elif start < end:
                if start <= minute < end and today in self.days:
                    return True
            else:                                   # crosses midnight
                if minute >= start and today in self.days:
                    return True
                if minute < end and yesterday in self.days:
                    return True
        return False

    def next_start(self, t: datetime) -> datetime | None:
        """The next time (minute resolution, within a week) research becomes active after `t`."""
        if not self.windows or not self.days:
            return None
        probe = t.replace(second=0, microsecond=0) + timedelta(minutes=1)
        was = self.active(t)
        for _ in range(7 * 24 * 60 + 1):
            now = self.active(probe)
            if now and not was:
                return probe
            was = now
            probe += timedelta(minutes=1)
        return None

    def __str__(self):
        return self.text
