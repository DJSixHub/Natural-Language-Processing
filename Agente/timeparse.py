from __future__ import annotations

import re


def parse_minutes(tiempo_raw: str) -> int | None:
    """Parse strings like '30m', '1h', '1h 20m', '45m'. Returns minutes."""
    if not tiempo_raw:
        return None
    s = tiempo_raw.lower().strip()
    # normalize separators
    s = s.replace(" ", "")
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", s)
    if not m:
        return None
    hours = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    total = hours * 60 + mins
    return total if total > 0 else None
