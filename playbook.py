"""
playbook.py — Pattern matcher for pre-market brief.

Loads playbook.yaml, evaluates conditions against brief JSON fields,
returns sorted list of matched patterns.

No imports from dashboard.py (avoids circular imports).
"""

import os
from pathlib import Path

import yaml

# ─── YAML loading with mtime cache ──────────────────────────────────────────

_playbook_cache = {"mtime": 0, "data": None}
_DEFAULT_PATH = Path(__file__).parent / "playbook.yaml"

_PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def load_playbook(path=None):
    """Load playbook YAML, caching by file mtime. Returns {"patterns": {}} if missing."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {"patterns": {}}

    mtime = os.path.getmtime(p)
    if _playbook_cache["data"] is not None and _playbook_cache["mtime"] == mtime:
        return _playbook_cache["data"]

    with open(p) as f:
        data = yaml.safe_load(f) or {}

    if "patterns" not in data:
        data = {"patterns": {}}

    _playbook_cache["mtime"] = mtime
    _playbook_cache["data"] = data
    return data


# ─── Brief enrichment ───────────────────────────────────────────────────────

def _enrich_brief(brief_data):
    """Add derived fields to a copy of the brief for pattern matching."""
    b = dict(brief_data)

    # prior_day_range = PDH - PDL
    pdh = b.get("PDH")
    pdl = b.get("PDL")
    if pdh is not None and pdl is not None:
        b["prior_day_range"] = round(float(pdh) - float(pdl), 2)

    # gap_direction: up / down / flat
    gap = b.get("gap_from_pdc")
    if gap is not None:
        gap = float(gap)
        if gap > 20:
            b["gap_direction"] = "up"
        elif gap < -20:
            b["gap_direction"] = "down"
        else:
            b["gap_direction"] = "flat"

    return b


# ─── Condition evaluator ────────────────────────────────────────────────────

def _evaluate_condition(condition, brief_data):
    """Evaluate a single condition against brief data. Returns True/False."""
    field = condition.get("field")
    op = condition.get("op")
    expected = condition.get("value")

    actual = brief_data.get(field)
    if actual is None:
        return False

    try:
        if op == "eq":
            return actual == expected
        elif op == "gt":
            return float(actual) > float(expected)
        elif op == "gte":
            return float(actual) >= float(expected)
        elif op == "lt":
            return float(actual) < float(expected)
        elif op == "lte":
            return float(actual) <= float(expected)
        elif op == "in":
            return actual in expected
        elif op == "not_in":
            return actual not in expected
        elif op == "bool":
            return bool(actual) == bool(expected)
        elif op == "range":
            lo, hi = expected
            return float(lo) <= float(actual) <= float(hi)
        else:
            return False
    except (TypeError, ValueError):
        return False


# ─── Main matching logic ────────────────────────────────────────────────────

def match_patterns(brief_data, playbook_path=None):
    """
    Match brief data against playbook patterns.

    Returns list of match dicts sorted by: full_match > priority > match_ratio.
    Only patterns with >= 50% conditions met are returned.
    """
    playbook = load_playbook(playbook_path)
    patterns = playbook.get("patterns", {})
    if not patterns:
        return []

    enriched = _enrich_brief(brief_data)
    matches = []

    for pattern_id, pattern in patterns.items():
        conditions = pattern.get("conditions", [])
        if not conditions:
            continue

        met = sum(1 for c in conditions if _evaluate_condition(c, enriched))
        total = len(conditions)
        ratio = met / total

        if ratio < 0.5:
            continue

        matches.append({
            "pattern_id": pattern_id,
            "name": pattern.get("name", pattern_id),
            "priority": pattern.get("priority", "LOW"),
            "conditions_total": total,
            "conditions_met": met,
            "match_ratio": round(ratio, 2),
            "full_match": met == total,
            "bias": pattern.get("bias"),
            "bias_override": pattern.get("bias_override", False),
            "warning": pattern.get("warning"),
            "action": pattern.get("action"),
            "notes": pattern.get("notes"),
        })

    # Sort: full matches first, then by priority rank, then by match ratio desc
    matches.sort(key=lambda m: (
        not m["full_match"],
        _PRIORITY_RANK.get(m["priority"], 9),
        -m["match_ratio"],
    ))

    return matches


# ─── Terminal formatting ────────────────────────────────────────────────────

def format_playbook_matches(matches):
    """Format playbook matches for terminal output."""
    if not matches:
        return ""

    lines = []
    lines.append("")
    lines.append("  ── PLAYBOOK MATCHES ──────────────────────────────────────")

    for m in matches:
        full = "FULL" if m["full_match"] else "PARTIAL"
        lines.append(
            f"  [{m['priority']:6s}] {m['name']:40s} "
            f"{m['conditions_met']}/{m['conditions_total']} ({full})"
        )

        if m.get("warning"):
            lines.append(f"           ** {m['warning']}")
        if m.get("action"):
            lines.append(f"           -> {m['action']}")
        if m.get("bias") and m["bias_override"]:
            lines.append(f"           BIAS OVERRIDE: {m['bias']}")

    lines.append("  ──────────────────────────────────────────────────────────")
    lines.append("")

    return "\n".join(lines)
