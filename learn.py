"""
learn.py — Pattern learning from journal entries.

Scans journal/*.json, scores existing playbook patterns, finds condition
correlations, and suggests new pattern candidates.

Usage:
    python learn.py                    # terminal report only
    python learn.py --yaml             # also write playbook_suggestions.yaml
    python learn.py --min-n 3          # require at least 3 observations
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

from playbook import load_playbook, _evaluate_condition, _enrich_brief

JOURNAL_DIR = Path(__file__).parent / "journal"


# ─── Numeric buckets (domain-aware thresholds) ──────────────────────────────

NUMERIC_BUCKETS = {
    "gap_from_pdc": [
        ("large_gap_down", "lt", -50),
        ("gap_down", "lt", 0),
        ("gap_up", "gt", 0),
        ("large_gap_up", "gt", 50),
    ],
    "vix_price": [
        ("vix_low", "lt", 15),
        ("vix_normal", "range", [15, 20]),
        ("vix_elevated", "range", [20, 25]),
        ("vix_high", "gt", 25),
    ],
    "premarket_range": [
        ("range_tight", "lt", 80),
        ("range_normal", "range", [80, 140]),
        ("range_wide", "gt", 140),
    ],
    "dist_to_inv": [
        ("inv_near", "lt", 40),
        ("inv_medium", "range", [40, 80]),
        ("inv_far", "gt", 80),
    ],
}

# Fields to analyze
ENUM_FIELDS = ["vix_regime", "regime", "premarket_bias", "trade_state", "mode"]
BOOL_FIELDS = ["inside_inventory", "has_major_earnings", "runner_permission"]
NUMERIC_FIELDS = list(NUMERIC_BUCKETS.keys())


# ─── Data loading ───────────────────────────────────────────────────────────

def load_all_journals(journal_dir=JOURNAL_DIR):
    """Load all journal JSON files, sorted by date ascending."""
    if not journal_dir.exists():
        return []

    entries = []
    for f in sorted(journal_dir.glob("*.json")):
        try:
            with open(f) as fh:
                entry = json.load(fh)
            if "date" in entry and "brief" in entry and "outcome" in entry:
                entries.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue

    return entries


# ─── Data quality ───────────────────────────────────────────────────────────

def _is_sane_outcome(outcome):
    """Basic sanity check — reject obvious data errors."""
    day_range = outcome.get("day_range")
    if day_range is not None and day_range > 2000:
        return False
    mae = outcome.get("mae_from_orb_break")
    if mae is not None and mae > 1000:
        return False
    orb_low = outcome.get("orb_low")
    orb_high = outcome.get("orb_high")
    if orb_low is not None and orb_high is not None:
        if orb_high - orb_low > 2000 or orb_low < 1000:
            return False
    return True


# ─── Outcome scoring ───────────────────────────────────────────────────────

def compute_outcome_score(outcome, brief):
    """
    Composite score 0.0–1.0.
    T1 hit: +0.4, MFE ratio: +0.3, MAE containment: +0.2, Runner: +0.1.
    Returns None if outcome is unusable.
    """
    if outcome.get("orb_break_dir") is None:
        return None

    target = brief.get("suggested_target", 80)
    mfe = outcome.get("mfe_from_orb_break") or 0
    mae = outcome.get("mae_from_orb_break") or 0

    score = 0.0

    if outcome.get("t1_hit"):
        score += 0.4

    if target > 0:
        mfe_ratio = min(mfe / target, 2.0) / 2.0
        score += 0.3 * mfe_ratio
        mae_ratio = max(0, 1 - mae / target)
        score += 0.2 * mae_ratio

    if outcome.get("runner_hit"):
        score += 0.1

    return round(min(score, 1.0), 3)


# ─── Entry enrichment ──────────────────────────────────────────────────────

def enrich_journal_entry(entry):
    """Add outcome_score, enriched_brief, and discretized_fields to a copy."""
    e = dict(entry)
    brief = e.get("brief", {})
    outcome = e.get("outcome", {})

    e["outcome_score"] = compute_outcome_score(outcome, brief)
    e["enriched_brief"] = _enrich_brief(brief)

    # Discretize numeric fields
    disc = {}
    enriched = e["enriched_brief"]
    for field, buckets in NUMERIC_BUCKETS.items():
        val = enriched.get(field)
        if val is None:
            continue
        for bucket_name, op, threshold in buckets:
            cond = {"field": field, "op": op, "value": threshold}
            if _evaluate_condition(cond, enriched):
                disc.setdefault(field, []).append(bucket_name)
    e["discretized"] = disc

    return e


# ─── Confidence & grading ──────────────────────────────────────────────────

def confidence_label(n):
    if n == 0:
        return "NO DATA"
    elif n <= 2:
        return "ANECDOTAL"
    elif n <= 5:
        return "EARLY"
    elif n <= 10:
        return "DEVELOPING"
    else:
        return "SOLID"


def grade_pattern(t1_rate, avg_score):
    if t1_rate >= 0.8 and avg_score >= 0.7:
        return "A"
    elif t1_rate >= 0.6 and avg_score >= 0.5:
        return "B"
    elif t1_rate >= 0.4 and avg_score >= 0.35:
        return "C"
    elif t1_rate >= 0.2:
        return "D"
    else:
        return "F"


# ─── Section 1: Pattern Scorecard ──────────────────────────────────────────

def compute_pattern_scorecard(entries, playbook):
    """Grade each existing playbook pattern using journal outcome data."""
    patterns = playbook.get("patterns", {})
    scorecard = []

    for pid, pdef in patterns.items():
        # Find entries where this pattern matched
        matching = []
        for e in entries:
            matched_ids = e.get("matched_patterns", [])
            if pid in matched_ids:
                matching.append(e)

        n = len(matching)
        if n == 0:
            scorecard.append({
                "pattern_id": pid,
                "name": pdef.get("name", pid),
                "n": 0,
                "t1_count": 0, "t1_rate": 0,
                "runner_count": 0, "runner_rate": 0,
                "avg_mfe": None, "avg_mae": None,
                "avg_day_range": None, "avg_score": None,
                "grade": "-", "confidence": "NO DATA",
            })
            continue

        t1_count = sum(1 for e in matching if e["outcome"].get("t1_hit"))
        runner_count = sum(1 for e in matching if e["outcome"].get("runner_hit"))
        t1_rate = t1_count / n
        runner_rate = runner_count / n

        mfes = [e["outcome"]["mfe_from_orb_break"] for e in matching
                if e["outcome"].get("mfe_from_orb_break") is not None]
        maes = [e["outcome"]["mae_from_orb_break"] for e in matching
                if e["outcome"].get("mae_from_orb_break") is not None]
        ranges = [e["outcome"]["day_range"] for e in matching
                  if e["outcome"].get("day_range") is not None]
        scores = [e["outcome_score"] for e in matching
                  if e.get("outcome_score") is not None]

        avg_mfe = round(sum(mfes) / len(mfes), 1) if mfes else None
        avg_mae = round(sum(maes) / len(maes), 1) if maes else None
        avg_range = round(sum(ranges) / len(ranges), 1) if ranges else None
        avg_score = round(sum(scores) / len(scores), 3) if scores else None

        g = grade_pattern(t1_rate, avg_score) if avg_score is not None else "-"

        scorecard.append({
            "pattern_id": pid,
            "name": pdef.get("name", pid),
            "n": n,
            "t1_count": t1_count, "t1_rate": round(t1_rate, 2),
            "runner_count": runner_count, "runner_rate": round(runner_rate, 2),
            "avg_mfe": avg_mfe, "avg_mae": avg_mae,
            "avg_day_range": avg_range, "avg_score": avg_score,
            "grade": g, "confidence": confidence_label(n),
        })

    # Sort: patterns with data first (by avg_score desc), then no-data
    scorecard.sort(key=lambda s: (s["n"] == 0, -(s["avg_score"] or 0)))
    return scorecard


# ─── Dashboard fast-path loaders ──────────────────────────────────────────

def load_scorecard_cache():
    """Load all journals, enrich, compute scorecard. Returns dict keyed by pattern_id."""
    result = load_live_intelligence()
    return result["scorecard"]


def load_live_intelligence():
    """Load journals once, compute scorecard + decay warnings. Returns dict with both."""
    entries = load_all_journals()
    if not entries:
        return {"scorecard": {}, "decay_warnings": []}
    enriched = [enrich_journal_entry(e) for e in entries if _is_sane_outcome(e.get("outcome", {}))]
    usable = [e for e in enriched if e.get("outcome_score") is not None]
    if not usable:
        return {"scorecard": {}, "decay_warnings": []}
    playbook = load_playbook()
    scorecard = compute_pattern_scorecard(usable, playbook)
    decay_warnings = compute_pattern_decay(usable, playbook)
    return {
        "scorecard": {s["pattern_id"]: s for s in scorecard},
        "decay_warnings": decay_warnings,
    }


def compute_pattern_decay(entries, playbook, recent_n=5):
    """Compare recent vs all-time hit rates. Returns list of decay warnings."""
    patterns = playbook.get("patterns", {})
    warnings = []

    for pid in patterns:
        # All matches for this pattern
        all_matches = [e for e in entries if pid in e.get("matched_patterns", [])]
        if len(all_matches) < recent_n + 2:  # need enough data for meaningful split
            continue

        all_t1 = sum(1 for e in all_matches if e["outcome"].get("t1_hit")) / len(all_matches)

        recent = all_matches[-recent_n:]
        recent_t1 = sum(1 for e in recent if e["outcome"].get("t1_hit")) / len(recent)

        if all_t1 > 0 and recent_t1 < all_t1 * 0.6:
            warnings.append({
                "pattern_id": pid,
                "name": patterns[pid].get("name", pid),
                "all_time_t1": round(all_t1, 2),
                "recent_t1": round(recent_t1, 2),
                "recent_n": recent_n,
                "total_n": len(all_matches),
            })

    return warnings


# ─── Section 2: Condition Correlations ──────────────────────────────────────

def compute_condition_correlations(entries):
    """Find which brief conditions best predict good outcomes."""
    results = []

    for e in entries:
        if e.get("outcome_score") is None:
            continue

    # Enum fields
    for field in ENUM_FIELDS:
        values = set()
        for e in entries:
            v = e["enriched_brief"].get(field)
            if v is not None:
                values.add(v)

        for val in values:
            true_group = [e for e in entries if e["enriched_brief"].get(field) == val
                          and e.get("outcome_score") is not None]
            false_group = [e for e in entries if e["enriched_brief"].get(field) != val
                           and e.get("outcome_score") is not None]
            if not true_group:
                continue

            true_avg = sum(e["outcome_score"] for e in true_group) / len(true_group)
            false_avg = (sum(e["outcome_score"] for e in false_group) / len(false_group)
                         if false_group else None)
            delta = (true_avg - false_avg) if false_avg is not None else None

            results.append({
                "condition": f"{field}={val}",
                "field": field, "op": "eq", "value": val,
                "true_n": len(true_group), "true_avg": round(true_avg, 3),
                "false_n": len(false_group),
                "false_avg": round(false_avg, 3) if false_avg is not None else None,
                "delta": round(delta, 3) if delta is not None else None,
            })

    # Boolean fields
    for field in BOOL_FIELDS:
        true_group = [e for e in entries if e["enriched_brief"].get(field) is True
                      and e.get("outcome_score") is not None]
        false_group = [e for e in entries if e["enriched_brief"].get(field) is False
                       and e.get("outcome_score") is not None]
        if not true_group and not false_group:
            continue

        for val, group, other in [(True, true_group, false_group),
                                  (False, false_group, true_group)]:
            if not group:
                continue
            g_avg = sum(e["outcome_score"] for e in group) / len(group)
            o_avg = (sum(e["outcome_score"] for e in other) / len(other)
                     if other else None)
            delta = (g_avg - o_avg) if o_avg is not None else None

            results.append({
                "condition": f"{field}={val}",
                "field": field, "op": "bool", "value": val,
                "true_n": len(group), "true_avg": round(g_avg, 3),
                "false_n": len(other),
                "false_avg": round(o_avg, 3) if o_avg is not None else None,
                "delta": round(delta, 3) if delta is not None else None,
            })

    # Numeric fields (bucketed)
    for field, buckets in NUMERIC_BUCKETS.items():
        for bucket_name, op, threshold in buckets:
            cond = {"field": field, "op": op, "value": threshold}
            true_group = [e for e in entries
                          if _evaluate_condition(cond, e["enriched_brief"])
                          and e.get("outcome_score") is not None]
            false_group = [e for e in entries
                           if not _evaluate_condition(cond, e["enriched_brief"])
                           and e.get("outcome_score") is not None]
            if not true_group:
                continue

            g_avg = sum(e["outcome_score"] for e in true_group) / len(true_group)
            o_avg = (sum(e["outcome_score"] for e in false_group) / len(false_group)
                     if false_group else None)
            delta = (g_avg - o_avg) if o_avg is not None else None

            results.append({
                "condition": f"{field}:{bucket_name}",
                "field": field, "op": op, "value": threshold,
                "true_n": len(true_group), "true_avg": round(g_avg, 3),
                "false_n": len(false_group),
                "false_avg": round(o_avg, 3) if o_avg is not None else None,
                "delta": round(delta, 3) if delta is not None else None,
            })

    # Sort by abs(delta) descending, None deltas last
    results.sort(key=lambda r: (r["delta"] is None, -abs(r["delta"] or 0)))
    return results


# ─── Section 3: Pattern Discovery ──────────────────────────────────────────

def build_candidate_conditions(entries):
    """Build all testable condition atoms from observed brief values."""
    candidates = []

    # Enum: one condition per observed value
    for field in ENUM_FIELDS:
        values = set()
        for e in entries:
            v = e["enriched_brief"].get(field)
            if v is not None:
                values.add(v)
        for val in values:
            candidates.append({"field": field, "op": "eq", "value": val})

    # Boolean: condition for True
    for field in BOOL_FIELDS:
        candidates.append({"field": field, "op": "bool", "value": True})
        candidates.append({"field": field, "op": "bool", "value": False})

    # Numeric: one condition per bucket
    for field, buckets in NUMERIC_BUCKETS.items():
        for bucket_name, op, threshold in buckets:
            candidates.append({
                "field": field, "op": op, "value": threshold,
                "_label": bucket_name,
            })

    return candidates


def score_single_conditions(entries, candidates, min_n=1):
    """Score each candidate condition by outcome quality of matching entries."""
    scored = []

    for cand in candidates:
        matching = [e for e in entries
                    if _evaluate_condition(cand, e["enriched_brief"])
                    and e.get("outcome_score") is not None]
        if len(matching) < min_n:
            continue

        avg_score = sum(e["outcome_score"] for e in matching) / len(matching)
        t1_count = sum(1 for e in matching if e["outcome"].get("t1_hit"))
        t1_rate = t1_count / len(matching)

        if avg_score < 0.4:
            continue

        scored.append({
            "condition": cand,
            "n": len(matching),
            "avg_score": round(avg_score, 3),
            "t1_rate": round(t1_rate, 2),
            "t1_count": t1_count,
        })

    scored.sort(key=lambda s: -s["avg_score"])
    return scored[:15]


def _condition_overlap(candidate_conds, existing_conds):
    """Jaccard similarity between two condition sets."""
    c_set = {(c["field"], str(c.get("value", ""))) for c in candidate_conds}
    e_set = {(c["field"], str(c.get("value", ""))) for c in existing_conds}
    if not c_set or not e_set:
        return 0.0
    return len(c_set & e_set) / len(c_set | e_set)


def discover_pattern_candidates(entries, top_conditions, existing_patterns, min_n=1):
    """Combine top single conditions into 2-3 condition pattern candidates."""
    if not top_conditions:
        return []

    usable = [e for e in entries if e.get("outcome_score") is not None]
    candidates = []

    # 2-condition combos
    for a, b in combinations(top_conditions, 2):
        ca, cb = a["condition"], b["condition"]
        if ca["field"] == cb["field"]:
            continue

        matching = [e for e in usable
                    if _evaluate_condition(ca, e["enriched_brief"])
                    and _evaluate_condition(cb, e["enriched_brief"])]
        if len(matching) < min_n:
            continue

        avg_score = sum(e["outcome_score"] for e in matching) / len(matching)
        t1_count = sum(1 for e in matching if e["outcome"].get("t1_hit"))
        t1_rate = t1_count / len(matching)

        if t1_rate < 0.5 and avg_score < 0.5:
            continue

        mfes = [e["outcome"]["mfe_from_orb_break"] for e in matching
                if e["outcome"].get("mfe_from_orb_break") is not None]
        avg_mfe = round(sum(mfes) / len(mfes), 1) if mfes else None

        conds = [ca, cb]
        # Check overlap with existing patterns
        best_overlap = 0.0
        similar_to = None
        for pid, pdef in existing_patterns.items():
            ov = _condition_overlap(conds, pdef.get("conditions", []))
            if ov > best_overlap:
                best_overlap = ov
                similar_to = pid

        rank_score = avg_score * math.log2(len(matching) + 1)

        candidates.append({
            "conditions": conds,
            "n_matching": len(matching),
            "matching_dates": [e["date"] for e in matching],
            "t1_rate": round(t1_rate, 2),
            "t1_count": t1_count,
            "avg_mfe": avg_mfe,
            "avg_outcome_score": round(avg_score, 3),
            "rank_score": round(rank_score, 3),
            "similar_to": similar_to if best_overlap > 0.6 else None,
            "overlap_score": round(best_overlap, 2),
        })

    # 3-condition combos (only if enough data)
    if len(usable) >= 5 and len(top_conditions) >= 3:
        for a, b, c in combinations(top_conditions[:10], 3):
            ca, cb, cc = a["condition"], b["condition"], c["condition"]
            fields = {ca["field"], cb["field"], cc["field"]}
            if len(fields) < 2:
                continue

            matching = [e for e in usable
                        if _evaluate_condition(ca, e["enriched_brief"])
                        and _evaluate_condition(cb, e["enriched_brief"])
                        and _evaluate_condition(cc, e["enriched_brief"])]
            if len(matching) < min_n:
                continue

            avg_score = sum(e["outcome_score"] for e in matching) / len(matching)
            t1_count = sum(1 for e in matching if e["outcome"].get("t1_hit"))
            t1_rate = t1_count / len(matching)

            if t1_rate < 0.5 and avg_score < 0.5:
                continue

            mfes = [e["outcome"]["mfe_from_orb_break"] for e in matching
                    if e["outcome"].get("mfe_from_orb_break") is not None]
            avg_mfe = round(sum(mfes) / len(mfes), 1) if mfes else None

            conds = [ca, cb, cc]
            best_overlap = 0.0
            similar_to = None
            for pid, pdef in existing_patterns.items():
                ov = _condition_overlap(conds, pdef.get("conditions", []))
                if ov > best_overlap:
                    best_overlap = ov
                    similar_to = pid

            rank_score = avg_score * math.log2(len(matching) + 1)

            candidates.append({
                "conditions": conds,
                "n_matching": len(matching),
                "matching_dates": [e["date"] for e in matching],
                "t1_rate": round(t1_rate, 2),
                "t1_count": t1_count,
                "avg_mfe": avg_mfe,
                "avg_outcome_score": round(avg_score, 3),
                "rank_score": round(rank_score, 3),
                "similar_to": similar_to if best_overlap > 0.6 else None,
                "overlap_score": round(best_overlap, 2),
            })

    candidates.sort(key=lambda c: -c["rank_score"])
    return candidates[:5]


# ─── YAML generation ───────────────────────────────────────────────────────

def generate_pattern_id(conditions):
    """Generate a snake_case pattern ID from conditions."""
    parts = ["suggested"]
    for c in conditions:
        field = c["field"].replace("_", "")
        val = str(c.get("value", "")).replace(" ", "").lower()
        # Shorten common prefixes
        val = val.replace("[", "").replace("]", "").replace("'", "").replace(",", "_")
        parts.append(f"{field}_{val}"[:20])
    return "_".join(parts)[:60]


def format_as_playbook_yaml(candidates):
    """Convert candidates into valid playbook.yaml format."""
    output = {"patterns": {}}

    for i, cand in enumerate(candidates):
        pid = generate_pattern_id(cand["conditions"])
        # Deduplicate IDs
        if pid in output["patterns"]:
            pid = f"{pid}_{i}"

        # Clean conditions for YAML (remove _label)
        clean_conds = []
        for c in cand["conditions"]:
            clean = {k: v for k, v in c.items() if not k.startswith("_")}
            clean_conds.append(clean)

        # Human-readable name
        name_parts = []
        for c in cand["conditions"]:
            label = c.get("_label", f"{c['field']}={c.get('value', '')}")
            name_parts.append(str(label).replace("_", " ").title())
        name = " + ".join(name_parts)

        n = cand["n_matching"]
        t1 = cand["t1_count"]
        dates = ", ".join(cand["matching_dates"])

        output["patterns"][pid] = {
            "name": name,
            "priority": "MEDIUM",
            "conditions": clean_conds,
            "bias": None,
            "bias_override": False,
            "warning": None,
            "action": f"REVIEW: discovered from {n} entries, {t1}/{n} T1 hit",
            "notes": f"Auto-discovered. Dates: {dates}",
        }

    return yaml.dump(output, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ─── Terminal formatting ────────────────────────────────────────────────────

def print_report(scorecard, correlations, candidates, total_entries):
    """Print the full 3-section report."""
    print_scorecard_section(scorecard)
    print_correlations_section(correlations, total_entries)
    print_suggestions_section(candidates)


def print_scorecard_section(scorecard):
    """Print pattern scorecard table."""
    print(f"\n  ── PATTERN SCORECARD ────────────────────────────────────────")
    print(f"  {'Pattern':<36s} {'N':>3s}  {'T1':>5s}  {'MFE':>7s}  {'MAE':>7s}  "
          f"{'Grade':>5s}  {'Confidence'}")
    print(f"  {'─' * 36} {'─' * 3}  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 11}")

    for s in scorecard:
        if s["n"] == 0:
            print(f"  {s['name']:<36s} {s['n']:>3d}  {'---':>5s}  {'---':>7s}  "
                  f"{'---':>7s}  {s['grade']:>5s}  {s['confidence']}")
        else:
            t1_str = f"{s['t1_count']}/{s['n']}"
            mfe_str = f"{s['avg_mfe']:.1f}" if s['avg_mfe'] is not None else "---"
            mae_str = f"{s['avg_mae']:.1f}" if s['avg_mae'] is not None else "---"
            print(f"  {s['name']:<36s} {s['n']:>3d}  {t1_str:>5s}  {mfe_str:>7s}  "
                  f"{mae_str:>7s}  {s['grade']:>5s}  {s['confidence']}")

    print()


def print_correlations_section(correlations, total_entries):
    """Print condition correlations."""
    print(f"  ── CONDITION CORRELATIONS ───────────────────────────────────")

    if total_entries < 2:
        print(f"  Need at least 2 journal entries for meaningful correlations.")
        print(f"  Current: {total_entries}")
        print()
        return

    # Show top 12 by delta
    shown = 0
    print(f"  {'Condition':<30s} {'When TRUE':>12s}  {'When FALSE':>12s}  {'Delta':>7s}")
    print(f"  {'─' * 30} {'─' * 12}  {'─' * 12}  {'─' * 7}")

    for r in correlations:
        if shown >= 12:
            break
        true_str = f"{r['true_avg']:.3f} (N={r['true_n']})"
        if r["false_avg"] is not None:
            false_str = f"{r['false_avg']:.3f} (N={r['false_n']})"
        else:
            false_str = f"--- (N={r['false_n']})"
        delta_str = f"{r['delta']:+.3f}" if r["delta"] is not None else "---"

        print(f"  {r['condition']:<30s} {true_str:>12s}  {false_str:>12s}  {delta_str:>7s}")
        shown += 1

    print()


def print_suggestions_section(candidates):
    """Print pattern suggestions with YAML blocks."""
    print(f"  ── PATTERN SUGGESTIONS ──────────────────────────────────────")

    if not candidates:
        print(f"  No pattern candidates found.")
        print(f"  Continue journaling -- suggestions improve at N >= 3.")
        print()
        return

    for i, cand in enumerate(candidates, 1):
        n = cand["n_matching"]
        t1 = cand["t1_count"]
        score = cand["avg_outcome_score"]

        print(f"\n  [{i}] N={n}, T1: {t1}/{n}, score: {score:.3f}")

        # List conditions
        for c in cand["conditions"]:
            label = c.get("_label", "")
            if label:
                print(f"      - {c['field']} {c['op']} {c['value']}  ({label})")
            else:
                print(f"      - {c['field']} {c['op']} {c['value']}")

        if cand.get("similar_to"):
            print(f"      NOTE: Similar to '{cand['similar_to']}' "
                  f"(overlap: {cand['overlap_score']:.2f})")

        print(f"      Dates: {', '.join(cand['matching_dates'])}")

    # Print YAML block for all suggestions
    if candidates:
        yaml_str = format_as_playbook_yaml(candidates)
        print(f"\n  Ready-to-paste YAML:")
        print(f"  {'─' * 56}")
        for line in yaml_str.split("\n"):
            print(f"  {line}")

    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Learn patterns from journal entries")
    parser.add_argument("--yaml", action="store_true",
                        help="Write suggested patterns to playbook_suggestions.yaml")
    parser.add_argument("--min-n", type=int, default=1,
                        help="Min journal entries matching a condition (default: 1)")
    args = parser.parse_args()

    # 1. Load data
    entries = load_all_journals()
    if not entries:
        print("\n  No journal entries found in journal/")
        print("  Run: python journal.py --date YYYY-MM-DD")
        sys.exit(1)

    playbook = load_playbook()

    print(f"\n  PATTERN LEARNING REPORT")
    print(f"  ════════════════════════════════════════════════════════════")
    print(f"  Journal entries: {len(entries)}")
    print(f"  Date range:      {entries[0]['date']} to {entries[-1]['date']}")
    print(f"  Playbook patterns: {len(playbook.get('patterns', {}))}")

    # 2. Enrich + filter
    enriched = []
    excluded = 0
    for e in entries:
        if not _is_sane_outcome(e.get("outcome", {})):
            print(f"  [WARN] Excluding {e['date']}: data quality issue")
            excluded += 1
            continue
        enriched.append(enrich_journal_entry(e))

    # Filter to usable (has outcome_score)
    usable = [e for e in enriched if e.get("outcome_score") is not None]

    if excluded:
        print(f"  Excluded: {excluded} (data quality)")
    print(f"  Usable entries: {len(usable)}")

    if not usable:
        print("\n  No usable entries (need ORB break data).")
        sys.exit(1)

    # 3. Analyses
    scorecard = compute_pattern_scorecard(usable, playbook)
    correlations = compute_condition_correlations(usable)
    cand_atoms = build_candidate_conditions(usable)
    top_singles = score_single_conditions(usable, cand_atoms, min_n=args.min_n)
    suggestions = discover_pattern_candidates(
        usable, top_singles, playbook.get("patterns", {}),
        min_n=args.min_n,
    )

    # 4. Print report
    print_report(scorecard, correlations, suggestions, len(usable))

    # 5. Optional YAML output
    if args.yaml and suggestions:
        yaml_str = format_as_playbook_yaml(suggestions)
        out_path = Path(__file__).parent / "playbook_suggestions.yaml"
        with open(out_path, "w") as f:
            f.write(yaml_str)
        print(f"  Suggestions written to {out_path}")


if __name__ == "__main__":
    main()
