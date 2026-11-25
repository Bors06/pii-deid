"""Evaluate the de-identification pipeline against a hand-labeled gold file.

Gold file format (CSV): row_id,text,pii
  pii = the PII strings in that row, separated by "|"; empty = clean row.

Usage:
  python eval.py                      # rules only, with roster
  python eval.py --llm                # rules only AND rules + LLM
  python eval.py --no-roster          # pretend there is no roster.csv
  python eval.py --gold data/real_gold.csv
  python eval.py --ner both           # compare spaCy vs GLiNER for finding names
"""
import argparse
import csv
import re
import time
from pathlib import Path

import deid
from llm_check import check_ollama, llm_find_pii, mask_findings

DEFAULT_GOLD = Path(__file__).parent / "examples" / "eval_gold.csv"
PLACEHOLDER = re.compile(r"<[A-Z0-9_]+>")


def load_gold(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            (r["row_id"], r["text"], [p.strip() for p in (r["pii"] or "").split("|") if p.strip()])
            for r in csv.DictReader(f)
        ]


def evaluate(rows, use_llm):
    llm_label = deid.make_operator("LLM_PII") if use_llm else None
    total_pii = caught = clean_rows = clean_changed = 0
    leaks, damaged, possible_overmask = [], [], []
    start = time.time()

    for row_id, text, gold in rows:
        out = deid.deidentify(text)
        if use_llm:
            out, _ = mask_findings(out, llm_find_pii(out), llm_label)

        # 1. Catch rate: a gold item still visible ANYWHERE (case-insensitive) = leak.
        #    Substring on purpose: "nakamura" surviving inside "nakamuraSolution" is a leak.
        for item in gold:
            total_pii += 1
            if item.lower() in out.lower():
                leaks.append((row_id, item, out))
            else:
                caught += 1

        # 2. Clean rows that got changed (over-masking)
        if not gold:
            clean_rows += 1
            if out != text:
                clean_changed += 1
                damaged.append((row_id, out))
        # 3. Rows with PII but more placeholders than expected (worth a manual look)
        else:
            expected = sum(text.lower().count(item.lower()) for item in gold)
            if len(PLACEHOLDER.findall(out)) > expected:
                possible_overmask.append((row_id, out))

    return {
        "recall": caught / total_pii if total_pii else 1.0,
        "caught": caught, "total_pii": total_pii,
        "clean_changed": clean_changed, "clean_rows": clean_rows,
        "leaks": leaks, "damaged": damaged, "possible_overmask": possible_overmask,
        "seconds": time.time() - start,
    }


def report(name, r):
    print(f"\n=== {name} ===")
    print(f"Catch rate (recall): {r['recall']:.0%}  ({r['caught']}/{r['total_pii']} PII items masked)")
    print(f"Clean rows changed:  {r['clean_changed']}/{r['clean_rows']}")
    print(f"Time:                {r['seconds']:.1f}s")
    for row_id, item, out in r["leaks"]:
        print(f"  LEAK      row {row_id}: {item!r}  ->  {out}")
    for row_id, out in r["damaged"]:
        print(f"  DAMAGED   row {row_id}: {out}")
    for row_id, out in r["possible_overmask"]:
        print(f"  CHECK     row {row_id}: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate de-identification against a gold file.")
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="gold CSV (default: examples/eval_gold.csv)")
    parser.add_argument("--llm", action="store_true", help="also evaluate rules + LLM")
    parser.add_argument("--no-roster", action="store_true", help="ignore roster.csv")
    parser.add_argument("--ner", choices=["spacy", "gliner", "both"], default="spacy",
                        help="model used to find names; 'both' compares them (default: spacy)")
    args = parser.parse_args()

    rows = load_gold(args.gold)
    netids = deid.harvest_netids(text for _, text, _ in rows)
    info = deid.configure(use_roster=not args.no_roster, extra_netids=netids, persist=False)
    roster_note = "no roster" if args.no_roster else f"roster: {info['roster_students']} students"
    print(f"{len(rows)} labeled rows from {Path(args.gold).name} ({roster_note}, "
          f"{info['netids_from_data']} UCInetIDs found in data)")

    if args.llm:
        check_ollama()
    for ner in (["spacy", "gliner"] if args.ner == "both" else [args.ner]):
        deid.set_ner(ner)
        deid.deidentify("warm-up so model loading isn't counted in the timing")
        report(f"Rules only [{ner}]", evaluate(rows, use_llm=False))
        if args.llm:
            report(f"Rules + LLM [{ner}]", evaluate(rows, use_llm=True))
