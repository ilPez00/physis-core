#!/usr/bin/env python3
"""E61 — ask a local interpreter for the entity links the rules could not find.

The link layer is the measured bottleneck: with gold links the antecedent arm
reaches 0.698 on this repository's own prose, and with rule-extracted links it
reaches 0.209, below the 0.349 of using no links at all (E60). This script is
the first time a model enters the pipeline, and it enters at exactly one place:
naming what a sentence is about.

Two conditions, both pre-registered before the run:

    all        the interpreter extracts links for every sentence, replacing the
               rules entirely
    fallback   the interpreter is asked ONLY for sentences where the rules
               produced nothing (29 of 168), and its output is merged with the
               rules

`all` asks whether a model beats hand-written rules. `fallback` asks the cheaper
and more interesting question: whether a model is worth calling only where the
cheap path fails.

Determinism: temperature 0, one fixed prompt, model and prompt recorded in the
output. A local model at temperature 0 is deterministic in practice and not by
guarantee — re-running and diffing the two files is the check, and the scorer
reads the file rather than the model, so a scored number is always regenerable.

Usage:
    python3 research/e61_interpreter_extraction.py --input <rules.json> \
        --out-all <all.json> --out-fallback <fallback.json>
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://100.116.39.57:11434/v1/chat/completions"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

PROMPT = """List the things this sentence is about.

Rules:
- Output ONLY a comma-separated list, nothing else. No explanation, no numbering.
- Each item is one to three words, or a code identifier exactly as written.
- Name specific things (files, modules, functions, named concepts, named objects),
  not generic words like "result", "number", "thing", "problem".
- If the sentence names nothing specific, output NONE.

Sentence: {text}

Things:"""

GENERIC = {
    "result", "results", "number", "numbers", "thing", "things", "problem",
    "problems", "sentence", "text", "run", "value", "values", "case", "cases",
    "none", "n/a", "example", "examples", "work", "point", "way", "part",
}


def key(s: str) -> str:
    s = s.strip().strip("`\"'").lower()
    s = re.sub(r"[^a-z0-9\-_./: ]", "", s)
    return s.replace(" ", "-").strip("-")


def ask(text: str, retries: int = 3) -> list:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            "temperature": 0,
            "max_tokens": 120,
        }
    ).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            content = out["choices"][0]["message"].get("content") or ""
            # Some servers put the answer in reasoning_content and leave content
            # empty. That exact harness bug voided a whole C3 run once.
            if not content.strip():
                content = out["choices"][0]["message"].get("reasoning_content") or ""
            items = []
            for raw in content.replace("\n", ",").split(","):
                k = key(raw)
                if len(k) > 2 and k not in GENERIC and k not in items:
                    items.append(k)
            return items
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                print(f"  FAILED after {retries}: {e}", file=sys.stderr)
                return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-all", required=True)
    ap.add_argument("--out-fallback", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.input))
    all_rows, fb_rows = [], []
    empty = 0
    for i, r in enumerate(rows):
        links = ask(r["text"])
        all_rows.append({"pos": r["pos"], "links": links})
        if not r["rule_links"]:
            empty += 1
            fb_rows.append({"pos": r["pos"], "links": links})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr)

    meta = {"model": MODEL, "endpoint": ENDPOINT, "prompt": PROMPT}
    json.dump(all_rows, open(args.out_all, "w"), indent=1)
    json.dump(fb_rows, open(args.out_fallback, "w"), indent=1)
    json.dump(meta, open(args.out_all + ".meta", "w"), indent=1)
    got = sum(1 for r in all_rows if r["links"])
    print(f"{len(rows)} sentences, {got} with links; fallback set = {empty}")
    print(f"model {MODEL}")


if __name__ == "__main__":
    main()
