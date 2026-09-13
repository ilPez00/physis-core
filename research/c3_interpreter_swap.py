#!/usr/bin/env python3
"""C3 — what survives an interpreter swap, measured on physis's own corpus.

Conceptual problem 2 rests on one borrowed measurement: the graphiti state-hash
experiment found that entities, edge topology and temporal boundaries came back
byte-identical from a 7B and a 120B interpreter while the *prose* differed --
and differed between two runs of the same model. If that holds, a ledger that
identifies claims by their sentence is keyed on the layer shown to be noise.

It was measured on graphiti's corpus, not on physis's. This runs the same test
here, so the premise of problem 2 is either confirmed on the data physis
actually holds or written down as weaker than stated.

## The construction

Each interpreter reads the same evidence document and returns four layers:

    L1  entities          the things named
    L2  + relations       (subject, relation, object) triples
    L3  + interval        when the claim is asserted to hold
    L4  + prose           a one-sentence rendering

Layers are hashed cumulatively, after normalisation -- lowercased, stripped,
sorted -- so that ordering and whitespace cannot manufacture a difference. A
layer is STABLE when both interpreters produce the same hash for every document.

## The control, and it is the important half

Two runs of the SAME model, same input, same settings. Without it a changed
layer says nothing: it could be the interpreter swap or it could be sampling.
Graphiti's control moved L4, which is what demoted prose from "interpreter
signature" to "noise". Any layer that moves under the control cannot be
evidence about the swap.

Usage:
    python3 research/c3_interpreter_swap.py [--docs N] [--out FILE]

Needs `OLLAMA_HOST` (default the fleet's omo box) and NVIDIA_API_KEY in
~/.env. Reports NOT RUN rather than inventing a number when either is absent.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "benchmarks" / "ground-truth"

LOCAL_BASE = os.environ.get("OLLAMA_HOST", "http://100.116.39.57:11434/v1")
LOCAL_MODEL = os.environ.get("C3_LOCAL_MODEL", "qwen2.5:7b-instruct-q4_K_M")
HOSTED_BASE = "https://integrate.api.nvidia.com/v1"
HOSTED_MODEL = os.environ.get("C3_HOSTED_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

SCHEMA = """Return ONLY a JSON object, no prose outside it, with exactly these keys:
{"entities": ["..."], "relations": [{"subject":"...","relation":"...","object":"..."}],
 "interval": {"from":"...","until":"..."}, "statement": "one sentence"}
Rules: entities are the named things; relations are factual triples drawn from the
text; interval is when the claim holds ("unknown" if the text does not say);
statement is a single sentence rendering of what the document asserts."""


def read_env_key(name):
    """~/.env does not source cleanly; grep the single line out."""
    p = Path.home() / ".env"
    if not p.exists():
        return None
    for line in p.read_text(errors="ignore").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def call(base, model, text, api_key=None, timeout=180):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SCHEMA},
            {"role": "user", "content": text},
        ],
        # Greedy. A layer that moves under sampling would say nothing about the
        # interpreter, and the control below exists to catch it if it still does.
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 800,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            # NIM's nemotron returns its answer in `reasoning_content` and
            # leaves `content` empty. Reading only `content` scored four of
            # eight documents as unparseable and charged the model for a
            # harness bug -- the first C3 run did exactly that.
            return msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:  # 503 under load is documented for NIM
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model}: {last}")


def parse(raw):
    """Pull the JSON object out of whatever the model wrapped it in."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm_list(xs):
    out = []
    for x in xs or []:
        if isinstance(x, str):
            s = " ".join(x.lower().split())
            if s:
                out.append(s)
    return sorted(set(out))


def norm_rels(rs):
    out = []
    for r in rs or []:
        if not isinstance(r, dict):
            continue
        t = tuple(
            " ".join(str(r.get(k, "")).lower().split()) for k in ("subject", "relation", "object")
        )
        if any(t):
            out.append("|".join(t))
    return sorted(set(out))


def layers(obj):
    """Cumulative layers. Each includes everything above it."""
    if obj is None:
        return None
    ents = norm_list(obj.get("entities"))
    rels = norm_rels(obj.get("relations"))
    iv = obj.get("interval") or {}
    interval = "|".join(
        " ".join(str(iv.get(k, "unknown")).lower().split()) for k in ("from", "until")
    )
    prose = " ".join(str(obj.get("statement", "")).lower().split())
    l1 = json.dumps(ents, sort_keys=True)
    l2 = l1 + json.dumps(rels, sort_keys=True)
    l3 = l2 + interval
    l4 = l3 + prose
    return {
        "L1 entities": l1,
        "L2 + topology": l2,
        "L3 + temporal": l3,
        "L4 + fact prose": l4,
        # Kept beside the hashes so the graded metric can be computed without
        # re-parsing. An exact hash says whether two runs agree; these say by
        # how much, which is the difference between "the interpreter changed
        # the facts" and "the interpreter changed the notation".
        "_sets": {"entities": ents, "relations": rels, "interval": [interval], "prose": [prose]},
    }


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def jaccard(a, b):
    """Graded agreement, because an exact hash is too brittle to be informative.

    The diagnostic on deploy-00 showed why. Entities: the 7B returned four and
    the 550B returned six, and all four were a strict subset of the six -- one
    interpreter is more complete, not in disagreement. Exact-hash scores that
    CHANGED and tells you nothing about the size of the change.

    Relations diverge harder (jaccard 0.12) and for a reason that is not
    evidential either: the 7B writes `rollout|verified by|release 17` and the
    550B writes `rollout|verified|true`. A relational convention against a
    boolean-object convention. Both assert that the rollout was verified.
    """
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--out", default="benchmarks/results/c3-interpreter-swap.json")
    a = ap.parse_args()

    key = read_env_key("NVIDIA_API_KEY")
    if not key:
        print("BIG MODEL LEG: NOT CONFIGURED — no NVIDIA_API_KEY. No claim made.")
        return 2

    docs = sorted(CORPUS.glob("*.md"))[: a.docs]
    if not docs:
        print(f"no corpus at {CORPUS}")
        return 2
    print(f"{len(docs)} documents · local {LOCAL_MODEL} · hosted {HOSTED_MODEL}\n")

    runs = {"A1": [], "A2": [], "B1": []}
    failures = []
    for d in docs:
        text = d.read_text(errors="ignore").strip()
        for tag, (base, model, k) in {
            "A1": (LOCAL_BASE, LOCAL_MODEL, None),
            "A2": (LOCAL_BASE, LOCAL_MODEL, None),
            "B1": (HOSTED_BASE, HOSTED_MODEL, key),
        }.items():
            try:
                obj = parse(call(base, model, text, k))
            except Exception as e:
                obj = None
                failures.append(f"{d.name} {tag}: {e}")
            runs[tag].append(layers(obj))
        got = sum(1 for t in runs if runs[t][-1] is not None)
        print(f"  {d.name:<28} {got}/3 parsed")

    # A document only counts where every arm returned parseable JSON. Comparing
    # a hash against a None would report a difference the interpreters did not
    # make.
    usable = [
        i for i in range(len(docs)) if all(runs[t][i] is not None for t in ("A1", "A2", "B1"))
    ]
    print(f"\n{len(usable)} of {len(docs)} documents usable in every arm")
    if not usable:
        print("nothing to compare — NOT RUN")
        return 2

    names = ["L1 entities", "L2 + topology", "L3 + temporal", "L4 + fact prose"]
    set_of = {
        "L1 entities": "entities",
        "L2 + topology": "relations",
        "L3 + temporal": "interval",
        "L4 + fact prose": "prose",
    }
    result = {
        "documents": len(docs),
        "usable": len(usable),
        "local_model": LOCAL_MODEL,
        "hosted_model": HOSTED_MODEL,
        "failures": failures,
        "layers": [],
    }
    print("\n  layer            swap identical   control identical   swap overlap")
    for n in names:
        swap = sum(1 for i in usable if h(runs["A1"][i][n]) == h(runs["B1"][i][n]))
        ctrl = sum(1 for i in usable if h(runs["A1"][i][n]) == h(runs["A2"][i][n]))
        u = len(usable)
        k = set_of[n]
        js = [jaccard(runs["A1"][i]["_sets"][k], runs["B1"][i]["_sets"][k]) for i in usable]
        jmean = sum(js) / max(1, len(js))
        # A layer is only evidence about the swap where the control holds it
        # fixed. Graphiti's L4 moved under the control, which is what demoted
        # prose from signature to noise.
        if ctrl < u:
            verdict = "UNUSABLE — moves under the control"
        elif swap == u:
            verdict = "STABLE across interpreters"
        else:
            verdict = "CHANGED by the interpreter"
        print(f"  {n:<16} {swap}/{u}              {ctrl}/{u}                 {jmean:.2f}   {verdict}")
        result["layers"].append(
            {"layer": n, "swap_identical": swap, "control_identical": ctrl, "of": u,
             "swap_jaccard_mean": round(jmean, 3), "verdict": verdict}
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nartifact -> {out}")
    if failures:
        print(f"{len(failures)} call/parse failure(s); first: {failures[0][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
