#!/usr/bin/env python3
"""Can the direction control tell a compression from a deletion?

The control shipped broken (see src/direction.rs module header): validated on one
authored example, it inverted on the first real text — rating deletion above
genuine compression. This is the instrument that should have existed first.

## Why the arms are generated mechanically

Every arm below is produced by a RULE over a real source text, so the correct
verdict is objective and nobody authored the answer. That is the difference
between a benchmark and a demo, and the demo is what shipped.

  destop     every content word, no filler     HOLD — shorter, nothing dropped
  truncate   first N words                     FAIL — the null itself
  decimate   every other sentence              FAIL — deletion between sentences
  clause     first clause of every sentence    FAIL — deletion WITHIN sentences
  shuffle    same words, scrambled, half kept  FAIL — order destroyed
  verbatim   unchanged                         FAIL — did not move

`destop` vs `decimate` is the whole test. Both shorten by roughly the same
amount. One keeps every content word of every sentence; the other abandons half
the sentences outright. A retention statistic that cannot separate those two is
not measuring retention.

## A mislabelled arm, and why it is worth recording

The first version of this file used `clause` as the correct HOLD, on the
reasoning that keeping the opening of every sentence covers every unit. Looking
at what it actually produced killed that: on `becoming`, it turns

    "did a term's meaning *change*, or does one name cover two things?"

into

    "did a term's meaning *change*."

which drops the crux of the module. `clause` is deletion within sentences, not
compression, and labelling it HOLD made the benchmark's ground truth wrong — the
exact defect class the metric it grades is accused of. It is kept as a FAIL arm
because deletion-within-sentence is a distinct failure worth detecting.

**You cannot mechanically generate a genuine compression**, because compression
is precisely the semantic operation under test. `destop` is the nearest honest
approximation: dropping function words shortens the text while leaving every
content word of every sentence in place, so nothing is abandoned.

## Reading the result

`accuracy` is over arms whose verdict is known. A metric that says HOLD to
everything scores 1/6; one that says FAIL to everything scores 5/6 — which is
why `separation` is reported too: the margin gap between `clause` and the best
failing arm. Positive separation is the thing that matters. Accuracy alone can
be gamed by a constant.

Run:  python3 benchmarks/direction/run.py [--texts N]
"""
import argparse, json, pathlib, random, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT.parent / "target" / "release" / "physis-core"


def sources(limit):
    """Real prose from this crate: module headers, which are written for humans
    and are not test fixtures."""
    out = []
    for f in sorted((ROOT / "src").rglob("*.rs")):
        lines, buf = f.read_text(errors="ignore").splitlines(), []
        for ln in lines:
            if ln.startswith("//!"):
                buf.append(ln[3:].strip())
            elif buf:
                break
        text = " ".join(x for x in buf if x and not x.startswith(("|", "```", "-", "#")))
        text = re.sub(r"\[`([^`]+)`\]", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 90:
            out.append((f.stem, text))
        if len(out) >= limit:
            break
    return out


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=\.)\s+", t) if len(s.split()) >= 3]


STOP = set(
    "a an the of to in on at by for with from and or but is are was were be been being "
    "that this these those it its as not no so if then than which who whom whose what "
    "do does did done have has had having will would shall should can could may might "
    "must there here when where why how all any each every both few more most other "
    "some such only own same too very just also into over under again further once".split()
)


def arms(text):
    """Rule-generated arms with objective labels. `True` = the verdict should hold."""
    sents = sentences(text)
    words = text.split()
    half = max(1, len(words) // 2)
    # Content-preserving: every content word of every sentence survives.
    destop = " ".join(w for w in words if w.lower().strip(".,;:!?()") not in STOP)
    # Deletion within a sentence — kept as a FAIL arm, see the header.
    clause = " ".join(s.split(",")[0].split(";")[0].strip().rstrip(".") + "." for s in sents)
    shuf = words[:]
    random.Random(11).shuffle(shuf)
    return {
        "destop":   (destop, True),
        "truncate": (" ".join(words[:half]), False),
        "decimate": (" ".join(sents[::2]), False),
        "clause":   (clause, False),
        "shuffle":  (" ".join(shuf[:half]), False),
        "verbatim": (text, False),
    }


def run(before, after):
    with tempfile.TemporaryDirectory() as d:
        b, a = pathlib.Path(d) / "b.txt", pathlib.Path(d) / "a.txt"
        b.write_text(before)
        a.write_text(after)
        p = subprocess.run(
            [str(BIN), "direction", "--before", str(b), "--after", str(a),
             "--want", "shorter", "--json"],
            capture_output=True, text=True, timeout=600,
        )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=int, default=6)
    args = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"missing {BIN} — cargo build --release --features cli,embed-onnx --bin physis-core")

    texts = sources(args.texts)
    print(f"direction benchmark — {len(texts)} real source texts, 5 rule-generated arms each\n")

    correct = total = 0
    margins = {k: [] for k in ("destop", "truncate", "decimate", "clause", "shuffle", "verbatim")}
    for name, text in texts:
        row = []
        for arm, (after, should_hold) in arms(text).items():
            v = run(text, after)
            if v is None:
                row.append(f"{arm}=ERR")
                continue
            margin = v["retention"] - v["null_retention"]
            held = v["moved"] and margin > 0.02
            margins[arm].append(margin)
            total += 1
            ok = held == should_hold
            correct += ok
            row.append(f"{arm}={'H' if held else 'f'}{'' if ok else '!'}")
        print(f"  {name:<18} " + "  ".join(row))

    print(f"\naccuracy {correct}/{total} = {correct / max(total, 1):.2f}")
    print("\nmean margin by arm (the statistic itself, before any threshold)")
    for k, v in margins.items():
        if v:
            print(f"  {k:<10} {sum(v) / len(v):+.4f}")

    good = sum(margins["destop"]) / len(margins["destop"]) if margins["destop"] else 0
    bad = max(
        (sum(v) / len(v) for k, v in margins.items() if k != "destop" and v),
        default=0,
    )
    sep = good - bad
    print(f"\nSEPARATION  destop - best failing arm = {sep:+.4f}")
    if sep > 0.02:
        print("  The statistic separates compression from deletion.")
    else:
        print("  IT DOES NOT SEPARATE. A threshold cannot rescue this: the arms")
        print("  are not ordered correctly, so no cutoff classifies them.")
    sys.exit(0 if sep > 0.02 else 1)


if __name__ == "__main__":
    main()
