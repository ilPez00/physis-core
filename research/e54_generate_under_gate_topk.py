#!/usr/bin/env python3
TOPK = 3
"""E54 — does the A->B path GENERATE, judged by the compiler?

E53 showed `transform`'s analogy carries real structure for RANKING. Ranking is
not generation: a method can order candidates well and still never produce a
usable artifact. This asks the harder question with a gate nobody can argue
with.

Task. A module M contains `use crate::T::{items};` that it actually depends on.
Delete it and `cargo check` FAILS — verified per case, so every scored item is
known load-bearing. The system is then told the items and must supply the
missing module T. It emits the line and the compiler judges it: a wrong T means
the items do not exist there, and the build stays red.

So the gate is binary and external. No partial credit, no metric of our own
choosing, no way to grade generously.

Arms (same three as E53, so the two experiments are comparable):
  ANALOGY     modules sharing the most other targets with M vote for theirs
  POPULARITY  global in-degree — the majority baseline
  PERMUTED    ANALOGY over a degree-preserving shuffle: the construction-
              matched control that isolates structure from degree

Run from physis-core/:  python3 research/e54_generate_under_gate.py [--limit N]
"""
import argparse, collections, pathlib, random, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
USE_RE = re.compile(r'^use crate::([a-z_][a-z0-9_]*)::(.+);\s*$')


def cargo_check():
    """True when `cargo check --lib` is green."""
    r = subprocess.run(["cargo", "check", "--lib", "--quiet"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def restore(path):
    subprocess.run(["git", "checkout", "--", str(path.relative_to(ROOT))],
                   cwd=ROOT, capture_output=True, timeout=120)


def dependency_graph():
    """(module -> set of crate:: targets it references), from the source."""
    edges = collections.defaultdict(set)
    for f in sorted(SRC.rglob("*.rs")):
        m = f.stem
        if m in ("lib", "main"):
            continue
        for tgt in re.findall(r'crate::([a-z_][a-z0-9_]*)', f.read_text(errors="ignore")):
            if tgt != m:
                edges[m].add(tgt)
    return edges


def analogy(edges, subj, known):
    votes = collections.defaultdict(float)
    for other, targets in edges.items():
        if other == subj:
            continue
        overlap = len(known & targets)
        if not overlap:
            continue
        w = overlap / max(len(known), 1) ** 0.5
        for t in targets - known:
            votes[t] += w
    return [t for t, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))]


def popularity(edges, subj, known):
    votes = collections.Counter()
    for other, targets in edges.items():
        if other != subj:
            votes.update(targets - known)
    return [t for t, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))]


def permute(edges, seed=20260912):
    """Degree-preserving shuffle: both degree sequences survive, only the
    pairing dies. POPULARITY is therefore unchanged by construction."""
    rng = random.Random(seed)
    pairs = [(s, t) for s, ts in edges.items() for t in ts]
    for _ in range(len(pairs) * 8):
        i, j = rng.randrange(len(pairs)), rng.randrange(len(pairs))
        (a, b), (c, d) = pairs[i], pairs[j]
        if a == c or b == d:
            continue
        pairs[i], pairs[j] = (a, d), (c, b)
    out = collections.defaultdict(set)
    for s, t in set(pairs):
        out[s].add(t)
    return out


def cases():
    """Every `use crate::T::{items};` line, with its file and position."""
    out = []
    for f in sorted(SRC.rglob("*.rs")):
        if f.stem in ("lib", "main"):
            continue
        lines = f.read_text(errors="ignore").splitlines(keepends=True)
        for i, line in enumerate(lines):
            m = USE_RE.match(line.rstrip("\n"))
            if m:
                out.append({"file": f, "line": i, "target": m.group(1),
                            "items": m.group(2), "text": line})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--topk", type=int, default=3,
                    help="candidates to try per case before giving up")
    args = ap.parse_args()
    global TOPK
    TOPK = args.topk

    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"src/ is dirty — this experiment rewrites files in place:\n{dirty}")

    print("E54 — generation under a compiler gate\n")
    if not cargo_check():
        sys.exit("baseline `cargo check --lib` is already red; fix before measuring")
    print("baseline green\n")

    edges = dependency_graph()
    pedges = permute(edges)
    all_cases = cases()
    random.Random(7).shuffle(all_cases)

    score = {a: [0, 0] for a in ("ANALOGY", "POPULARITY", "PERMUTED")}  # [pass, n]
    tries = {}
    skipped = 0
    t0 = time.time()

    for c in all_cases:
        if score["ANALOGY"][1] >= args.limit:
            break
        f, orig = c["file"], c["file"].read_text()
        lines = orig.splitlines(keepends=True)

        # 1. delete the import; it must actually break the build.
        del lines[c["line"]]
        f.write_text("".join(lines))
        if cargo_check():
            restore(f)
            skipped += 1
            continue  # not load-bearing: nothing to generate

        subj, truth = f.stem, c["target"]
        known = edges[subj] - {truth}

        preds = {
            "ANALOGY": analogy(edges, subj, known),
            "POPULARITY": popularity(edges, subj, known),
            "PERMUTED": analogy(pedges, subj, pedges.get(subj, set()) - {truth}),
        }
        row = []
        for arm, ranked in preds.items():
            # An agent does not get one shot: it emits, compiles, and retries.
            # Scoring only the top candidate measures a workflow nobody uses.
            # `attempts` is the honest cost — 1 means first try.
            hit, attempts = False, 0
            for guess in (ranked[:TOPK] or ["__none__"]):
                attempts += 1
                emitted = lines[:]
                emitted.insert(c["line"], f"use crate::{guess}::{c['items']};\n")
                f.write_text("".join(emitted))
                if cargo_check():
                    hit = True
                    break
            score[arm][0] += hit
            score[arm][1] += 1
            tries.setdefault(arm, []).append(attempts if hit else None)
            row.append(f"{arm[:4]}={'PASS' if hit else 'fail'}@{attempts}")
        restore(f)
        print(f"  {subj}.rs -> crate::{truth}   " + "  ".join(row))

    n = score["ANALOGY"][1]
    print(f"\n{n} load-bearing cases, {skipped} skipped as not load-bearing, "
          f"{time.time() - t0:.0f}s\n")
    print(f"GATE PASS RATE within {TOPK} attempts (the compiler decides)")
    for arm, (p, tot) in score.items():
        got = [t for t in tries.get(arm, []) if t]
        mean = sum(got) / len(got) if got else float("nan")
        print(f"  {arm:<11} {p}/{tot} = {p / max(tot, 1):.3f}   mean attempts when it passed: {mean:.2f}")

    a = score["ANALOGY"][0] / max(n, 1)
    pm = score["PERMUTED"][0] / max(n, 1)
    pp = score["POPULARITY"][0] / max(n, 1)
    print(f"\n  ANALOGY - PERMUTED   {a - pm:+.3f}")
    print(f"  ANALOGY - POPULARITY {a - pp:+.3f}")
    print()
    if a - pm > 0.05 and a - pp > 0.05:
        print("  SUPPORTED: the path generates artifacts a compiler accepts,")
        print("  above both its construction-matched null and the majority baseline.")
    elif a - pm <= 0.05:
        print("  NOT SUPPORTED: the permuted null matches it. Ranking structure")
        print("  (E53) did not carry through to generation. This is a result.")
    else:
        print("  PARTIAL: beats the null but not the majority baseline.")

    left = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    print(f"\nsrc/ after run: {'CLEAN' if not left else 'DIRTY — ' + left}")


if __name__ == "__main__":
    main()
