# Local divergence — measured 2026-11-23

Recorded because this checkout's `master` and its `origin/master` did not agree,
and the working tree carried noise that looked like work but was not.

## State at measurement

| item | value |
|---|---|
| checkout | `/home/gio/dev/physis-pro/physis-core` (submodule of `physis-pro`) |
| remote | `git@github.com:ilPez00/physis-core.git` |
| branch | `master` |
| local HEAD | `9426b91` (2026-09-19) `refactor(core): retire the product modules, keep the austere core` |
| `origin/master` | `dac78ea` (2026-11-23) `docs(readme): 0.4.0 is now published on crates.io` |
| ahead / behind | **0 / 10** — pure *behind*, fast-forwardable; nothing unpushed |
| superproject pointer | `physis-pro` records `9426b91`, i.e. it matches local, not upstream |

There were **no unpushed commits**. The divergence is the other direction: ten
upstream commits landed after this checkout was last synced, and the superproject
still pins the older one.

## What the ten upstream commits are

```
dac78ea 2026-11-23 docs(readme): 0.4.0 is now published on crates.io
0b9a87f 2026-11-23 docs(readme): state current status — frozen 0.1.26 research repo
10d12f0 2026-09-21 test(machines): pin reader to shipped results artifact
2171046 2026-09-21 feat(machines): typed ImpossibleReport readers for run artifacts
230cf3d 2026-09-21 Move E61-E65 result JSON into the submodule
c737ee9 2026-09-21 E61-E65: singular-series fit series closed
8aa0172 2026-09-20 research(E60): singular-series fit at N=1e6 — PARTIAL (G1+G2 pass, G3 miss 1.47/1.49 vs 1.32)
ed8dc29 2026-09-20 fix(grid): migrate 35 ontology packs to the nine-mode axis
9359b46 2026-09-19 refactor(grid): rework the mode axis to nine owned modes
c3ddb8e 2026-09-19 style: apply cargo fmt across the workspace member
```

Substantive: a grid mode-axis rework to nine owned modes with the 35 ontology
packs migrated onto it, typed `ImpossibleReport` readers for run artifacts, and
the E60–E65 research results moved into this submodule.

## Working-tree noise (not work)

`git status` reported **329 modified files**, which reads as uncommitted work. It
was not:

- **327 were file-mode flips only** — every tracked file on disk was `0755`
  while the index holds `100644`. `git diff --numstat` showed `0 0` for all of
  them; the content was byte-identical. This is a filesystem artefact (this
  volume reports `0755` for everything), not an edit.
- **2 were real**: tracked `__pycache__/*.pyc` files deleted from disk
  (`benchmarks/direction/__pycache__/run.cpython-313.pyc`,
  `research/__pycache__/c3_interpreter_swap.cpython-314.pyc`). Compiled caches do
  not belong in the index.

Resolution in the same change: restore the on-disk modes to the index, untrack the
two `.pyc`, and ignore `__pycache__/` so they cannot be re-added.

## Reading rule for this checkout

A dirty `physis-core` tree on this box does not by itself mean unfinished edits —
check `git diff --numstat` before treating ` M` as content. Only two of the 329
above carried a byte of difference.
