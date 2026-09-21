# MVP transcript — /home/gio/dev/physis-pro/physis-core

## startup

```
PHYSIS — physis-core

files          339
objects        2992
relations      0
sample map     repeats 9 · differences 8 · contradictions 1
index          ready

```


## input

```
where is retrieval implemented?
```


## intent: where is retrieval implemented?

```
PROJECT
 └─ 
     └─ DEMO.md
         └─ DEMO.md:1-40  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.01 × 0.60
     └─ NEUROSEMANTIC_LAYER.md
         └─ NEUROSEMANTIC_LAYER.md:41-80  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
         └─ NEUROSEMANTIC_LAYER.md:1-40  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ README.md
         └─ README.md:121-160  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ SHIP_PLAN.md
         └─ SHIP_PLAN.md:41-72  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
 └─ src
     └─ act_recall.rs
         └─ src/act_recall.rs:641-680  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ embed.rs
         └─ src/embed.rs:201-240  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ map.rs
         └─ src/map.rs:601-640  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ notebook.rs
         └─ src/notebook.rs:121-160  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ rag.rs
         └─ src/rag.rs:241-280  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ system_mcp.rs
         └─ src/system_mcp.rs:161-200  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
     └─ worldstate.rs
         └─ src/worldstate.rs:481-520  ↔ 0.24  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.01 × 0.60

trail: retrieval implemented?

── CONTEXT ──
packed 21/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.856  config/health_fitness_ontology.json:201-205  (7 tokens)
   0.837  .gitignore:1-1  (2 tokens)
   0.829  config/category_ontology.json:921-923  (3 tokens)
   0.831  config/semiotic_ontology.json:841-841  (1 tokens)
   0.833  src/map.rs:681-682  (3 tokens)
   0.833  src/coverage.rs:441-442  (2 tokens)
   0.833  src/system.rs:1121-1122  (2 tokens)
   0.831  src/tokenizer.rs:161-161  (1 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 21
model/provider used minilm (extractive; oracle when configured)
agent calls 0
latency 51473 ms
  [0.24] src/worldstate.rs  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.01 × 0.60
  [0.23] DEMO.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.01 × 0.60
  [0.23] src/act_recall.rs  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
  [0.23] NEUROSEMANTIC_LAYER.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60
  [0.23] NEUROSEMANTIC_LAYER.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.00 × 0.60

```


## answer: where is retrieval implemented?

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

"treatment"       ] [1]
/target [2]
} } [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1 PHYSIS_ORACLE_MODEL=<model> PHYSIS_ORACLE_KEY=ollama

── SOURCES ──
  [1] config/health_fitness_ontology.json:201-205 (0.831)
      "treatment"
  [2] .gitignore:1-1 (0.817)
      /target
  [3] src/coverage.rs:441-442 (0.809)
      }
  [4] src/system.rs:1121-1122 (0.809)
      }
  [5] config/semiotic_ontology.json:841-841 (0.808)
      }
  [6] src/tokenizer.rs:161-161 (0.808)
      }
  [7] src/map.rs:681-682 (0.807)
      })
  [8] config/category_ontology.json:921-923 (0.802)
      }

8 documents · context 21 tokens (conventional retrieval 21) · 165 ms
retrieval backend: minilm

```


## input

```
differences
```


## input

```
history
```


## recent

```
d65858b feat(system): delegation planning, an MCP surface for it, and the reference-spectrum example
8448393 fix(doc): a shell snippet in a doc comment was compiled as Rust
6a450bd fix(examples): a feature-gated main stopped `cargo test` for the whole crate
75acec6 feat(system): predict backs off to a borrowed prior, gated by coverage
60d3b31 feat: declare prediction-from-the-record as a capability
1210b1a feat(system): a run reads the prior it is about to defy
2c3a2dd feat(system): predict — read the workspace's own record before acting
9188b6f fix(cli): `physis … | head` printed a panic after the output it was asked for
145609e feat(system): --include-dir, because the default exclusions hide published artifacts
194e2d1 feat: declare the capabilities, so the claim is checkable instead of narrative
65e3ccc fix(system): history printed every note twice
56b5d6f feat(system): a workspace interface people and agents share, and a packer that pays for itself
88d61ed feat(worldstate): navigational layer over the machine's own log
0203b46 feat(c3): grade the interpreter swap instead of hashing it
9798c43 feat(modes): weight the unowned supersenses by what the corpus actually does

```


## input

```
quit
```

