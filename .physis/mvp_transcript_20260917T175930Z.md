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
what does this project do?
```


## intent: what does this project do?

```
PROJECT
 └─ examples
     └─ experiment47_cross_model_divergence.rs
         └─ examples/experiment47_cross_model_divergence.rs:841-880  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
         └─ examples/experiment47_cross_model_divergence.rs:241-280  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
         └─ examples/experiment47_cross_model_divergence.rs:961-1000  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ impossible_machine_experiment.rs
         └─ examples/impossible_machine_experiment.rs:281-320  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ perspective_invention_experiment.rs
         └─ examples/perspective_invention_experiment.rs:521-560  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ src
     └─ propose.rs
         └─ src/propose.rs:161-200  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ src/bin
     └─ world.rs
         └─ src/bin/world.rs:121-160  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60

trail: ?

── CONTEXT ──
packed 2000/2000 tokens of 1204962 candidate tokens from 6 chunks (2761 candidates over 348 files, 2755 dropped)
   0.795  examples/experiment47_cross_model_divergence.rs:841-880  (460 tokens)
   0.795  examples/impossible_machine_experiment.rs:281-320  (386 tokens)
   0.792  examples/experiment47_cross_model_divergence.rs:241-280  (437 tokens)
   0.793  examples/experiment34_shared_substructure.rs:241-280  (471 tokens)
   0.781  src/transform.rs:441-480  (244 tokens)
   0.746  src/coverage.rs:441-442  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 6
candidate/raw context tokens 1204962
final context tokens 2000
model/provider used random-projection (extractive; oracle when configured)
agent calls 0
latency 546 ms
  [0.55] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/impossible_machine_experiment.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/perspective_invention_experiment.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60

```


## answer: what does this project do?

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

Box::new(move |r: &Row| coh(r)),             ), [1]
let v = x.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;         sd[j] = v.sqrt().max(1e-9); [2]
let n = self.n_max as usize;         let mut worst = 0.0f64; [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1 PHYSIS_ORACLE_MODEL=<model> PHYSIS_ORACLE_KEY=ollama

── SOURCES ──
  [1] examples/experiment47_cross_model_divergence.rs:841-880 (0.657)
      Box::new(move |r: &Row| coh(r)),
  [2] examples/experiment47_cross_model_divergence.rs:241-280 (0.653)
      let v = x.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;
  [3] examples/impossible_machine_experiment.rs:281-320 (0.638)
      let n = self.n_max as usize;
  [4] examples/experiment34_shared_substructure.rs:241-280 (0.633)
      hi_i.len(),
  [5] src/transform.rs:441-480 (0.599)
      } else {
  [6] src/coverage.rs:441-442 (0.540)
      }

6 documents · context 2000 tokens (conventional retrieval 2000) · 17 ms
retrieval backend: random-projection

```


## input

```
quit
```

