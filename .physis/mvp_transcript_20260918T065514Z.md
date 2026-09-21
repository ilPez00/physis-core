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
 └─ benchmarks/results
     └─ mode-inventory.json
         └─ mode-inventory.json  ↔ 0.12  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
     └─ run.json
         └─ benchmarks/results/run.json:41-43  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.05 × 0.60
     └─ worldstate-e59-docs-llmfb.json
         └─ worldstate-e59-docs-llmfb.json  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
     └─ worldstate-e59-log-llmfb.json
         └─ worldstate-e59-log-llmfb.json  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.05 × 0.60
 └─ examples
     └─ experiment47_cross_model_divergence.rs
         └─ experiment47_cross_model_divergence.rs  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.05 × 0.60
     └─ experiment49_self_consistent_dimension.rs
         └─ experiment49_self_consistent_dimension.rs  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
 └─ research
     └─ E62_MACHINE_LOG.md
         └─ E62_MACHINE_LOG.md  ↔ 0.12  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.07 × 0.60
     └─ E63_LEXICAL_SEARCH.md
         └─ E63_LEXICAL_SEARCH.md  ↔ 0.13  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.08 × 0.60
     └─ E64_WORD_TABLE.md
         └─ E64_WORD_TABLE.md  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
     └─ E65_FILTER_CHOICE.md
         └─ E65_FILTER_CHOICE.md  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
 └─ research/impossible_machine/visualizations
     └─ offline_signature.svg
         └─ research/impossible_machine/visualizations/offline_signature.svg:1-1  ↔ 0.11  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.05 × 0.60
 └─ src
     └─ bench.rs
         └─ bench.rs  ↔ 0.12  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.08 × 0.60

trail: ?

── CONTEXT ──
packed 8/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.996  config/semiotic_ontology.json:841-841  (1 tokens)
   0.996  src/tokenizer.rs:161-161  (1 tokens)
   0.996  src/oracle.rs:161-161  (1 tokens)
   0.996  benchmarks/results/worldstate-e59-log-llmfb.json:161-161  (1 tokens)
   0.996  benchmarks/results/c3-interpreter-swap.json:41-41  (1 tokens)
   0.996  benchmarks/results/worldstate-e59-log-llm.json:161-161  (1 tokens)
   0.996  benchmarks/results/worldstate-e59-docs-llm.json:161-161  (1 tokens)
   0.996  benchmarks/results/grid-anchor.json:641-641  (1 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 8
model/provider used minilm (extractive; oracle when configured)
agent calls 0
latency 1144 ms
  [0.13] research/E63_LEXICAL_SEARCH.md  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.08 × 0.60
  [0.12] src/bench.rs  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.08 × 0.60
  [0.12] research/E62_MACHINE_LOG.md  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.07 × 0.60
  [0.12] benchmarks/results/mode-inventory.json  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60
  [0.11] research/E64_WORD_TABLE.md  why: lexical 0.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.06 × 0.60

```


## answer: what does this project do?

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

} [1]
} [2]
} [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1 PHYSIS_ORACLE_MODEL=<model> PHYSIS_ORACLE_KEY=ollama

── SOURCES ──
  [1] config/semiotic_ontology.json:841-841 (0.895)
      }
  [2] src/tokenizer.rs:161-161 (0.895)
      }
  [3] src/oracle.rs:161-161 (0.895)
      }
  [4] benchmarks/results/worldstate-e59-log-llmfb.json:161-161 (0.895)
      }
  [5] benchmarks/results/c3-interpreter-swap.json:41-41 (0.895)
      }
  [6] benchmarks/results/worldstate-e59-log-llm.json:161-161 (0.895)
      }
  [7] benchmarks/results/worldstate-e59-docs-llm.json:161-161 (0.895)
      }
  [8] benchmarks/results/grid-anchor.json:641-641 (0.895)
      }

8 documents · context 8 tokens (conventional retrieval 8) · 52 ms
retrieval backend: minilm

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
         └─ DEMO.md:1-40  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
     └─ NEUROSEMANTIC_LAYER.md
         └─ NEUROSEMANTIC_LAYER.md:41-80  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ NEUROSEMANTIC_LAYER.md:1-40  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ README.md
         └─ README.md:121-160  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ SHIP_PLAN.md
         └─ SHIP_PLAN.md:41-72  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
 └─ src
     └─ act_recall.rs
         └─ src/act_recall.rs:641-680  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ embed.rs
         └─ src/embed.rs:201-240  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ map.rs
         └─ src/map.rs:601-640  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ notebook.rs
         └─ src/notebook.rs:121-160  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ rag.rs
         └─ src/rag.rs:241-280  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ system_mcp.rs
         └─ src/system_mcp.rs:161-200  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ worldstate.rs
         └─ src/worldstate.rs:481-520  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60

trail: ? › retrieval implemented?

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
latency 1162 ms
  [0.23] src/worldstate.rs  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
  [0.23] DEMO.md  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
  [0.23] src/act_recall.rs  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.23] NEUROSEMANTIC_LAYER.md  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.23] NEUROSEMANTIC_LAYER.md  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

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

8 documents · context 21 tokens (conventional retrieval 21) · 161 ms
retrieval backend: minilm

```


## input

```
inspect
```


## inspect src/worldstate.rs:481-520

```
--- src/worldstate.rs:481-520 (Passage) ---
provenance: src/worldstate.rs
parents:
  fs     src
  note   worldstate.rs
  kind   Passage
relations:
  (none)

source:
        let s = score_arm("random", &cands, &adj, &adj, 3);
        assert!(s.top1 < 0.15, "random top1 = {}", s.top1);
    }

    #[test]
    fn bootstrap_interval_brackets_the_mean() {
        let v: Vec<f64> = (0..50).map(|i| f64::from(u8::from(i % 4 == 0))).collect();
        let (lo, hi) = bootstrap_ci(&v, 500, 5);
        let m = mean(&v);
        assert!(lo <= m && m <= hi, "mean {m} outside [{lo}, {hi}]");
    }
}

// ---------------------------------------------------------------------------
// E56 — position in text as the temporal coordinate.
//
// The state sequence is `W_0 … W_n` where `p` is the sentence's ordinal
// position, not a clock. Three legs, each with a null that can fire:
//
//   L1  transition derivation   — do set operations over the state sequence
//                                 recover the gold transition labels, and does
//                                 the recovery collapse when order is shuffled?
//   L2  position in geometry    — do embedding neighbours sit near each other
//                                 in the text, or is |Δpos| at its null?
//   L3  antecedent retrieval    — "what preceded this?" Gold is mechanical: the
//                                 nearest earlier sentence mentioning the same
//                                 entity. Arms are order-blind cosine, cosine
//                                 restricted to earlier positions, position
//                                 alone, and the shuffled-position null.
//
// L1 and L3 are the two questions a file browser cannot answer and a world
// model should.

/// A derived transition between adjacent positions.
pub const TRANSITIONS: [&str; 6] = [
    "emerge",
    "persist",
    "change",
    "recur",
    "disappear",

```


## input

```
dependencies
```


## dependencies src/worldstate.rs

```
dependencies of src/worldstate.rs
  ↔ CHANGELOG.md
  ↔ README.md
  ↔ benchmarks/results/worldstate-e59-docs-lex.json
  ↔ benchmarks/results/worldstate-e59-docs-llm.json
  ↔ benchmarks/results/worldstate-e59-docs-llmfb.json
  ↔ benchmarks/results/worldstate-e59-docs-model.json
  ↔ benchmarks/results/worldstate-e59-docs-table-count.json
  ↔ benchmarks/results/worldstate-e59-docs-table-model.json
  ↔ benchmarks/results/worldstate-e59-docs.json
  ↔ benchmarks/results/worldstate-e59-hand-lex.json
  ↔ benchmarks/results/worldstate-e59-log-lex.json
  ↔ benchmarks/results/worldstate-e59-log-llm.json
  ↔ benchmarks/results/worldstate-e59-log-llmfb.json
  ↔ benchmarks/results/worldstate-e59-log-model.json
  ↔ benchmarks/results/worldstate-e59-log-table-count.json
  ↔ benchmarks/results/worldstate-e59-log-table-model.json
  ↔ benchmarks/results/worldstate-e59-log.json
  ↔ benchmarks/results/worldstate-e59.json
  ↔ examples/coverage_demo.rs
  ↔ examples/experiment10_integration.rs
  ↔ examples/experiment11_generalized_certification.rs
  ↔ examples/experiment12_integrated_synthesis.rs
  ↔ examples/experiment13_local_calibration_and_ensemble_gate.rs
  ↔ examples/experiment14_borderline_item_detector.rs
  ↔ examples/experiment15_real_ontology_prevalence.rs
  ↔ examples/experiment16_ablation_matrix.rs
  ↔ examples/experiment17_jepa_ontology_binding.rs
  ↔ examples/experiment18_multi_embedder_comparison.rs
  ↔ examples/experiment19_cross_embedder_corroboration.rs
  ↔ examples/experiment20_fixed_point_discovery.rs
  ↔ examples/experiment21_fixed_cell_linkage.rs
  ↔ examples/experiment22_wordnet_discovery.rs
  ↔ examples/experiment23_wordnet_disambiguated.rs
  ↔ examples/experiment24_cross_embedder_split_agreement.rs
  ↔ examples/experiment25_capacity_constrained_loss_gate.rs
  ↔ examples/experiment26_persistent_registry.rs
  ↔ examples/experiment28_unsupervised_clustering.rs
  ↔ examples/experiment30_holdout_rediscovery.rs
  ↔ examples/experiment31_outward_loop.rs
  ↔ examples/experiment32_convergence_crossval.rs
  ↔ examples/experiment33_mechanical_disposer.rs
  ↔ examples/experiment34_shared_substructure.rs
  ↔ examples/experiment35_independent_substructure.rs
  ↔ examples/experiment36_episodic_substructure.rs
  ↔ examples/experiment37_instance_level.rs
  ↔ examples/experiment38_soundness_audit.rs
  ↔ examples/experiment44_worklist_ranking.rs
  ↔ examples/experiment47_cross_model_divergence.rs
  ↔ examples/experiment55_structure_before_mapping.rs
  ↔ examples/experiment56_position_first.rs
  ↔ examples/experiment57_supersession_retrieval.rs
  ↔ examples/experiment58_extraction.rs
  ↔ examples/experiment59_noun_phrases.rs
  ↔ examples/experiment65_redundancy.rs
  ↔ examples/experiment66_axis_fitness.rs
  ↔ examples/experiment6_dataset_a_hierarchy.rs
  ↔ examples/experiment7_recursion.rs
  ↔ examples/experiment8_dataset_d_crosscutting.rs
  ↔ examples/experiment9_certification_gate.rs
  ↔ examples/human_rerank_export.rs
  ↔ examples/impossible_machine_experiment.rs
  ↔ examples/ontology_metric_learning_experiment.rs
  ↔ examples/perspective_discovery_experiment.rs
  ↔ examples/perspective_discovery_experiment2.rs
  ↔ examples/perspective_discovery_experiment3.rs
  ↔ examples/perspective_discovery_experiment5_context.rs
  ↔ examples/perspective_invention_experiment.rs
  ↔ research/E55_STRUCTURE_BEFORE_MAPPING.md
  ↔ research/E56_POSITION_FIRST.md
  ↔ research/E57_SUPERSESSION_RETRIEVAL.md
  ↔ research/E58_EXTRACTION.md
  ↔ research/E59_NOUN_PHRASES.md
  ↔ research/E60_UNSEEN_PROSE.md
  ↔ research/E61_INTERPRETER_EXTRACTION.md
  ↔ research/E62_MACHINE_LOG.md
  ↔ research/E63_LEXICAL_SEARCH.md
  ↔ research/E64_WORD_TABLE.md
  ↔ research/E65_FILTER_CHOICE.md
  ↔ research/E66_AXIS_FITNESS.md
  ↔ src/act.rs
  ↔ src/becoming.rs
  ↔ src/bin/world.rs
  ↔ src/chain.rs
  ↔ src/claim_identity.rs
  ↔ src/classify.rs
  ↔ src/coverage.rs
  ↔ src/direction.rs
  ↔ src/discovery.rs
  ↔ src/embed.rs
  ↔ src/grid_fitness.rs
  ↔ src/lib.rs
  ↔ src/rag.rs
  ↔ src/studio_lab.rs
  ↔ src/transform.rs
  ↔ tests/transform_bench.rs

```


## input

```
context
```


## context

```
── CONTEXT PACKET ──

INTENT
where is retrieval implemented?

TARGETS
- .gitignore
- config/category_ontology.json
- config/health_fitness_ontology.json
- config/semiotic_ontology.json
- src/coverage.rs
- src/map.rs
- src/system.rs
- src/tokenizer.rs

DEPENDENCIES
- CHANGELOG.md
- Cargo.lock
- Cargo.toml
- DEMO.md
- GETTING_STARTED.md
- NEUROSEMANTIC_LAYER.md
- PLANNING.md
- README.md
- SHIP_PLAN.md
- benchmarks/results/metrics.json
- benchmarks/results/run.json
- config/abstract_ontology.json

CONSTRAINTS
- public interfaces of TARGETS, as consumed by DEPENDENCIES
- file-type shapes hold (rs/md/json/toml — see invariants)

RECENT DELTA
(no recorded changes this session)

RELEVANT SOURCE
--- config/health_fitness_ontology.json:201-205 (7 tokens, score 0.856) ---
        "treatment"
      ]
    }
  ]
}
--- .gitignore:1-1 (2 tokens, score 0.837) ---
/target
--- config/category_ontology.json:921-923 (3 tokens, score 0.829) ---
    }
  ]
}
--- config/semiotic_ontology.json:841-841 (1 tokens, score 0.831) ---
}
--- src/map.rs:681-682 (3 tokens, score 0.833) ---
    })
}
--- src/coverage.rs:441-442 (2 tokens, score 0.833) ---
    }
}
--- src/system.rs:1121-1122 (2 tokens, score 0.833) ---
    }
}
--- src/tokenizer.rs:161-161 (1 tokens, score 0.831) ---
}

EXCLUDED
100.0% of indexed workspace (21 of 1176280 tokens, 2680 chunks, 2992 objects considered)
workspace source: 1176280 tokens · selected context: 21 tokens · excluded: 100.0%
generator: none (deterministic/manual path) — packet is complete without it

```


## input

```
focus retrieval
```


## intent: retrieval

```
PROJECT
 └─ 
     └─ DEMO.md
         └─ DEMO.md:1-40  ↔ 0.39  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
     └─ NEUROSEMANTIC_LAYER.md
         └─ NEUROSEMANTIC_LAYER.md:41-80  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
         └─ NEUROSEMANTIC_LAYER.md:1-40  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
     └─ PLANNING.md
         └─ PLANNING.md:441-480  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ README.md
         └─ README.md:121-160  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
         └─ README.md:1-40  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
 └─ examples
     └─ experiment57_supersession_retrieval.rs
         └─ experiment57_supersession_retrieval.rs  ↔ 0.39  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
 └─ research
     └─ E56_POSITION_FIRST.md
         └─ research/E56_POSITION_FIRST.md:41-80  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
     └─ E57_SUPERSESSION_RETRIEVAL.md
         └─ E57_SUPERSESSION_RETRIEVAL.md  ↔ 0.39  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
         └─ research/E57_SUPERSESSION_RETRIEVAL.md:1-40  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
     └─ E59_NOUN_PHRASES.md
         └─ research/E59_NOUN_PHRASES.md:41-80  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
 └─ src
     └─ worldstate.rs
         └─ src/worldstate.rs:481-520  ↔ 0.38  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

trail: ? › retrieval implemented? › retrieval

── CONTEXT ──
packed 17/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.875  config/semiotic_ontology.json:841-841  (1 tokens)
   0.858  config/health_fitness_ontology.json:201-205  (7 tokens)
   0.868  .gitignore:1-1  (2 tokens)
   0.873  src/map.rs:681-682  (3 tokens)
   0.875  src/tokenizer.rs:161-161  (1 tokens)
   0.875  src/oracle.rs:161-161  (1 tokens)
   0.875  benchmarks/results/worldstate-e59-log-llmfb.json:161-161  (1 tokens)
   0.875  benchmarks/results/c3-interpreter-swap.json:41-41  (1 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 17
model/provider used minilm (extractive; oracle when configured)
agent calls 0
latency 785 ms
  [0.39] examples/experiment57_supersession_retrieval.rs  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
  [0.39] DEMO.md  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
  [0.39] research/E57_SUPERSESSION_RETRIEVAL.md  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.02 × 0.60
  [0.38] NEUROSEMANTIC_LAYER.md  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60
  [0.38] research/E57_SUPERSESSION_RETRIEVAL.md  why: lexical 1.00 × 0.30 · recency 0.96 × 0.08 · semantic 0.01 × 0.60

```


## focus: retrieval

```
── ATTENTION ──
HOT (3)
  0.39  examples/experiment57_supersession_retrieval.rs
  0.39  DEMO.md
  0.39  research/E57_SUPERSESSION_RETRIEVAL.md
WARM (5)
  0.38  NEUROSEMANTIC_LAYER.md
  0.38  research/E57_SUPERSESSION_RETRIEVAL.md
  0.38  README.md
  0.38  NEUROSEMANTIC_LAYER.md
  0.38  research/E56_POSITION_FIRST.md
COLD (52) — collapsed
  · src/worldstate.rs
  · research/E59_NOUN_PHRASES.md
  · README.md
  · PLANNING.md
  · README.md
  · DEMO.md
  · research/E56_POSITION_FIRST.md
  · README.md
  … and 44 more
pinned objects never demote (0 pins)

```


## input

```
what changed recently?
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
improve retrieval without changing its public API
```


## intent: improve retrieval without changing its public API

```
PROJECT
 └─ 
     └─ CHANGELOG.md
         └─ CHANGELOG.md:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ DEMO.md
         └─ DEMO.md:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ DEMO.md:561-600  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ DEMO.md:81-120  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ README.md
         └─ README.md:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ README.md:81-120  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ SHIP_PLAN.md
         └─ SHIP_PLAN.md:41-72  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
 └─ src
     └─ lib.rs
         └─ src/lib.rs:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ rag.rs
         └─ src/rag.rs:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ src/rag.rs:241-280  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ system.rs
         └─ src/system.rs:121-160  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ src/system.rs:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

trail: ? › retrieval implemented? › retrieval › retrieval its public API

── CONTEXT ──
packed 58/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.854  src/map.rs:681-682  (3 tokens)
   0.826  config/health_fitness_ontology.json:201-205  (7 tokens)
   0.816  .github/FUNDING.yml:1-2  (27 tokens)
   0.849  .gitignore:1-1  (2 tokens)
   0.790  .git:1-1  (13 tokens)
   0.850  config/semiotic_ontology.json:841-841  (1 tokens)
   0.848  config/category_ontology.json:921-923  (3 tokens)
   0.850  src/coverage.rs:441-442  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 58
model/provider used minilm (extractive; oracle when configured)
agent calls 0
latency 816 ms
  [0.30] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/system.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/rag.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] README.md  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/system.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

```


## plan: improve retrieval without changing its public API

```
── PLAN ──
intent: improve retrieval without changing its public API
scoped files: .git, .github/FUNDING.yml, .gitignore, config/category_ontology.json, config/health_fitness_ontology.json, config/semiotic_ontology.json, src/coverage.rs, src/map.rs
agent: existing harness reads the context; no new framework.
(pass --yes to approve)

```


## input

```
improve retrieval without changing its public API --yes
```


## intent: improve retrieval without changing its public API

```
PROJECT
 └─ 
     └─ CHANGELOG.md
         └─ CHANGELOG.md:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ DEMO.md
         └─ DEMO.md:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ DEMO.md:561-600  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ DEMO.md:81-120  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ README.md
         └─ README.md:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ README.md:81-120  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ SHIP_PLAN.md
         └─ SHIP_PLAN.md:41-72  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
 └─ src
     └─ lib.rs
         └─ src/lib.rs:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ rag.rs
         └─ src/rag.rs:41-80  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ src/rag.rs:241-280  ↔ 0.23  why: lexical 0.50 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
     └─ system.rs
         └─ src/system.rs:121-160  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
         └─ src/system.rs:1-40  ↔ 0.30  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

trail: ? › retrieval implemented? › retrieval › retrieval its public API › retrieval its public API

── CONTEXT ──
packed 58/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.854  src/map.rs:681-682  (3 tokens)
   0.826  config/health_fitness_ontology.json:201-205  (7 tokens)
   0.816  .github/FUNDING.yml:1-2  (27 tokens)
   0.849  .gitignore:1-1  (2 tokens)
   0.790  .git:1-1  (13 tokens)
   0.850  config/semiotic_ontology.json:841-841  (1 tokens)
   0.848  config/category_ontology.json:921-923  (3 tokens)
   0.850  src/coverage.rs:441-442  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 58
model/provider used minilm (extractive; oracle when configured)
agent calls 0
latency 830 ms
  [0.30] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/system.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/rag.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] README.md  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60
  [0.30] src/system.rs  why: lexical 0.75 × 0.30 · recency 0.96 × 0.08 · semantic 0.00 × 0.60

```

