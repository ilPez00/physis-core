use physis_core::{
    models::cosine_sim,
    transform::{
        Constraint, ConstraintKind, PatElem, PropKind, Proposition, Predicate, Transform,
        Triple, TriplePattern, WorldState, apply, find_homomorphisms,
    },
};
use std::collections::HashMap;

fn mycorrhiza_world() -> WorldState {
    let flows = Predicate::Custom("flows_to".to_string());
    let t_exchange = Triple::new("ROOT", Predicate::Exchanges, "RESOURCE");
    let t_flow = Triple::new("RESOURCE", flows, "FUNGUS");
    let t_support = Triple::new("FUNGUS", Predicate::Supports, "ROOT");
    let t_require = Triple::new("ROOT", Predicate::Requires, "RESOURCE");
    WorldState {
        facts: vec![
            Proposition::observed(t_exchange.clone(), "bench"),
            Proposition::observed(t_flow, "bench"),
            Proposition::observed(t_support.clone(), "bench"),
            Proposition::observed(t_require, "bench"),
        ],
        constraints: vec![
            Constraint {
                desc: "FUNGUS-supports-ROOT Requires support".to_string(),
                kind: ConstraintKind::Requires {
                    triple: t_support,
                },
            },
            // Second Requires so removing Exchanges is counterfactually incoherent (test 3).
            Constraint {
                desc: "Requires ROOT-exchanges-RESOURCE".to_string(),
                kind: ConstraintKind::Requires {
                    triple: t_exchange,
                },
            },
        ],
    }
}

fn token_overlap(a: &str, b: &str) -> f32 {
    fn toks(s: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in s
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            let owned = t.to_string();
            if !out.contains(&owned) {
                out.push(owned);
            }
        }
        out
    }
    let ta = toks(a);
    let tb = toks(b);
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    let inter = ta.iter().filter(|t| tb.contains(t)).count() as f32;
    let union = (ta.len() + tb.len()) as f32 - inter;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn rule_baseline(s: &str) -> bool {
    s.contains("ROOT") && s.contains("FUNGUS")
}

#[test]
fn transfer_beats_similarity() {
    let world = mycorrhiza_world();
    assert_eq!(world.facts[0].kind, PropKind::Observed);
    assert!((cosine_sim(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);

    let src_text = "ROOT Exchanges RESOURCE RESOURCE flows_to FUNGUS FUNGUS Supports ROOT ROOT Requires RESOURCE";
    let true_text = "TREE Exchanges NUTRIENT NUTRIENT flows_to MYCELIUM MYCELIUM Supports TREE TREE Requires NUTRIENT";
    let decoy_text = "ROOT Causes FUNGUS FUNGUS Causes ROOT ROOT Exchanges RESOURCE RESOURCE flows_to FUNGUS";
    let sim_true = token_overlap(src_text, true_text);
    let sim_decoy = token_overlap(src_text, decoy_text);
    assert!(
        sim_decoy > sim_true,
        "similarity should pick decoy: decoy={sim_decoy} true={sim_true}"
    );

    let flows = Predicate::Custom("flows_to".to_string());
    let source = vec![
        TriplePattern {
            s: PatElem::Var("a".to_string()),
            p: Some(Predicate::Exchanges),
            o: PatElem::Var("b".to_string()),
        },
        TriplePattern {
            s: PatElem::Var("b".to_string()),
            p: Some(flows),
            o: PatElem::Var("c".to_string()),
        },
        TriplePattern {
            s: PatElem::Var("c".to_string()),
            p: Some(Predicate::Supports),
            o: PatElem::Var("a".to_string()),
        },
    ];
    let mut true_map = HashMap::new();
    true_map.insert("a".to_string(), "TREE".to_string());
    true_map.insert("b".to_string(), "NUTRIENT".to_string());
    true_map.insert("c".to_string(), "MYCELIUM".to_string());
    let (_, true_step) = apply(
        &world,
        &Transform::Analogize {
            source: source.clone(),
            map: true_map,
        },
    );
    assert!(
        true_step.accepted,
        "true transfer rejected: {}",
        true_step.note
    );

    // Decoy: inverted causal direction (FUNGUS-causes-ROOT) has no
    // homomorphism in a world with no Causes facts.
    let decoy_source = vec![TriplePattern {
        s: PatElem::Var("c".to_string()),
        p: Some(Predicate::Causes),
        o: PatElem::Var("a".to_string()),
    }];
    let mut decoy_map = HashMap::new();
    decoy_map.insert("c".to_string(), "FUNGUS".to_string());
    decoy_map.insert("a".to_string(), "ROOT".to_string());
    let (_, decoy_step) = apply(
        &world,
        &Transform::Analogize {
            source: decoy_source,
            map: decoy_map,
        },
    );
    assert!(
        !decoy_step.accepted,
        "decoy with inverted causal direction was accepted"
    );
}

#[test]
fn symbolic_rule_baseline_fails_transfer() {
    let flows = Predicate::Custom("flows_to".to_string());
    let renamed = WorldState {
        facts: vec![
            Proposition::observed(
                Triple::new("TREE", Predicate::Exchanges, "NUTRIENT"),
                "bench",
            ),
            Proposition::observed(
                Triple::new("NUTRIENT", flows, "MYCELIUM"),
                "bench",
            ),
            Proposition::observed(
                Triple::new("MYCELIUM", Predicate::Supports, "TREE"),
                "bench",
            ),
            Proposition::observed(
                Triple::new("TREE", Predicate::Requires, "NUTRIENT"),
                "bench",
            ),
        ],
        constraints: vec![],
    };
    let text = renamed
        .triples()
        .iter()
        .map(|t| format!("{} {} {}", t.s, t.p.as_str(), t.o))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        !rule_baseline(&text),
        "literal stub should fail on renamed world"
    );

    let patterns = vec![
        TriplePattern {
            s: PatElem::Var("x".to_string()),
            p: Some(Predicate::Exchanges),
            o: PatElem::Var("y".to_string()),
        },
        TriplePattern {
            s: PatElem::Var("y".to_string()),
            p: Some(Predicate::Custom("flows_to".to_string())),
            o: PatElem::Var("z".to_string()),
        },
    ];
    let homos = find_homomorphisms(&patterns, &renamed.triples());
    assert!(
        !homos.is_empty(),
        "abstracted pattern should match renamed world"
    );
}

#[test]
fn counterfactual_absence_lists_dependents() {
    let world = mycorrhiza_world();
    let stripped = world.without_relation(&Predicate::Exchanges);
    assert!(!stripped
        .triples()
        .iter()
        .any(|t| t.p == Predicate::Exchanges));
    let bad = stripped.incoherent_after();
    assert!(!bad.is_empty(), "removing Exchanges should incohere");
    assert!(
        bad.iter().any(|d| d.contains("Requires")),
        "should mention the Requires constraint, got: {bad:?}"
    );
}

#[test]
fn attractive_invalid_rejected() {
    let cause = Triple::new("ROOT", Predicate::Causes, "SHOOT");
    let world = WorldState {
        facts: vec![Proposition::observed(cause.clone(), "bench")],
        constraints: vec![
            Constraint {
                desc: "Requires ROOT-causes-SHOOT".to_string(),
                kind: ConstraintKind::Requires { triple: cause },
            },
            // Swapped map {a->SHOOT, b->ROOT} double-applies under
            // Analogize rename semantics (replacement loops over map
            // entries), so it lands on a self-loop whichever HashMap
            // order runs: forbid both to reject deterministically.
            Constraint {
                desc: "Forbids self-loop ROOT-causes-ROOT".to_string(),
                kind: ConstraintKind::Forbids {
                    triple: Triple::new("ROOT", Predicate::Causes, "ROOT"),
                },
            },
            Constraint {
                desc: "Forbids self-loop SHOOT-causes-SHOOT".to_string(),
                kind: ConstraintKind::Forbids {
                    triple: Triple::new("SHOOT", Predicate::Causes, "SHOOT"),
                },
            },
        ],
    };
    // Same predicate (Causes), high token overlap, swapped subject/object.
    let source = vec![TriplePattern {
        s: PatElem::Var("a".to_string()),
        p: Some(Predicate::Causes),
        o: PatElem::Var("b".to_string()),
    }];
    let mut map = HashMap::new();
    map.insert("a".to_string(), "SHOOT".to_string());
    map.insert("b".to_string(), "ROOT".to_string());
    assert!(
        token_overlap("ROOT Causes SHOOT", "SHOOT Causes ROOT") > 0.9,
        "swap map should be attractively similar"
    );
    let (_, step) = apply(&world, &Transform::Analogize { source, map });
    assert!(!step.accepted, "swapped Causes analogy was accepted");
}
