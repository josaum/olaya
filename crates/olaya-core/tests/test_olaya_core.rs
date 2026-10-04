use arrow::array::{Array, StringArray};
use olaya_core::{
    enforce_disjointness_inplace, enforce_subsumption_monotonicity_inplace, read_ontology_ipc,
    write_ontology_ipc, OlayaEngine, OntologyBatchBuilder,
};
use tempfile::NamedTempFile;

fn create_sample_arrow_ontology() -> arrow::record_batch::RecordBatch {
    let mut builder = OntologyBatchBuilder::new();

    // 0: Root
    builder.append_concept(
        0,
        "INCIDENT:ROOT",
        "Operational Incident",
        "Root incident class",
        &[],
        &[],
        &["incident"],
    );
    // 1: Data (parent: 0)
    builder.append_concept(
        1,
        "INCIDENT:DATA",
        "Data & Storage Incident",
        "Storage and database failures",
        &[0],
        &[2],
        &["database", "db", "storage"],
    );
    // 2: Billing (parent: 0, disjoint: 1)
    builder.append_concept(
        2,
        "INCIDENT:BILLING",
        "Billing Incident",
        "Payment and subscription errors",
        &[0],
        &[1],
        &["billing", "payment", "invoice"],
    );
    // 3: Deadlock (parent: 1)
    builder.append_concept(
        3,
        "INCIDENT:DB_DEADLOCK",
        "Database Deadlock",
        "Lock wait timeouts on SQL queries",
        &[1],
        &[],
        &["deadlock", "postgres lock"],
    );

    builder
        .finish()
        .expect("Failed to build ontology RecordBatch")
}

#[test]
fn test_arrow_ontology_ipc_roundtrip() {
    let batch = create_sample_arrow_ontology();
    assert_eq!(batch.num_rows(), 4);

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write_ontology_ipc(&batch, temp_file.path()).expect("Failed to write IPC");

    let loaded = read_ontology_ipc(temp_file.path()).expect("Failed to read IPC");
    assert_eq!(loaded.num_rows(), 4);

    let curies = loaded
        .column_by_name("curie")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(curies.value(3), "INCIDENT:DB_DEADLOCK");
}

#[test]
fn test_inplace_axioms() {
    // Child has 0.85, parent has 0.30 -> Parent must be lifted to >= 0.85
    let mut probs = vec![0.20, 0.30, 0.10, 0.85]; // [root, data, billing, deadlock]
    let parent_edges = vec![(3, 1), (1, 0)]; // deadlock -> data, data -> root

    enforce_subsumption_monotonicity_inplace(&mut probs, &parent_edges);
    assert!(probs[1] >= probs[3]);
    assert!(probs[0] >= probs[1]);

    // Disjointness test: slot 1 (data) and slot 2 (billing) are disjoint
    let disjoint_edges = vec![(1, 2)];
    enforce_disjointness_inplace(&mut probs, &parent_edges, &disjoint_edges, 0.5);
    assert_eq!(probs[2], 0.0); // Billing must be suppressed to 0.0
}

#[test]
fn test_engine_process_batch_to_arrow() {
    let ontology = create_sample_arrow_ontology();
    let engine = OlayaEngine::new(ontology).expect("Failed to build OlayaEngine");

    let states = vec![
        "Fatal deadlock detected on PostgreSQL customer table while locking rows",
        "Payment gateway webhook failed with HTTP 500 when billing subscription",
    ];

    let output_batch = engine
        .process_batch_to_arrow(&states, "triage_q1")
        .expect("Failed to process batch to Arrow");

    assert_eq!(output_batch.num_rows(), 2);

    let selected_curies = output_batch
        .column_by_name("selected_curie")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let confidences = output_batch
        .column_by_name("confidence")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();

    println!("\nRow 0 Selected Curie: {}", selected_curies.value(0));
    println!("Row 0 Confidence: {:.4}", confidences.value(0));
    println!("Row 1 Selected Curie: {}", selected_curies.value(1));
    println!("Row 1 Confidence: {:.4}", confidences.value(1));

    assert!(selected_curies.value(0).contains("DATA") || selected_curies.value(0).contains("DB"));
    assert!(selected_curies.value(1).contains("BILLING"));
}

#[test]
fn test_owl_fast_core_direct_arrow_ingestion() {
    let ttl = br#"
        @prefix ex: <http://example.org/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

        ex:Incident a owl:Class ;
            rdfs:label "Incident" .

        ex:NetworkIncident a owl:Class ;
            rdfs:label "Network Outage" ;
            rdfs:subClassOf ex:Incident ;
            skos:altLabel "Packet Loss" .

        ex:SecurityBreach a owl:Class ;
            rdfs:label "Security Breach" ;
            rdfs:subClassOf ex:Incident ;
            owl:disjointWith ex:NetworkIncident .
    "#;

    let batch = OntologyBatchBuilder::from_owl_bytes(ttl, owl_fast_core::RdfFormat::Turtle)
        .expect("Failed to parse TTL into Arrow batch");

    assert_eq!(batch.num_rows(), 3);

    let engine = OlayaEngine::new(batch).expect("Failed to construct OlayaEngine from parsed TTL");
    let res = engine
        .system_one(
            "High latency and severe packet loss on core backbone router",
            4,
        )
        .unwrap();

    println!("\nTTL Ingested Query Result:");
    println!("Selected Curie: {}", res.selected_curie);
    println!("Selected Label: {}", res.selected_label);
    println!("Confidence: {:.4}", res.confidence);
    println!("Axioms Verified: {}", res.axioms_verified);

    assert_eq!(res.selected_curie, "NetworkIncident");
    assert!(res.axioms_verified);
}

#[test]
fn test_tie_break_prefers_deepest_regardless_of_row_order() {
    // Child rows come *before* their parents, so row order cannot be relied on.
    let mut builder = OntologyBatchBuilder::new();
    builder.append_concept(
        10,
        "INCIDENT:DB_DEADLOCK",
        "Database Deadlock",
        "Lock waits",
        &[11],
        &[],
        &["deadlock"],
    );
    builder.append_concept(
        11,
        "INCIDENT:DATA",
        "Data Incident",
        "Storage failures",
        &[12],
        &[],
        &[],
    );
    builder.append_concept(
        12,
        "INCIDENT:ROOT",
        "Operational Incident",
        "Root",
        &[],
        &[],
        &[],
    );
    let engine = OlayaEngine::new(builder.finish().unwrap()).unwrap();

    let res = engine.system_one("deadlock", 4).unwrap();
    assert_eq!(res.selected_curie, "INCIDENT:DB_DEADLOCK");
}

#[test]
fn test_subgraph_includes_full_ancestor_chain() {
    let mut builder = OntologyBatchBuilder::new();
    builder.append_concept(0, "R", "Root", "", &[], &[], &[]);
    builder.append_concept(1, "A", "Level One", "", &[0], &[], &[]);
    builder.append_concept(2, "B", "Level Two", "", &[1], &[], &[]);
    builder.append_concept(3, "C", "Kafka Consumer Lag", "", &[2], &[], &[]);
    let batch = builder.finish().unwrap();
    let router = olaya_core::ArrowCandidateRouter::build(&batch).unwrap();

    let sub = router.extract_subgraph("kafka lag spike", 1);
    assert_eq!(sub.row_indices, vec![0, 1, 2, 3]);
    assert_eq!(sub.parent_edges.len(), 3);
}

#[test]
fn test_acronyms_route_to_matching_concepts() {
    let mut builder = OntologyBatchBuilder::new();
    builder.append_concept(0, "R", "Incident", "", &[], &[], &[]);
    builder.append_concept(1, "HTTP_5XX", "HTTP Server Error", "", &[0], &[], &[]);
    builder.append_concept(2, "PG", "PostgreSQL Failure", "", &[0], &[], &[]);
    let engine = OlayaEngine::new(builder.finish().unwrap()).unwrap();

    assert_eq!(
        engine
            .system_one("upstream returned http 503", 4)
            .unwrap()
            .selected_curie,
        "HTTP_5XX"
    );
    assert_eq!(
        engine
            .system_one("POSTGRESQL replica down", 4)
            .unwrap()
            .selected_curie,
        "PG"
    );
}

#[test]
fn test_engine_rejects_invalid_schema() {
    use arrow::array::{StringArray as S, UInt32Array as U};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    // `curie` column typed as UInt32 instead of Utf8.
    let schema = Arc::new(Schema::new(vec![
        Field::new("concept_id", DataType::UInt32, false),
        Field::new("curie", DataType::UInt32, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(U::from(vec![0])),
            Arc::new(U::from(vec![0])),
            Arc::new(S::from(vec!["x"])),
        ],
    )
    .unwrap();
    assert!(OlayaEngine::new(batch).is_err());
}

#[test]
fn test_ubiquitous_tokens_do_not_drown_rare_matches() {
    let n = olaya_core::MAX_QUERY_POSTINGS + 100;
    let mut builder = OntologyBatchBuilder::new();
    builder.append_concept(0, "R", "Root", "", &[], &[], &[]);
    for i in 1..n {
        builder.append_concept(
            i as u32,
            &format!("G{i}"),
            &format!("Generated Type {i}"),
            "",
            &[0],
            &[],
            &[],
        );
    }
    builder.append_concept(n as u32, "DL", "Database Deadlock", "", &[0], &[], &[]);
    let engine = OlayaEngine::new(builder.finish().unwrap()).unwrap();

    let res = engine.system_one("generated type deadlock", 8).unwrap();
    assert_eq!(res.selected_curie, "DL");
}

#[test]
fn test_axiom_kernel_matches_shared_golden_vectors() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../spec/axioms_golden.json");
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tol = golden["tolerance"].as_f64().unwrap() as f32;
    let edges = |v: &serde_json::Value| -> Vec<(usize, usize)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                )
            })
            .collect()
    };

    for case in golden["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut probs: Vec<f32> = case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let parent_edges = edges(&case["parent_edges"]);
        let disjoint_edges = edges(&case["disjoint_edges"]);
        let conf =
            olaya_core::calibrate_and_project_inplace(&mut probs, &parent_edges, &disjoint_edges);

        for (got, want) in probs.iter().zip(case["expected_probs"].as_array().unwrap()) {
            let want = want.as_f64().unwrap() as f32;
            assert!(
                (got - want).abs() <= tol,
                "{name}: probs {probs:?} vs expected {want}"
            );
        }
        let want_conf = case["expected_confidence"].as_f64().unwrap() as f32;
        assert!(
            (conf - want_conf).abs() <= tol,
            "{name}: confidence {conf} vs {want_conf}"
        );
        let verified = olaya_core::verify_axioms(
            &probs,
            &parent_edges,
            &disjoint_edges,
            olaya_core::DISJOINT_THRESHOLD,
        );
        assert_eq!(
            verified,
            case["expected_axioms_verified"].as_bool().unwrap(),
            "{name}: axioms_verified"
        );
    }
}

/// A distilled scorer trained on a different ontology (missing CURIEs) must be rejected when
/// attached instead of silently scoring unknown concepts at index 0; a complete one attaches and
/// scores.
#[test]
fn test_distilled_scorer_rejects_unknown_curies() {
    use olaya_core::DistilledNeuralScorer;
    use std::sync::Arc;

    let scorer_json = |curies: &[&str]| {
        let (buckets, dim, hidden) = (16usize, 4usize, 6usize);
        serde_json::json!({
            "format": "leio-cone-v1",
            "num_buckets": buckets,
            "dim": dim,
            "hidden_dim": hidden,
            "alpha_parent": 0.2,
            "beta_disjoint": 0.3,
            "concept_curies": curies,
            "token_embeddings": vec![vec![0.1f32, 0.2, 0.3, 0.4]; buckets],
            "mlp_w1": vec![vec![0.05f32; dim]; hidden],
            "mlp_b1": vec![0.01f32; hidden],
            "mlp_w2": vec![vec![0.05f32; hidden]; dim],
            "mlp_b2": vec![0.01f32; dim],
            "concept_embeddings": (0..curies.len())
                .map(|i| vec![0.1 * (i as f32 + 1.0), 0.2, 0.3, 0.4])
                .collect::<Vec<_>>(),
        })
        .to_string()
    };

    let partial = Arc::new(
        DistilledNeuralScorer::from_json(&scorer_json(&["INCIDENT:ROOT", "INCIDENT:DATA"]))
            .unwrap(),
    );
    let err = OlayaEngine::new(create_sample_arrow_ontology())
        .unwrap()
        .with_distilled_neural_scorer(partial)
        .err()
        .expect("missing CURIEs must be an error");
    let msg = err.to_string();
    assert!(msg.contains("2 of 4"), "{msg}");
    assert!(msg.contains("INCIDENT:BILLING"), "{msg}");

    let complete = Arc::new(
        DistilledNeuralScorer::from_json(&scorer_json(&[
            "INCIDENT:ROOT",
            "INCIDENT:DATA",
            "INCIDENT:BILLING",
            "INCIDENT:DB_DEADLOCK",
        ]))
        .unwrap(),
    );
    let engine = OlayaEngine::new(create_sample_arrow_ontology())
        .unwrap()
        .with_distilled_neural_scorer(complete)
        .unwrap();
    let res = engine
        .system_one("deadlock on the billing database", 4)
        .unwrap();
    assert!(res.axioms_verified);
    assert!(res.confidence.is_finite());
}

/// The engine must select the same CURIEs as the Node, WASM and Python runtimes on the shared
/// fixture (`spec/decision_parity.json`), so the runtimes stay tied together beyond the kernel.
#[test]
fn test_decisions_match_shared_parity_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/decision_parity.json"
    );
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let raw = fixture["ontology"]["concepts"].as_array().unwrap();
    let id_of: std::collections::HashMap<&str, u32> = raw
        .iter()
        .enumerate()
        .map(|(i, c)| (c["curie"].as_str().unwrap(), i as u32))
        .collect();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| id_of[x.as_str().unwrap()])
            .collect()
    };

    let mut builder = OntologyBatchBuilder::new();
    for (i, c) in raw.iter().enumerate() {
        let synonyms: Vec<&str> = c["synonyms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        builder.append_concept(
            i as u32,
            c["curie"].as_str().unwrap(),
            c["label"].as_str().unwrap(),
            c["description"].as_str().unwrap_or(""),
            &ids(&c["parents"]),
            &ids(&c["disjoint_with"]),
            &synonyms,
        );
    }
    let engine = OlayaEngine::new(builder.finish().unwrap()).unwrap();
    let top_k = fixture["max_candidates"].as_u64().unwrap() as usize;

    for case in fixture["queries"].as_array().unwrap() {
        let state = case["state"].as_str().unwrap();
        let want = case["expected_curie"].as_str().unwrap();
        let res = engine.system_one(state, top_k).unwrap();
        assert_eq!(
            res.selected_curie, want,
            "{state:?}: {:?}",
            res.probabilities
        );
        assert!(res.axioms_verified, "{state:?}");
        assert!(
            res.confidence > 0.0,
            "{state:?}: confidence {} (lifted chains must not zero it)",
            res.confidence
        );
        // Disjointness: the listed concepts must be in the subgraph and zeroed.
        for excluded in case["expected_excluded"].as_array().into_iter().flatten() {
            let curie = excluded.as_str().unwrap();
            let prob = res
                .probabilities
                .iter()
                .find(|(c, _)| c == curie)
                .unwrap_or_else(|| panic!("{state:?}: {curie} not in subgraph"))
                .1;
            assert_eq!(prob, 0.0, "{state:?}: {curie} should be excluded");
        }
    }
}
