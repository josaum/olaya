use tempfile::NamedTempFile;
use arrow::array::{Array, StringArray};
use olaya_core::{
    OntologyBatchBuilder, OlayaEngine,
    write_ontology_ipc, read_ontology_ipc,
    enforce_subsumption_monotonicity_inplace,
    enforce_disjointness_inplace,
};

fn create_sample_arrow_ontology() -> arrow::record_batch::RecordBatch {
    let mut builder = OntologyBatchBuilder::new();

    // 0: Root
    builder.append_concept(0, "INCIDENT:ROOT", "Operational Incident", "Root incident class", &[], &[], &["incident"]);
    // 1: Data (parent: 0)
    builder.append_concept(1, "INCIDENT:DATA", "Data & Storage Incident", "Storage and database failures", &[0], &[2], &["database", "db", "storage"]);
    // 2: Billing (parent: 0, disjoint: 1)
    builder.append_concept(2, "INCIDENT:BILLING", "Billing Incident", "Payment and subscription errors", &[0], &[1], &["billing", "payment", "invoice"]);
    // 3: Deadlock (parent: 1)
    builder.append_concept(3, "INCIDENT:DB_DEADLOCK", "Database Deadlock", "Lock wait timeouts on SQL queries", &[1], &[], &["deadlock", "postgres lock"]);

    builder.finish().expect("Failed to build ontology RecordBatch")
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
    enforce_disjointness_inplace(&mut probs, &disjoint_edges, 0.5);
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
        .system_one("High latency and severe packet loss on core backbone router", 4)
        .unwrap();

    println!("\nTTL Ingested Query Result:");
    println!("Selected Curie: {}", res.selected_curie);
    println!("Selected Label: {}", res.selected_label);
    println!("Confidence: {:.4}", res.confidence);
    println!("Axioms Verified: {}", res.axioms_verified);

    assert_eq!(res.selected_curie, "NetworkIncident");
    assert!(res.axioms_verified);
}

