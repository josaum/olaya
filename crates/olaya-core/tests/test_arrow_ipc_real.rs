use std::path::Path;
use olaya_core::{read_ontology_ipc, OlayaEngine};

#[test]
fn test_real_schemaorg_arrow_ipc_inference() {
    let arrow_path = Path::new("models/schemaorg.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read schemaorg.arrow");
    assert!(batch.num_rows() > 900);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from schemaorg.arrow");

    let decision = engine
        .system_one("High performance wireless noise cancelling bluetooth headphones with mic", 8)
        .expect("Decision failed");

    println!(
        "🎯 Schema.org Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_doid_arrow_ipc_inference() {
    let arrow_path = Path::new("models/doid.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read doid.arrow");
    assert!(batch.num_rows() >= 700);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from doid.arrow");

    let decision = engine
        .system_one("Patient diagnosed with acute lymphoblastic leukemia of bone marrow", 8)
        .expect("Decision failed");

    println!(
        "🎯 DOID Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_gs1_full_arrow_ipc_inference() {
    let arrow_path = Path::new("models/gs1_full.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read gs1_full.arrow");
    assert!(batch.num_rows() >= 1900);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from gs1_full.arrow");

    let decision = engine
        .system_one("Refrigerated organic whole milk dairy beverage with pasteurized cream", 8)
        .expect("Decision failed");

    println!(
        "🎯 GS1 Full Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}
