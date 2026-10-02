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

#[test]
fn test_real_fibo_arrow_ipc_inference() {
    let arrow_path = Path::new("models/fibo.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read fibo.arrow");
    assert!(batch.num_rows() >= 17000);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from fibo.arrow");

    let decision = engine
        .system_one("Credit default swap derivative contract hedging corporate bond default risk", 8)
        .expect("Decision failed");

    println!(
        "🎯 FIBO Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_ontopix_arrow_ipc_inference() {
    let arrow_path = Path::new("models/ontopix.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read ontopix.arrow");
    assert!(batch.num_rows() >= 20);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from ontopix.arrow");

    let decision = engine
        .system_one("Transação de pagamento instantâneo com contestação por fraude Mecanismo Especial de Devolução", 8)
        .expect("Decision failed");

    println!(
        "🎯 OntoPix Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_apto_arrow_ipc_inference() {
    let arrow_path = Path::new("models/apto.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read apto.arrow");
    assert!(batch.num_rows() >= 20);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from apto.arrow");

    let decision = engine
        .system_one("Contrato futuro de boi gordo B3 com base no indicador Cepea Esalq", 8)
        .expect("Decision failed");

    println!(
        "🎯 APTO Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_c2sim_arrow_ipc_inference() {
    let arrow_path = Path::new("models/c2sim.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read c2sim.arrow");
    assert!(batch.num_rows() >= 20);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from c2sim.arrow");

    let decision = engine
        .system_one("Blue force tracker position report from armored mechanized company", 8)
        .expect("Decision failed");

    println!(
        "🎯 C2SIM Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_mondo_arrow_ipc_inference() {
    let arrow_path = Path::new("models/mondo.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read mondo.arrow");
    assert!(batch.num_rows() >= 18000);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from mondo.arrow");

    let decision = engine
        .system_one("Severe pediatric neurodegenerative lysosomal storage disorder with sphingomyelinase deficiency", 8)
        .expect("Decision failed");

    println!(
        "🎯 MONDO 19k Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}

#[test]
fn test_real_w3c_core_arrow_ipc_inference() {
    let arrow_path = Path::new("models/w3c_core.arrow");
    if !arrow_path.exists() {
        return;
    }

    let batch = read_ontology_ipc(arrow_path).expect("Failed to read w3c_core.arrow");
    assert!(batch.num_rows() >= 90);

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine from w3c_core.arrow");

    let decision = engine
        .system_one("Data provenance activity generating derivative dataset catalog record with historical agent attribution", 8)
        .expect("Decision failed");

    println!(
        "🎯 W3C Core Zero-Copy Decision: {} [{}] (Confidence: {:.2})",
        decision.selected_label, decision.selected_curie, decision.confidence
    );
    assert!(decision.axioms_verified);
}


