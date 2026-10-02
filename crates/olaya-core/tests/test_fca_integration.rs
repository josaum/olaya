use olaya_core::{OlayaEngine, OntologyBatchBuilder};
use owl_fast_core::fca::{ConceptLattice, FormalContext};

#[test]
fn test_fca_concept_lattice_to_arrow_decision_engine() {
    // 1. Build a formal context of GS1 / e-commerce product items and attributes
    let context_pairs = [
        ("OrangeJuice", "Beverage"),
        ("OrangeJuice", "FoodAndBeverage"),
        ("OrangeJuice", "ContainsCitrus"),
        ("OrangeJuice", "Refrigerated"),
        ("AppleCider", "Beverage"),
        ("AppleCider", "FoodAndBeverage"),
        ("AppleCider", "Refrigerated"),
        ("ColaCan", "Beverage"),
        ("ColaCan", "FoodAndBeverage"),
        ("ColaCan", "Carbonated"),
        ("MineralWater", "Beverage"),
        ("MineralWater", "FoodAndBeverage"),
        ("TShirt", "Wearable"),
        ("TShirt", "CottonFabric"),
        ("WoolSweater", "Wearable"),
        ("WoolSweater", "ColdWeather"),
    ];

    let ctx = FormalContext::from_named_pairs(&context_pairs);
    assert_eq!(ctx.num_objects(), 6);
    assert_eq!(ctx.num_attributes(), 8);

    // 2. Build the concept lattice using Close-by-One (CbO)
    let lattice = ConceptLattice::build(ctx);
    assert!(lattice.concepts.len() >= 6);

    // Verify Top and Bottom concepts
    assert_eq!(
        lattice.concepts[lattice.top].extent_size(),
        lattice.context.num_objects(),
        "Top concept must contain all objects in extent"
    );
    assert_eq!(
        lattice.concepts[lattice.bottom].intent_size(),
        lattice.context.num_attributes(),
        "Bottom concept must contain all attributes in intent"
    );

    // Verify Meet and Join properties
    let meet_idx = lattice.meet(lattice.top, lattice.bottom);
    assert_eq!(meet_idx, lattice.bottom);

    let join_idx = lattice.join(lattice.top, lattice.bottom);
    assert_eq!(join_idx, lattice.top);

    // 3. Extract sound implications
    let implications = lattice.compute_canonical_implications();
    assert!(!implications.is_empty());
    for imp in &implications {
        assert_eq!(
            imp.confidence, 1.0,
            "All lattice implications must be sound (100% confidence)"
        );
    }

    // 4. Convert FCA Concept Lattice to OWL Entities with full covering DAG relations
    let entities = lattice.to_ontology_entities("gs1_fca");
    assert_eq!(entities.len(), lattice.concepts.len());

    // 5. Ingest into Arrow RecordBatch
    let batch = OntologyBatchBuilder::from_entities(&entities)
        .expect("Failed to convert FCA entities to Arrow RecordBatch");
    assert_eq!(batch.num_rows(), entities.len());

    // 6. Initialize OlayaEngine on the Arrow batch
    let engine =
        OlayaEngine::new(batch).expect("Failed to initialize OlayaEngine on FCA Arrow RecordBatch");

    // 7. Query the FCA-derived decision engine
    let res = engine
        .system_one("Freshly squeezed chilled citrus juice bottle", 4)
        .expect("System 1 execution on FCA engine failed");

    println!("\nFCA Decision Result:");
    println!("Selected Curie: {}", res.selected_curie);
    println!("Selected Label: {}", res.selected_label);
    println!("Confidence: {:.4}", res.confidence);
    println!("Axioms Verified: {}", res.axioms_verified);

    assert!(res.axioms_verified);
    assert!(res.confidence > 0.0);
}

#[test]
fn test_induced_procurement_arrow_decision_engine() {
    let arrow_path = std::path::Path::new("models/induced_procurement.arrow");
    if !arrow_path.exists() {
        return;
    }
    let batch = olaya_core::read_ontology_ipc(arrow_path).expect("Failed to read induced_procurement.arrow");
    assert!(batch.num_rows() >= 10);

    let engine = OlayaEngine::new(batch).expect("Failed to build engine");
    let res = engine
        .system_one("High value serialized laptop with lithium ion UN3481 battery", 4)
        .expect("Decision failed");

    println!(
        "🎯 Induced Procurement Decision: {} [{}] (Confidence: {:.2})",
        res.selected_label, res.selected_curie, res.confidence
    );
    assert!(res.axioms_verified);
}

