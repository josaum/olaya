//! Keeps the README's FCA example compiling against the real API.
use owl_fast_core::fca::{ConceptLattice, FormalContext};

#[test]
fn readme_fca_example_compiles_and_runs() -> anyhow::Result<()> {
    let ctx = FormalContext::from_named_pairs(&[
        ("item_1", "perishable"),
        ("item_1", "cold_chain"),
        ("item_2", "gtin_tagged"),
    ]);
    let lattice = ConceptLattice::build(ctx);
    let entities = lattice.to_ontology_entities("FCA_PRODUCT");
    let batch = olaya_core::OntologyBatchBuilder::from_entities(&entities)?;
    assert!(batch.num_rows() > 0);
    Ok(())
}
