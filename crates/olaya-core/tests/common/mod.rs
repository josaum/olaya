use std::path::PathBuf;

/// Cargo runs integration tests from the crate directory, while tracked model
/// fixtures live at the workspace root. Never silently skip a missing fixture.
pub fn model_fixture(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name);
    assert!(
        path.is_file(),
        "Required tracked Arrow fixture is missing: {}. Restore it from Git before running this test.",
        path.display()
    );
    path
}
