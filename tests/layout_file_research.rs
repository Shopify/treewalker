//! The public layout-file API must never expose incomplete research data.
#![cfg(all(feature = "layout-file", feature = "research"))]

use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

#[test]
fn research_rebuilds_from_source_instead_of_using_layout_files() {
    let forest = Forest::from_bytes(
        include_bytes!("fixtures/import/categories_left.bin"),
        ModelFormat::TreeliteBinaryV4,
        WalkerConfig::builder(2)
            .max_group_width(4)
            .all_varying()
            .build()
            .unwrap(),
        &LoadOptions::default(),
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("treewalker-research-{}.twl", std::process::id()));
    let rejected = |result: Result<Forest, LoadError>| {
        assert!(matches!(result, Err(LoadError::Unsupported(ref s)) if s ==
            "layout files are production-only; research builds rebuild from the source model"));
    };
    assert!(matches!(
        forest.save_layout(&path),
        Err(LoadError::Unsupported(_))
    ));
    rejected(Forest::load_layout(&path));
    #[cfg(all(unix, target_pointer_width = "64"))]
    // SAFETY: no file is written or truncated; rejection precedes opening it.
    rejected(unsafe { Forest::map_layout(&path) });
    assert!(!path.exists());
}
