use treewalker_gbdt::WalkerConfig;

fn test_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("TEST_ARTIFACTS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/expedia/nt500_md8"
        )
        .into()
    }))
}

#[test]
fn test_load_walker_config() {
    let dir = test_dir();
    let config_path = dir.join("walker_config.json");
    if !config_path.exists() {
        eprintln!(
            "Skipping test_load_walker_config: {} not found",
            config_path.display()
        );
        return;
    }
    let config = WalkerConfig::from_file(&config_path).unwrap();
    // Expedia nt500_md8 (a factorial-grid config): 21 features, 10 varying
    // (indices 11-20), widest test session 37 rows.
    assert_eq!(config.n_features(), 21);
    assert_eq!(config.max_group_width(), 37);
    for v in 11..=20 {
        assert!(config.is_varying(v), "feature {v} should be varying");
    }
    for c in 0..=10 {
        assert!(!config.is_varying(c), "feature {c} should be constant");
    }
}
