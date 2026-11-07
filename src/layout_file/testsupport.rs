//! Public synthetic models and checksummed corruption, never serving artifacts.
use crate::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

pub(crate) fn fixture() -> Forest {
    Forest::from_bytes(
        include_bytes!("../../tests/fixtures/import/categories_left.bin"),
        ModelFormat::TreeliteBinaryV4,
        WalkerConfig::builder(2)
            .max_group_width(128)
            .varying([0])
            .build()
            .unwrap(),
        &LoadOptions::default(),
    )
    .unwrap()
}

pub(crate) fn forest(features: usize, varying: bool, trees: &[String]) -> Forest {
    let json = crate::load::json::regressor(false, features, trees);
    Forest::from_bytes(
        json.as_bytes(),
        ModelFormat::TreeliteJson,
        WalkerConfig::builder(features)
            .max_group_width(4)
            .varying(0..if varying { features } else { 0 })
            .build()
            .unwrap(),
        &LoadOptions::default(),
    )
    .unwrap()
}

pub(crate) fn numeric(features: usize, varying: bool) -> Forest {
    let trees: Vec<_> =
        (0..features)
            .map(|f| {
                crate::load::json::stump(
        r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
        true, (3, 1)).replace("\"split_feature_id\": 0", &format!("\"split_feature_id\": {f}"))
            })
            .collect();
    forest(features, varying, &trees)
}

pub(crate) fn categorical(features: usize) -> Forest {
    let trees: Vec<_> = (0..features)
        .map(|f| {
            category_tree(&[1, 2], (3, 1)).replace(
                "\"split_feature_id\": 0",
                &format!("\"split_feature_id\": {f}"),
            )
        })
        .collect();
    forest(features, true, &trees)
}

pub(crate) fn category_tree(categories: &[u32], counts: (u32, u32)) -> String {
    crate::load::json::stump(
        &format!(
            r#""node_type": "categorical_test_node", "category_list": {categories:?}, "category_list_right_child": false"#
        ),
        true,
        counts,
    )
}

pub(crate) struct File(pub(crate) std::path::PathBuf);
impl File {
    pub(crate) fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "treewalker-corruption-{}-{}.twl",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )))
    }
}
impl Drop for File {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Save recomputes the checksum, so an error must come from structural validation.
pub(crate) fn rejects(forest: &Forest, reason: &str) {
    let file = File::new();
    forest.save_layout_impl(&file.0).unwrap();
    let error = Forest::load_layout_impl(&file.0).unwrap_err();
    assert!(
        matches!(error, LoadError::Layout(_)) && error.to_string().contains(reason),
        "{error}"
    );
    #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
    {
        // SAFETY: no writes/truncation occur during this call or while its
        // result lives; all prior mapping owners were dropped before saving.
        let error = unsafe { Forest::map_layout_impl(&file.0) }.unwrap_err();
        assert!(
            matches!(error, LoadError::Layout(_)) && error.to_string().contains(reason),
            "{error}"
        );
    }
}
