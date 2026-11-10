//! Convert a model for serving: build its layout, save it, and check the saved file
//! before it takes its place.
//!
//! ```text
//! cargo run --release --manifest-path experiments/layout-file/Cargo.toml \
//!     --example convert -- MODEL CONFIG LAYOUT.twl
//! ```
//!
//! loads `MODEL`, its format by its extension, with the grouping configuration
//! `CONFIG`; saves the forest's layout to `LAYOUT.twl.new`; loads that file with
//! `Forest::load_layout`; compares its predictions with the forest's, bit for bit,
//! on generated rows, single and in groups; and only then renames it to
//! `LAYOUT.twl`. Loading validates that a layout is safe to run, not that it predicts
//! as its source model does, which is what the comparison is for. The text and array
//! caps are raised to 1 GiB, as a large model needs.

use std::error::Error;
use treewalker_gbdt::{Forest, LoadOptions};

/// Rows to compare on, in each grouping.
const ROWS: usize = 4096;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, config, layout] = args.as_slice() else {
        eprintln!("usage: convert MODEL CONFIG LAYOUT.twl");
        std::process::exit(2);
    };
    let options = LoadOptions {
        max_json_bytes: 1 << 30,
        max_binary_array_bytes: 1 << 30,
        ..LoadOptions::default()
    };
    let forest = Forest::load_with(model, config, &options)?;
    // Beside the destination, so that the rename stays within one file system.
    let new = format!("{layout}.new");
    forest.save_layout(&new)?;
    let saved = Forest::load_layout(&new)?;
    let widest = forest.config().max_group_width().min(ROWS);
    let mut widths = vec![1, widest];
    widths.dedup();
    for width in widths {
        let rows = rows(&forest, width);
        let predict = |forest: &Forest| {
            let mut out = vec![0.0; rows.len() / forest.config().n_features()];
            forest.predictor().predict_fixed(&rows, width, &mut out);
            out
        };
        let (expected, actual) = (predict(&forest), predict(&saved));
        if let Some(row) =
            (0..expected.len()).find(|&i| expected[i].to_bits() != actual[i].to_bits())
        {
            std::fs::remove_file(&new)?;
            return Err(format!(
                "{new} predicts {} where {model} predicts {}, at row {row} in groups of {width}",
                actual[row], expected[row]
            )
            .into());
        }
    }
    std::fs::rename(&new, layout)?;
    println!(
        "{layout}: {} bytes; predicts as {model} on generated rows, alone and in groups of {widest}",
        std::fs::metadata(layout)?.len()
    );
    Ok(())
}

/// About [`ROWS`] rows for `forest` in groups of `width`, as its configuration has
/// them: a group's constant features take one value, and its monotonic ones run in
/// their direction, without missing values. The values are missing, small whole
/// numbers, as categories are, or fractions in ±1,000.
fn rows(forest: &Forest, width: usize) -> Vec<f64> {
    let config = forest.config();
    let n = config.n_features();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut value = || match next() % 8 {
        0 => f64::NAN,
        1..=4 => (next() % 4096) as f64,
        _ => (next() % 2_000_000) as f64 / 1000.0 - 1000.0,
    };
    let mut rows = vec![0.0; ROWS / width * width * n];
    for group in rows.chunks_mut(width * n) {
        let constant: Vec<f64> = (0..n).map(|_| value()).collect();
        for row in group.chunks_mut(n) {
            for (f, x) in row.iter_mut().enumerate() {
                *x = if config.is_varying(f) {
                    value()
                } else {
                    constant[f]
                };
            }
        }
        for f in 0..n {
            let (up, down) = (config.is_increasing(f), config.is_decreasing(f));
            if up || down {
                let mut sorted: Vec<f64> = group
                    .chunks(n)
                    .map(|row| if row[f].is_nan() { 0.0 } else { row[f] })
                    .collect();
                sorted.sort_by(f64::total_cmp);
                if down {
                    sorted.reverse();
                }
                for (row, x) in group.chunks_mut(n).zip(sorted) {
                    row[f] = x;
                }
            }
        }
    }
    rows
}
