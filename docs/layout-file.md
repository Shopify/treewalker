# Layout files (experimental)

The opt-in `layout-file` feature saves an already built forest, avoiding the source
model's decoding, categorical chain construction and temporary build allocations.
The default build and representation are unchanged. This is an experimental local
cache format, not a replacement for a portable Treelite checkpoint.

Layout files are production-only. With `research` enabled, all three public
layout-file methods return `LoadError::Unsupported`; research builds rebuild from
the source model. No parsed research nodes or category lists are stored.

`Forest::save_layout(path)` includes the grouping configuration. Use `.twl` for
layout files. Writes are
buffered and flushed, not fsynced or atomically published. Write a new file and
rename it into place; do not overwrite files used by mapped readers.

## Format

The header identifies the magic, schema version, crate version, source build ID,
architecture, endianness, pointer width, link stride, configuration
fingerprint and threshold type. Readers reject mismatches. Arrays have a u64
element count and begin on a 16-byte boundary. Numeric arrays use native bytes;
all other metadata uses explicit scalar fields and checked enum/bool tags. No
pointers, Rust enum representations, padding or mutable latency cache are saved.
The field order is specified by `layout_file::codec` and the colocated schemas
that need private access, not Rust layout.

The `TWLAYOUT` format starts at schema version 1. It stores exception words
separately from feature-relative slots. The slots
section starts with an explicit u8 width tag: 2 for u16 slots, 4 for u32 slots,
followed by that width's aligned array. Both widths are supported model-wide.
The layout has one Wide-leaf vector, empty under a Fixed plan. Fixed values live
in layout nodes: word 0 is zero; word 1 has shift bits 0-6, zero at bit 7, the
signed significand in bits 8-31 and 33-62, the leaf flag at bit 32, and sign
extension at bit 63. Readers validate the node and packed encoding before converting
a fixed value.

An eight-byte little-endian checksum ends the file. Its streaming algorithm
consumes little-endian u64 words, starting at zero, with
`state = (rotate_left(state, 5) XOR word) * 0x517cc1b727220a95` modulo 2^64.
It then consumes the zero-padded final partial word and its byte count. This
detects accidental corruption, not malicious replacement; it is not authentication.
The configuration fingerprint uses the same algorithm on its encoded fields.

The build ID is a fixed 16-hex-digit multiply-rotate checksum of `build.rs` and
every file under `src/`, in sorted relative-path order, each path followed by its
contents. It excludes Cargo.toml, which packaging rewrites. Files from different
source inputs, crate versions or incompatible targets are rejected with
`LoadError::Layout`; regenerate them from the source model after such a change.
The ID enforces a conservative source-identity policy, not authenticity. Schema
changes still require a version bump. Rank-table and leaf-width schemas remain
colocated in `load/layout/layout_file.rs`.

Production API tests live in the excluded `experiments/layout-file` crate, whose
only treewalker feature is `layout-file`. CI runs it separately to avoid workspace
research-feature unification. Tests cover owned/mapped round trips and malformed
bodies with independently recomputed checksums. In-crate white-box tests call
internal entry points for this same production format.

Open design questions include cache compatibility across source changes,
configurable limits and atomic/durable publication.
