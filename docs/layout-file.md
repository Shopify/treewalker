# Layout files (experimental)

The opt-in `layout-file` feature saves an already built forest, avoiding the source
model's decoding, categorical chain construction and temporary build allocations.
The default build and representation are unchanged. This is an experimental local
cache format, not a replacement for a portable Treelite checkpoint.

`Forest::save_layout(path)` includes the grouping configuration. Use `.twl` for
layout files. Writes are
buffered and flushed, not fsynced or atomically published. Write a new file and
rename it into place; do not overwrite files used by mapped readers.

`Forest::load_layout(path)` streams into owned vectors and validates before
returning. I/O failures use `LoadError::Io`, incompatible/corrupt layouts use
`LoadError::Layout`, and the file-size cap uses `LoadError::Limit`. The initial
limits are 4 GiB per file, 2^28 entries per array (also bounded by file size), and
2^20 rows per group. No timing or heap-inventory API is exposed.
Readers also apply a conservative scratch address-space bound, particularly for
32-bit targets; an otherwise valid large layout may need a 64-bit reader.
Array counts are bounded by remaining encoded bytes, not Rust object sizes.
POD byte extents are checked before reserving or initializing their vectors.

Validation checks the configuration, precision, dummy behavior, node fields, links,
leaves, full binary tree topology and coverage, recomputed scratch counts, constant
prefixes, numeric threshold order, categorical domains, matching exception slot/word
lengths, monotone bounded offsets and relative slots within each feature's extent,
mask-chain and flip ranges, predicate coverage and exact-sum bounds. Packed fixed
leaf nodes must round-trip through `fixed_value`/`fixed`. Their values must have a
canonical zero or an odd nonzero significand with magnitude below 2^53 and a shift
below 125; shifted magnitudes and forest sums must fit the fixed plan. Leaves are
identified by their flag before split mask/sign checks: fixed value bits are not mask
fields. Wide leaves retain clear mask/sign bits and checked table indices. Research
data has separate parsed-node/category checks. Predictors rebuild chunks and scratch;
neither allocation sizes nor unchecked-kernel invariants are trusted from disk.
The checksum is an additional corruption check, never a substitute for validation.

Writable numeric/categorical and held-code slots have disjoint owners, and code 0
has no exceptions: fresh bins already represent that code, so an unchanged-code
fill may skip writing. Thresholds, codebooks, offset tables and exception tables
also have non-overlapping per-feature ownership. Ownership is checked before any
per-feature scans; research category lists are indexed and checked once.

Slot ownership decides whether a node needs a varying mask. Numeric mask fields
must match the threshold, default direction and side encoded by the node. For
categorical predicates, the constant encodings of both heavy sides must agree on
all codes. The loader reconstructs chain points from nodes, rebuilds the mask plan
from validated bins/predicates, and compares every feature, chain, point and flip;
it never follows untrusted mask-plan ranges. Features are unique. Tree extents and
their disjoint coverage are checked before visiting any tree's nodes.

For two-sided categorical predicates, a per-slot (code, word) index is built once,
only if needed. Defaults cover all codes absent from both slots, and sparse
two-pointer merges visit only their exception slices. To keep long shared chains
from repeating those merges, a linear merge allowance switches to batched queries
per (slot, shift). Each sweep toggles the codes whose truth differs from the
default's in a keyed 128-bit set hash, the XOR of seeded random keys, and the
sides' hashes must be equal. Their agreement is semantic, single-row against
grouped predictions, never an index the kernel trusts, so a collision (about
2^-128 a comparison) is no safety risk. At most 32 fields per slot and one toggle
per event bound work and memory by stored records, without rejecting valid saved
forests when the merge allowance runs out.

Research nodes must agree with layout leaf/split kinds and links. Finite research
leaves must reconstruct the exact-sum plan and the stored leaf encodings before
research prediction can convert them. CI runs release tests with both `research`
and `layout-file`, in addition to its existing feature/ISA checks.
Production-only paths are covered by clippy and an external consumer; the benchmark
member unconditionally depends on `research` and is not a production-only test host.

## Format

The header identifies the magic, schema version, crate version, architecture,
endianness, pointer width, link stride, research-tail presence, configuration
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

Research builds have a separate schema tail and retain category lists and parsed
nodes. Files cannot move between production and research schemas. Schema changes
require a version bump, even when the crate version is unchanged. In particular,
rank-table and leaf-width changes belong in `load/layout/layout_file.rs`; those arrays
are intentionally not scattered through the generic format code.

Open design questions include cache compatibility across patch releases, explicit
build identifiers, configurable limits, atomic/durable publication, and whether
research data should be a separate optional artifact.
