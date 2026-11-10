# Layout files (experimental)

The opt-in `layout-file` feature saves an already built forest, avoiding the source
model's decoding, categorical chain construction and temporary build allocations.
The default build and representation are unchanged. This is an experimental local
cache format, not a replacement for a portable Treelite checkpoint.

Layout files are production-only. With `research` enabled, all three public
layout-file methods return `LoadError::Unsupported`; research builds rebuild from
the source model. No parsed research nodes or category lists are stored.

`Forest::save_layout(path)` includes the grouping configuration. Use `.twl` for
layout files. Writes are buffered and flushed, not fsynced or atomically published.
Write a new file and rename it into place; do not overwrite files used by mapped
readers.

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
prefixes, categorical domains, matching exception slot/word lengths, monotone bounded
offsets and relative slots within each feature's extent,
predicate coverage and exact-sum bounds. Packed fixed
leaf nodes must round-trip through `fixed_value`/`fixed`. Their values must have a
canonical zero or an odd nonzero significand with magnitude below 2^53 and a shift
below 125; shifted magnitudes and forest sums must fit the fixed plan. Leaves are
identified by their flag before split mask/sign checks: fixed value bits are not mask
fields. Wide leaves retain clear mask/sign bits and checked table indices.
Predictors rebuild chunks and scratch; neither allocation sizes nor unchecked-kernel
invariants are trusted from disk.
The checksum is an additional corruption check, never a substitute for validation.

Writable numeric/categorical and held-code slots have disjoint owners. Thresholds,
codebooks, offset tables and exception tables also have non-overlapping per-feature
ownership. Ownership is checked before per-feature scans.

Slot ownership decides whether a node needs a varying mask and bounds its feature
and predicate kind. The file stores no mask plan: the loader reconstructs chain
points from nodes and builds the plan from validated bins/predicates, as a source
build does, so there are no untrusted mask-plan ranges to follow. Tree extents
and their disjoint coverage are checked before visiting any tree's nodes.

Validation is O(R log R) in the file's records: ownership sorts and per-split
binary searches are not linear. It allocates O(R) temporary heap in addition to
the decoded arrays, including ownership tables and traversal scratch. The mask
plan it builds stays with the forest, on the heap under either loader. Its flip
lists take 4 bytes per code of a varying categorical feature and at most 8 per
exception for each chain of the exception's word, of which a word holds at most 32.
The file's own codes and exceptions bound the plan, not a stored size, so a plan
can be larger than its file.

Measured with a counting allocator in release builds, retaining only the loaded
forest (no predictor), a private model with many categorical splits had peak/live
heap ratios of 1.00x owned and 1.05x mapped with its own configuration. With every
feature varying, they were 1.37x and 1.66x: predicate-heavy validation can require
substantial temporary heap, most of it here the capacity the plan's flip list
reserves as it grows, which the build then returns. A public generated forest with
1,024 two-sided categorical predicates used 0.313/0.251 MiB peak/live owned and
0.188/0.134 MiB mapped. These are heap measurements, not RSS; file-backed mapped
pages are not counted.

Checks needed only for prediction agreement are deliberately omitted: threshold
and codebook order, code-zero exceptions and agreement between a predicate's
two split sides. Bounds and predicate (feature, kind) ordering remain safety
requirements. A hostile writer can choose arbitrary
predictions; neither structural validation nor the checksum authenticates a model.

On 64-bit Unix, `unsafe Forest::map_layout(path)` borrows the layout nodes,
exception words and either width of relative exception slots from a read-only
private mapping. Fixed values travel in their mapped leaf nodes. Other metadata,
including wide leaves, remains owned. Validation and checksum checks are identical
to the owned loader. Mapping lifetime is retained by the forest and all
predictors/clones; array access is an unconditional pointer/length slice, not an
owned/mapped branch in the kernel. Without `layout-file`, arrays remain ordinary Vecs.

The **caller must ensure that no thread or process writes or truncates the backing
file during the call or until all returned forests, clones and predictors drop**.
Neither read-only access nor MAP_PRIVATE enforces that contract. Replacing a name
by atomic rename is fine if the old file remains immutable. This unsafe API is
deliberate: validation cannot protect a live slice from subsequent external writes.
An immutable artifact/ownership abstraction for a safe mapped API remains open.

## Format

The header identifies the magic, schema version, crate version, source build ID,
architecture, endianness, pointer width, link stride, configuration
fingerprint and threshold type. Readers reject mismatches. Arrays have a u64
element count and begin on a 16-byte boundary. Numeric arrays use native bytes;
all other metadata uses explicit scalar fields and checked enum/bool tags. No
pointers, Rust enum representations, padding or mutable latency cache are saved.
The field order is specified by `layout_file::codec` and the colocated schemas
that need private access, not Rust layout.

The `TWLAYOUT` format is at schema version 2; version 1 also stored the mask plan.
It stores exception words separately from feature-relative slots. The slots
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
