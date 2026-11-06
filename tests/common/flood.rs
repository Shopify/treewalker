//! Keys that collided in the loader's hash tables before 3.2.2, when the tables
//! hashed with fixed constants: a model could choose them, and decoding took time
//! quadratic in their number. The tables now hash with a key drawn per load or per
//! process, so these sets cost what random keys do; the tests check that they still
//! do, as the sets a reverted or copied fixed hash would fail on.

/// The multiplicative inverse of an odd 64-bit `k`, by Newton's iteration.
const fn inverse(k: u64) -> u64 {
    let mut inv = k;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(k.wrapping_mul(inv)));
        i += 1;
    }
    inv
}

/// A split's flags in the split table's key, as the decoders set them: missing sent
/// left, and a split on a feature the configuration declares varying.
pub const FLAG_DEFAULT_LEFT: u8 = 0x01;
pub const FLAG_VARYING: u8 = 0x10;

/// `n` distinct finite `f64` thresholds of numeric splits on `feature` with `flags`,
/// whose keys all hashed to one slot of the split table's fixed hash (3.2.1
/// `Splits::slot`: `h = (bits ^ key.rotl(40)) * K; (h ^ h >> 29) * K`, with `key =
/// feature << 8 | flags` and the slot the top bits) whatever its size up to 2^32
/// slots: the hash's top 32 bits agree.
pub fn colliding_thresholds(n: usize, feature: u32, flags: u8) -> Vec<f64> {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    const K_INV: u64 = inverse(K);
    let key = (u64::from(feature) << 8 | u64::from(flags)).rotate_left(40);
    let mut out = Vec::with_capacity(n);
    let mut t = 0u64;
    while out.len() < n {
        let h2 = 0x5A5A_5A5A_u64 << 32 | t;
        t += 1;
        // Undo the second multiply, then `h ^ h >> 29` (29 * 3 > 64), then the first
        // multiply and the key.
        let h1 = h2.wrapping_mul(K_INV);
        let h1 = h1 ^ h1 >> 29 ^ h1 >> 58;
        let bits = h1.wrapping_mul(K_INV) ^ key;
        let value = f64::from_bits(bits);
        if value.is_finite() {
            out.push(value);
        }
    }
    out
}

/// `n` distinct non-negative node IDs that all probed one group of a hashbrown
/// table under `rustc_hash::FxHasher` 2.1 (`hash = (id * K).rotl(26)`, bucket from
/// the low bits, control byte from the top 7), whatever the table's size up to 2^32
/// buckets: the JSON node map's hasher before 3.2.2.
pub fn colliding_node_ids(n: usize) -> Vec<u64> {
    const K: u64 = 0xf135_7aea_2e62_a9c5;
    const K_INV: u64 = inverse(K);
    let mut out = Vec::with_capacity(n);
    let mut j = 0u64;
    while out.len() < n {
        // Low 32 bits and top 7 bits fixed; bits 32..57 vary.
        assert!(j < 1 << 25, "the colliding IDs run out");
        let hash = 0x3C3C_3C3C_u64 | j << 32;
        j += 1;
        let id = hash.rotate_right(26).wrapping_mul(K_INV);
        if i64::try_from(id).is_ok() {
            out.push(id);
        }
    }
    out
}
