//! Packed low-six-bit scale codes and one high-bit mask per eight scales.
//! Vectors already require eight bits and store those bytes directly.
//! Training scales are nonnegative E4M3 bytes, so their canonical matrix
//! representation uses seven bits. Host tensors may carry an eighth-bit plane.
pub const HEADER_BYTES: usize = 128;
pub const POSITIVE_TILE_BYTES: usize = 896;
pub const FULL_TILE_BYTES: usize = 1024;
pub fn encoded_len(rows: usize, k: usize) -> usize {
    if rows == 1 {
        128 + k / 16
    } else {
        128 + rowwise_encoded_len(rows, k)
    }
}
pub const fn rowwise_encoded_len(rows: usize, k: usize) -> usize {
    rows.div_ceil(128) * k.div_ceil(128) * POSITIVE_TILE_BYTES
}
#[inline(always)]
fn code_offset(r: u32) -> u32 {
    (r / 16) * 96 + (r % 8) * 12 + (r % 16 / 8) * 3
}
#[inline(always)]
fn compact6(w: u32) -> u32 {
    (w & 63) | ((w & 0x3f00) >> 2) | ((w & 0x3f0000) >> 4) | ((w & 0x3f000000) >> 6)
}
#[inline(always)]
fn high1(w: u32) -> u8 {
    (((w >> 6) & 0x01010101).wrapping_mul(0x01020408) >> 24) as u8
}
#[inline(always)]
fn high2(w: u32) -> u8 {
    let v = (w >> 6) & 0x03030303;
    let v = (v | (v >> 6)) & 0x000f000f;
    ((v | (v >> 12)) & 255) as u8
}
pub fn encode_host(raw: &[u8], rows: usize, k: usize) -> Vec<u8> {
    assert!(k >= 128 && k.is_multiple_of(128));
    assert!(rows * k <= u32::MAX as usize);
    assert!(raw.len() <= rows * k / 16);
    if rows == 1 {
        return encode_vector(raw, k);
    }
    encode_matrix(raw, rows, k, k / 16, true)
}
pub fn encode_rowwise_host(raw: &[u8], rows: usize, k: usize) -> Vec<u8> {
    assert!(k > 0 && k % 16 == 0);
    assert_eq!(raw.len(), rows * (k / 16));
    encode_matrix(raw, rows, k.div_ceil(128) * 128, k / 16, false)
}
fn encode_matrix(raw: &[u8], rows: usize, k: usize, source_groups: usize, header: bool) -> Vec<u8> {
    let bytes = if raw.iter().any(|&v| v >= 128) { 2 } else { 1 };
    let stride = 768 + 128 * bytes;
    let offset = if header { 128 } else { 0 };
    let padded = rows.div_ceil(128) * 128;
    let tiles = padded / 128 * (k / 128);
    let mut out = vec![0; offset + tiles * stride];
    if header {
        out[..4].copy_from_slice(&(k as u32).to_le_bytes());
        out[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
        out[8] = (k / 16).trailing_zeros() as u8;
        out[9] = 6 + bytes as u8;
    }
    for row in 0..padded {
        for kg in (0..k / 16).step_by(8) {
            let v: [u8; 8] = std::array::from_fn(|i| {
                if row < rows && kg + i < source_groups {
                    raw.get(row * source_groups + kg + i).copied().unwrap_or(0)
                } else {
                    0
                }
            });
            let a = u32::from_le_bytes(v[..4].try_into().unwrap());
            let b = u32::from_le_bytes(v[4..].try_into().unwrap());
            let t = row / 128 * (k / 128) + kg / 8;
            let at = offset + t * stride + code_offset(row as u32 % 128) as usize;
            out[at..at + 3].copy_from_slice(&compact6(a).to_le_bytes()[..3]);
            out[at + 6..at + 9].copy_from_slice(&compact6(b).to_le_bytes()[..3]);
            let at = offset + t * stride + 768 + (row % 128) * bytes;
            if bytes == 1 {
                out[at] = high1(a) | (high1(b) << 4);
            } else {
                out[at] = high2(a);
                out[at + 1] = high2(b);
            }
        }
    }
    out
}
#[inline(always)]
unsafe fn read6(data: *const u8, at: usize, kg: u32) -> u8 {
    let shift = (kg % 4) * 6;
    let byte = at + (shift / 8) as usize;
    let bits = shift % 8;
    let a = unsafe { *data.add(byte) } as u16;
    let b = if bits > 2 {
        (unsafe { *data.add(byte + 1) }) as u16
    } else {
        0
    };
    (((a | (b << 8)) >> bits) & 63) as u8
}
#[inline(always)]
unsafe fn matrix_scale_at_bits(data: *const u8, k: u32, row: u32, kg: u32, bytes: u32) -> u8 {
    let t = row / 128 * (k / 128) + kg / 8;
    let r = row % 128;
    let stride = 768 + 128 * bytes;
    let q = unsafe {
        read6(
            data,
            (t * stride + code_offset(r) + (kg % 8 / 4) * 6) as usize,
            kg,
        )
    };
    let at = t * stride + 768 + r * bytes;
    if bytes == 1 {
        let mask = unsafe { *data.add(at as usize) };
        q | (((mask >> (kg % 8)) & 1) << 6)
    } else {
        let mask = unsafe { *data.add((at + kg % 8 / 4) as usize) };
        q | (((mask >> ((kg % 4) * 2)) & 3) << 6)
    }
}
#[inline(always)]
pub(crate) unsafe fn matrix_scale_at_data(
    data: *const u8,
    _rows: u32,
    k: u32,
    row: u32,
    kg: u32,
) -> u8 {
    unsafe { matrix_scale_at_bits(data, k, row, kg, 1) }
}
#[inline(always)]
pub(crate) unsafe fn matrix_scale_at(data: *const u8, _rows: u32, k: u32, row: u32, kg: u32) -> u8 {
    let bytes = u32::from(unsafe { *data.add(9) }) - 6;
    unsafe { matrix_scale_at_bits(data.add(128), k, row, kg, bytes) }
}
#[inline(always)]
pub fn scale_at(s: &[u8], g: usize) -> u8 {
    if s[8] == 255 {
        return vector_scale_at(s, g);
    }
    let k = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let rows = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
    let (row, kg) = flat_scale_coordinates(g as u32, k);
    unsafe { matrix_scale_at(s.as_ptr(), rows, k, row, kg) }
}
// The header carries the actual K. Its trailing-zero hint describes a
// shift only for power-of-two widths; e.g. K=5504 has 344 groups per row.
#[inline(always)]
fn flat_scale_coordinates(g: u32, k: u32) -> (u32, u32) {
    let groups = k / 16;
    if groups.is_power_of_two() {
        (g >> groups.trailing_zeros(), g & (groups - 1))
    } else {
        let row = g / groups;
        (row, g - row * groups)
    }
}
#[inline(always)]
pub fn scale4(s: &[u8], g: usize) -> u32 {
    u32::from(scale_at(s, g))
        | (u32::from(scale_at(s, g + 1)) << 8)
        | (u32::from(scale_at(s, g + 2)) << 16)
        | (u32::from(scale_at(s, g + 3)) << 24)
}
fn encode_vector(raw: &[u8], k: usize) -> Vec<u8> {
    let mut out = vec![0; 128 + k / 16];
    out[..4].copy_from_slice(&(k as u32).to_le_bytes());
    out[4..8].copy_from_slice(&1u32.to_le_bytes());
    out[8] = 255;
    out[9] = 8;
    out[128..128 + raw.len()].copy_from_slice(raw);
    out
}
#[inline(always)]
pub(crate) fn vector_scale_at(s: &[u8], g: usize) -> u8 {
    s[128 + g]
}
#[inline(always)]
unsafe fn store24(data: *mut u8, at: u32, v: u32) {
    unsafe {
        if at & 1 == 0 {
            *data.add(at as usize).cast::<u16>() = v as u16;
            *data.add((at + 2) as usize) = (v >> 16) as u8;
        } else {
            *data.add(at as usize) = v as u8;
            *data.add((at + 1) as usize).cast::<u16>() = (v >> 8) as u16;
        }
    }
}
#[inline(always)]
pub(crate) unsafe fn write_record(
    data: *mut u8,
    _rows: u32,
    k: u32,
    row: u32,
    kg: u32,
    a: u32,
    b: u32,
) {
    let t = row / 128 * (k / 128) + kg / 8;
    let r = row % 128;
    let at = t * 896 + code_offset(r);
    unsafe {
        store24(data, at, compact6(a));
        store24(data, at + 6, compact6(b));
        *data.add((t * 896 + 768 + r) as usize) = high1(a) | (high1(b) << 4);
    }
}
#[inline(always)]
fn warp_words(bits: u8) -> (u32, u32) {
    let mask: u32;
    unsafe {
        cuda_device::ptx_asm!("activemask.b32 %0;",out("=r")mask,options(register_only));
    }
    let half = cuda_device::warp::lane_id() % 2;
    macro_rules! get {
        ($i:expr) => {{
            let i = half * 16 + $i;
            let present = mask & (1u32 << i) != 0;
            let v =
                cuda_device::warp::shuffle_sync(mask, u32::from(bits), if present { i } else { 0 });
            if present { v } else { 0 }
        }};
    }
    let word = get!(0) | (get!(4) << 8) | (get!(8) << 16) | (get!(12) << 24);
    (
        cuda_device::warp::shuffle_sync(mask, word, 0),
        cuda_device::warp::shuffle_sync(mask, word, if mask & 2 != 0 { 1 } else { 0 }),
    )
}
#[inline(always)]
pub(crate) unsafe fn store_row_warp(data: *mut u8, rows: u32, k: u32, row: u32, kg: u32, bits: u8) {
    let (a, b) = warp_words(bits);
    if cuda_device::warp::lane_id() == 0 {
        unsafe {
            write_record(data, rows, k, row, kg, a, b);
        }
    }
}
#[inline(always)]
pub(crate) unsafe fn store_warp(s: *mut u8, g: usize, bits: u8) {
    let k = unsafe { *s.cast::<u32>() };
    let rows = unsafe { *s.cast::<u32>().add(1) };
    if rows == 1 {
        let (a, b) = warp_words(bits);
        if cuda_device::warp::lane_id() == 0 {
            let at = 128 + g as u32 / 8 * 8;
            unsafe {
                *s.add(at as usize).cast::<u32>() = a;
                *s.add((at + 4) as usize).cast::<u32>() = b;
            }
        }
    } else {
        let (row, kg) = flat_scale_coordinates(g as u32, k);
        unsafe {
            store_row_warp(s.add(128), rows, k, row, kg, bits);
        }
    }
}
