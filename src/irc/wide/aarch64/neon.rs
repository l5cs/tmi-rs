// The method used for emulating movemask is explained in the following article (the link goes to a table of operations):
// https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon#:~:text=Consider%20the%C2%A0result%20in%20both%20cases%20as%20the%20result%20of%20PMOVMSKB%20or%20shrn
//
// Archived link: https://web.archive.org/web/20230603011837/https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
//
// For example, to find the first `=` character in `s`:
//
// The implementation splits `s` into 16-byte chunks, loading each chunk into a single 8x16 vector.
//
// The resulting 8x16 vectors are compared against the pre-filled vector of a single character using `vceqq_u8`.
// Next, the 8x16 is reinterpreted as 16x8, to which we apply `vshrn_n_u16`.
//
// `vshrn_n_u16` performs a "vector shift right by constant and narrow".
// The way I understand it is that for every 16-bit element in the vector,
// it trims the 4 most significant bits + 4 least significant bits:
//
// ```text,ignore
// # for a single element:
// 1111111100000000 -> shift right by 4
// 0000111111110000 -> narrow to u8
//         11110000
// ```
//
// If we count the number of bits in the vector before the first bit set to `1`,
// then divide that number by `4`, we get the same result as a `movemask + ctz` would give us.
//
// So the last step is to reinterpret the resulting 8x8 vector as a single 64-bit integer,
// which is our mask.
// Just like before, we can check for the presence of the "needle" by comparing the mask
// against `0`.
// To obtain the position of the charater, divide its trailing zeros by 4.

use core::arch::aarch64::{
  uint8x16_t, vceqq_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8,
  vshrn_n_u16,
};

// NOTE: neon has no alignment requirements for loads,
//       but alignment is still better than no alignment.

#[repr(align(16))]
struct Align16([u8; 16]);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector(uint8x16_t);

impl Vector {
  /// Size in bytes.
  pub const SIZE: usize = 16;

  #[inline]
  pub const fn fill(v: u8) -> Self {
    Self(unsafe { core::mem::transmute::<[u8; 16], uint8x16_t>([v; 16]) })
  }

  /// Load 16 bytes from the given slice into a vector.
  ///
  /// `data[offset..].len()` must be greater than 16 bytes.
  #[inline(always)]
  pub fn load_unaligned(data: &[u8], offset: usize) -> Self {
    unsafe {
      debug_assert!(data[offset..].len() >= 16);
      Self(vld1q_u8(data.as_ptr().add(offset)))
    }
  }

  /// Load 16 bytes from the given slice into a vector.
  ///
  /// `data[offset..].len()` must be greater than 16 bytes.
  /// The data must be 16-byte aligned.
  #[inline(always)]
  pub fn load_aligned(data: &[u8], offset: usize) -> Self {
    unsafe {
      debug_assert!(data[offset..].len() >= 16);
      debug_assert!(data.as_ptr().add(offset).addr().is_multiple_of(16));
      Self(vld1q_u8(data.as_ptr().add(offset)))
    }
  }

  /// Load at most 16 bytes from the given slice into a vector
  /// by loading it into an intermediate buffer on the stack.
  #[inline(always)]
  pub fn load_unaligned_remainder(data: &[u8], offset: usize) -> Self {
    unsafe {
      let mut buf = Align16([0; 16]);
      buf.0[..data.len() - offset].copy_from_slice(&data[offset..]);

      Self(vld1q_u8(buf.0.as_ptr()))
    }
  }

  #[inline(always)]
  pub fn eq(self, byte: u8) -> Self {
    unsafe { Self(vceqq_u8(self.0, Self::fill(byte).0)) }
  }

  #[inline(always)]
  pub fn movemask(self) -> Mask {
    unsafe {
      let mask = vreinterpretq_u16_u8(self.0);
      let res = vshrn_n_u16(mask, 4); // the magic sauce
      let matches = vget_lane_u64(vreinterpret_u64_u8(res), 0);
      Mask(matches)
    }
  }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask(u64);

#[cfg(debug_assertions)]
impl core::fmt::Debug for Mask {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:064b}", self.0)
  }
}

impl Mask {
  #[inline(always)]
  pub fn has_match(&self) -> bool {
    // We have a match if the mask is not empty.
    self.0 != 0
  }

  /// Returns the character index of the first match.
  ///
  /// NEON packs 4 bits per character, so we divide by 4.
  #[inline(always)]
  pub fn first_match(&self) -> u32 {
    (self.0.trailing_zeros() >> 2) as u32
  }

  /// Clear the lowest set bit (BLSR).
  #[inline(always)]
  pub fn clear_to_first(&mut self) {
    self.0 &= self.0 - 1;
  }

  /// intersect this mask with `window`, returning a new mask
  #[inline(always)]
  pub fn window(&self, window: Self) -> Self {
    Self(self.0 & window.0)
  }

  /// get the bit window from the start of the chunk up to the first match
  ///
  /// handles the empty mask case by returning all-ones (the full chunk window).
  #[inline(always)]
  pub fn leading_window(&self) -> Self {
    let lsb = self.0 & self.0.wrapping_neg();
    Self(lsb.wrapping_shl(1).wrapping_sub(1))
  }

  /// create the bit window from a character index to the end of the mask
  ///
  /// `from` is a character index; NEON encodes 4 bits per character.
  #[inline(always)]
  pub fn trailing_window(from: u32) -> Self {
    let bit_pos = from << 2;
    Self(!((1_u64.wrapping_shl(bit_pos as u64)).wrapping_sub(1)))
  }

  /// create a bitmask covering bits from `from` (inclusive) to `to` (exclusive) in character indices.
  ///
  /// NEON encodes 4 bits per character, so character indices are multiplied by 4.
  #[inline(always)]
  pub fn between_window(from: u32, to: u32) -> Self {
    let from_bit = from << 2;
    let to_bit = to << 2;
    Self(
      ((1_u64.wrapping_shl(to_bit as u64)).wrapping_sub(1))
        & !((1_u64.wrapping_shl(from_bit as u64)).wrapping_sub(1)),
    )
  }
}