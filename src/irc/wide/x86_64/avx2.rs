use core::arch::x86_64::{
  __m256i, _mm256_cmpeq_epi8, _mm256_load_si256, _mm256_loadu_si256, _mm256_movemask_epi8,
};

#[repr(align(32))]
struct Align32([u8; 32]);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector(__m256i);

impl Vector {
  /// Size in bytes.
  pub const SIZE: usize = 32;

  #[inline]
  pub const fn fill(v: u8) -> Self {
    Self(unsafe { core::mem::transmute::<[u8; 32], __m256i>([v; 32]) })
  }

  /// Load 32 bytes from the given slice into a vector.
  ///
  /// `data[offset..].len()` must be greater than 32 bytes.
  #[inline(always)]
  pub fn load_unaligned(data: &[u8], offset: usize) -> Self {
    unsafe {
      debug_assert!(data[offset..].len() >= 32);
      Self(_mm256_loadu_si256(
        data.as_ptr().add(offset) as *const __m256i
      ))
    }
  }

  /// Load 32 bytes from the given slice into a vector.
  ///
  /// `data[offset..].len()` must be greater than 32 bytes.
  /// The data must be 32-byte aligned.
  #[inline(always)]
  pub fn load_aligned(data: &[u8], offset: usize) -> Self {
    unsafe {
      debug_assert!(data[offset..].len() >= 32);
      debug_assert!(data.as_ptr().add(offset).addr().is_multiple_of(32));
      Self(_mm256_load_si256(
        data.as_ptr().add(offset) as *const __m256i
      ))
    }
  }

  /// Load at most 32 bytes from the given slice into a vector
  /// by loading it into an intermediate buffer on the stack.
  #[inline(always)]
  pub fn load_unaligned_remainder(data: &[u8], offset: usize) -> Self {
    unsafe {
      let mut buf = Align32([0; 32]);
      buf.0[..data.len() - offset].copy_from_slice(&data[offset..]);

      Self(_mm256_load_si256(buf.0.as_ptr() as *const __m256i))
    }
  }

  #[inline(always)]
  pub fn eq(self, byte: u8) -> Self {
    unsafe { Self(_mm256_cmpeq_epi8(self.0, Self::fill(byte).0)) }
  }

  #[inline(always)]
  pub fn movemask(self) -> Mask {
    unsafe {
      let value = _mm256_movemask_epi8(self.0).cast_unsigned();
      Mask(value)
    }
  }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
// pub(super) just for the tests
pub struct Mask(pub(super) u32);

#[cfg(debug_assertions)]
impl core::fmt::Debug for Mask {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:032b}", self.0)
  }
}

impl Mask {
  #[inline(always)]
  pub fn has_match(&self) -> bool {
    self.0 != 0
  }

  /// return the position of the first match in the chunk
  #[inline(always)]
  pub fn first_match(&self) -> u32 {
    self.0.trailing_zeros()
  }

  /// clear the first match
  ///
  /// ```text
  /// 10101010 - input
  /// 10101001 - (input - 1)
  /// 10101000 - output
  /// ```
  #[inline(always)]
  pub fn clear_to_first(&mut self) {
    self.0 &= self.0 - 1;
  }

  /// intersect this mask with `window`, returning a new mask
  ///
  /// ```text
  /// 01010101 - mask
  /// 00011110 - window
  /// 00010100 - output
  /// ```
  #[inline(always)]
  pub fn window(&self, window: Self) -> Self {
    Self(self.0 & window.0)
  }

  /// get the bit window from the start of the chunk up to the first match
  ///
  /// ```text
  /// d;=c;b=a
  /// 01001000 - input - the first match is on character index 3
  /// 00001111 - output - window covers up to the first semicolon
  /// ```
  ///
  /// handles the empty mask case by returning all-ones (the full chunk window)
  ///
  /// ```text
  /// yek-gnol
  /// 00000000 - input
  /// 11111111 - output - window covers everything
  /// ```
  #[inline(always)]
  pub fn leading_window(&self) -> Self {
    let lsb = self.0 & self.0.wrapping_neg();
    Self(lsb.wrapping_shl(1).wrapping_sub(1))
  }

  /// create the bit window from a position in a mask to the end of the mask
  ///
  /// ```text
  ///  5 ~~~~~ - position
  /// 11100000 - output
  /// ```
  #[inline(always)]
  pub fn trailing_window(from: u32) -> Self {
    // using a larger integer in the case that `from` is 32
    // then the 1_u64 << from = 0x1_0000_0000 -> truncating to 0
    Self(!(1_u64.wrapping_shl(from).wrapping_sub(1)) as u32)
  }

  /// create a bitmask covering bits from `from` (inclusive) to `to` (exclusive)
  ///
  /// ```text
  /// 01010101 - from 1 to 5
  ///   ^   ^
  /// 00011110 - output
  /// ```
  #[inline(always)]
  pub fn between_window(from: u32, to: u32) -> Self {
    // using a larger integer in the case that `to` is 32
    // then `since` = 0x1_0000_0000 -> truncating to 0
    let until = 1_u64.wrapping_shl(to).wrapping_sub(1);
    let since = !(1_u64.wrapping_shl(from).wrapping_sub(1));
    Self((until & since) as u32)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_trailing_window() {
    let max_zeros = 0_u32.trailing_zeros(); // = 32
    let mask = Mask::trailing_window(max_zeros);
    // should be zeroed since there are no bits after 32nd position in a 32 bit vector
    assert_eq!(mask.0, 0);
  }

  #[test]
  fn test_between_window() {
    // full window
    let mask = Mask::between_window(0, 32);
    assert_eq!(mask.0, !0);

    // zero length window
    let mask = Mask::between_window(1, 1);
    assert_eq!(mask.0, 0);
  }
}
