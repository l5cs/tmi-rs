use core::arch::x86_64::{
  __m128i, _mm_cmpeq_epi8, _mm_load_si128, _mm_loadu_si128, _mm_movemask_epi8,
};

#[repr(align(16))]
struct Align16([u8; 16]);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector(__m128i);

impl Vector {
  /// Size in bytes.
  pub const SIZE: usize = 16;

  #[inline]
  pub const fn fill(v: u8) -> Self {
    Self(unsafe { core::mem::transmute::<[u8; 16], __m128i>([v; 16]) })
  }

  /// Load 16 bytes from the given slice into a vector.
  ///
  /// `data[offset..].len()` must be greater than 16 bytes.
  #[inline(always)]
  pub fn load_unaligned(data: &[u8], offset: usize) -> Self {
    unsafe {
      debug_assert!(data[offset..].len() >= 16);
      Self(_mm_loadu_si128(data.as_ptr().add(offset) as *const __m128i))
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
      Self(_mm_load_si128(data.as_ptr().add(offset) as *const __m128i))
    }
  }

  /// Load at most 16 bytes from the given slice into a vector
  /// by loading it into an intermediate buffer on the stack.
  #[inline(always)]
  pub fn load_unaligned_remainder(data: &[u8], offset: usize) -> Self {
    unsafe {
      let mut buf = Align16([0; 16]);
      buf.0[..data.len() - offset].copy_from_slice(&data[offset..]);

      Self(_mm_load_si128(buf.0.as_ptr() as *const __m128i))
    }
  }

  /// Compare 16 8-bit elements in `self` against `other`, leaving a `1` in each
  #[inline(always)]
  pub fn eq(self, byte: u8) -> Self {
    unsafe { Self(_mm_cmpeq_epi8(self.0, Self::fill(byte).0)) }
  }

  #[inline(always)]
  pub fn movemask(self) -> Mask {
    unsafe {
      let value = _mm_movemask_epi8(self.0).cast_unsigned();
      Mask(value)
    }
  }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
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
    Self(!((1_u32.wrapping_shl(from)).wrapping_sub(1)))
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
    Self(((1_u32.wrapping_shl(to)).wrapping_sub(1)) & !((1_u32.wrapping_shl(from)).wrapping_sub(1)))
  }
}
