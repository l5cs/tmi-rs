cfg_if::cfg_if! {
  if #[cfg(all(
    target_arch = "x86_64",
    any(
      target_feature = "sse2",
      target_feature = "avx2",
      all(target_feature = "avx512f", target_feature = "avx512bw")
    )
  ))] {
    pub(super) mod x86_64;
    pub(super) use x86_64::Vector;
    use x86_64::Mask;
  } else if #[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon"
  ))] {
    pub(super) mod aarch64;
    pub(super) use aarch64::Vector;
    use aarch64::Mask;
  } else {
    compile_error!("unsupported target architecture - please disable the `simd` feature");
  }
}

impl Mask {
  /// get the the bit window from the start of the chunk to the first match
  /// 
  /// ```text
  /// V::SIZE = 8
  /// d;=c;b=a
  /// 01001000 - input - the first match is on the 4th character
  /// 00001111 - output - window covers up to the first semicolon
  /// ```
  /// 
  /// if there are no semicolons in the chunk (meaning there are equal signs)
  /// then trailing_zeros returns 32
  /// but we want 0 to get the full chunk bit window
  /// 
  /// ```text
  /// V::SIZE = 8
  /// yek-gnol
  /// 00000000 - input - no matches
  /// 11111111 - output - window covers the entire mask
  /// ```
  pub fn leading_window(&self) -> u32 {
    // 1. Isolate the lowest set bit (LSB). 
    //    e.g., 0b110100 -> 0b000100
    let lsb = self.0 & self.0.wrapping_neg();

    // 2. Shift it left by 1 and subtract 1 with wrapping arithmetic.
    //    e.g., (0b000100 << 1) - 1 = 0b001000 - 1 = 0b000111
    lsb.wrapping_shl(1).wrapping_sub(1)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_leading_window() {
    let mask = Mask(0b00000010);
    let leading_bit_window = mask.leading_window();
    assert_eq!(leading_bit_window, 0b00000011);
    let mask = Mask(0b11111111_00100000);
    let leading_bit_window = mask.leading_window();
    assert_eq!(leading_bit_window, 0b00000000_00111111);
    let mask = Mask(0b0);
    let leading_bit_window = mask.leading_window();
    assert_eq!(leading_bit_window, u32::MAX);
  }

  fn mask_trailing_window() {
    let cursor = 0;
    let mask = 0b00000000;
  }

  #[test]
  fn test_trailing_window() {

  }
}
