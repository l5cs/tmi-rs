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
    pub(super) use x86_64::{Mask, Vector};
  } else if #[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon"
  ))] {
    pub(super) mod aarch64;
    pub(super) use aarch64::{Mask, Vector};
  } else {
    compile_error!("unsupported target architecture - please disable the `simd` feature");
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_clear_to_first() {
    // 0b00101100 -> after clearing first (bit 2): 0b00101000
    let mut mask = Mask(0b00101100);
    mask.clear_to_first();
    assert_eq!(mask.0, 0b00101000);
  }

  #[test]
  fn test_window() {
    let mask = Mask(0b00100100);
    let window = Mask(0b00011110);
    assert_eq!(mask.window(window).0, 0b00000100);
  }

  #[test]
  fn test_leading_window() {
    // simple: first match at bit 1 -> covers bits 0-1
    let mask = Mask(0b00000010);
    assert_eq!(mask.leading_window().0, 0b00000011);

    // complex: matches after the first, window only up to the first
    // bits set at positions 5 and 2 -> first at 2 -> window covers 0-2
    let mask = Mask(0b00100100);
    assert_eq!(mask.leading_window().0, 0b00000111);

    // empty mask -> full window (all bits set)
    let mask = Mask(0);
    assert_eq!(mask.leading_window().0, !0);
  }

  #[test]
  fn test_trailing_window() {
    // from cursor = 2: window covers bits 2 to MSB
    let window = Mask::trailing_window(2);
    assert_eq!(window.0, !0b00000011);
  }

  #[test]
  fn test_between_window() {
    // from = 2, to = 7: window covers bits 2..=6 (since `to` is exclusive)
    let window = Mask::between_window(2, 7);
    assert_eq!(window.0, 0b01111100);
  }
}
