use std::hint::cold_path;

use super::*;
use crate::irc::wide::Vector as V;

pub(crate) fn parse(src: &str, pos: &mut usize) -> Option<RawTags> {
  let src = src[*pos..].strip_prefix('@')?.as_bytes();

  // 1. scan for ASCII space to find tags end
  let end = find_first(src, b' ')?;
  *pos += end + 2; // skip '@' + space

  let remainder = &src[..end];
  let mut tags = Array::<128, TagPair>::new();
  let mut offset = 0;

  let mut state = State::Key { key_start: 0 };
  while offset + V::SIZE < remainder.len() {
    let chunk = V::load_unaligned(remainder, offset);
    parse_chunk(offset, chunk, &mut state, &mut tags);
    offset += V::SIZE;
  }

  if remainder.len() - offset > 0 {
    let chunk = V::load_unaligned_remainder(remainder, offset);
    parse_chunk(offset, chunk, &mut state, &mut tags);

    if let State::Value { key_start, key_end } = state {
      // value contains whatever is left after key_end

      let pos = remainder.len(); // pos of `;`

      tags.push(TagPair {
        // relative to original `src`
        key_start: key_start as u32 + 1,
        key_end: (key_end - key_start) as u16,
        // starts after `=`
        value_end: (pos - (key_end + 1)) as u16,
      });
    }
  }

  Some(RawTags(tags.to_vec()))
}

#[derive(Clone, Copy)]
enum State {
  // these `key_start`s are relative to the tags string (stripped `@` symbol)
  // while the `key_start` in the `TagPair` is relative to the `src` string (with `@` symbol)
  Key { key_start: usize },
  // the `key_end` field in `State` means the absolute index of the end of the key (with offset)
  // while `key_end` in `TagPair` is an offset from `key_start` of `TagPair`
  // TODO: rename them on tag pair to key_end_offset or smth
  Value { key_start: usize, key_end: usize },
}

#[inline(always)]
fn parse_chunk(offset: usize, chunk: V, state: &mut State, tags: &mut Array<128, TagPair>) {
  // 1. Get raw scalar integers directly out of the SIMD vectors
  let eq_mask = chunk.eq(b'=').movemask();
  let mut semi_mask = chunk.eq(b';').movemask();

  // Track where the current token began relative to this specific chunk
  let mut current_start_idx = match *state {
    State::Value { key_start, key_end } => {
      if !semi_mask.has_match() {
        return;
      }
      let semi_idx = semi_mask.first_match();
      let pos = offset + semi_idx as usize;
      *state = State::Key { key_start: pos + 1 };
      tags.push(TagPair {
        // relative to original `src`
        key_start: key_start as u32 + 1,
        key_end: (key_end - key_start) as u16,
        // the names `*_end` are fucking bait when they're actually offsets from `key_start` rather than end indicies
        // starts after `=`
        value_end: (pos - (key_end + 1)) as u16,
      });

      semi_mask.clear_to_first();
      semi_idx + 1
    }
    State::Key { key_start } => {
      // run the first iteration of the loop to get to a clean state
      // without leftovers from previous chunks
      if !semi_mask.has_match() {
        // or skip to the next chunk
        return;
      }
      let semi_idx = semi_mask.first_match();

      // !((1 << 0) - 1) == 1 == 0xF...F
      let bit_window = (1 << semi_idx) - 1;

      let eq_in_window = eq_mask.window(bit_window);

      if eq_in_window != 0 {
        // HAPPY PATH: key=value
        let eq_idx = eq_in_window.trailing_zeros();

        tags.push(TagPair {
          // `State`'s `key_start`s are relative to the tags string (stripped `@` symbol)
          // while the `key_start` in the `TagPair` is relative to the `src` string (with `@` symbol)
          key_start: key_start as u32 + 1,
          // offset - key_start = the part of the key in the previous chunk
          // eq_idx = the part of the key in this chunk
          key_end: ((offset - key_start) as u32 + eq_idx) as u16,
          value_end: (semi_idx - (eq_idx + 1)) as u16,
        });
      } else {
        cold_path();
        // VALUELESS PATH: key; (No equal sign bit fell into our window)
        tags.push(TagPair {
          key_start: key_start as u32 + 1,
          // offset - key_start = the part of the key in the previous chunk
          // semi_idx = the part of the key in this chunk
          key_end: ((offset - key_start) as u32 + semi_idx) as u16,
          value_end: 0, // Explicitly valueless
        });
      }

      // Clear the lowest set bit in the semicolon mask (BLSR instruction or bitwise equivalent)
      semi_mask.clear_to_first();

      semi_idx + 1
    }
  };

  while semi_mask.has_match() {
    // BLSI (Extract Lowest Set Isolated Bit) or TZCNT (Trailing Zeros)
    // Find the exact bit position of the first semicolon
    let semi_idx = semi_mask.first_match();

    // Create a bitmask that isolates everything from our current position up to this semicolon
    // Example: if current_start_idx = 2 and semi_idx = 7, mask is 0001111100
    let bit_window = ((1 << semi_idx) - 1) & !((1 << current_start_idx) - 1);

    // Is there an equal sign bit inside this exact window?
    let eq_in_window = eq_mask.window(bit_window);

    // there may be multiple equal signs because values can have it
    // but we only care for the first one since it's the separator
    // TODO: this obviously doesn't account for cross chunk state
    if eq_in_window != 0 {
      // HAPPY PATH: key=value
      let eq_idx = eq_in_window.trailing_zeros();

      tags.push(TagPair {
        key_start: offset as u32 + current_start_idx + 1,
        key_end: (eq_idx - current_start_idx) as u16,
        value_end: (semi_idx - (eq_idx + 1)) as u16,
      });
    } else {
      cold_path();
      // VALUELESS PATH: key; (No equal sign bit fell into our window)
      tags.push(TagPair {
        key_start: offset as u32 + current_start_idx + 1,
        key_end: (semi_idx - current_start_idx) as u16,
        value_end: 0, // Explicitly valueless
      });
    }

    // Advance our structural cursor past this semicolon
    current_start_idx = semi_idx + 1;

    // Clear the lowest set bit in the semicolon mask (BLSR instruction or bitwise equivalent)
    semi_mask.clear_to_first();
    // there is no need to mutate the equal mask because we're only interacting with it through the bit window
  }

  let key_start = offset + current_start_idx as usize;
  // the window over leftovers after the last semicolon
  let bit_window = !((1_u32 << current_start_idx) - 1);
  let eq_in_window = eq_mask.window(bit_window);
  // the state only matters cross chunk so we mutate it once we exit
  // TODO: this is obviously wrong since this doesn't account for long keys/values that started in a previous chunk
  // and may even potentially not end in this chunk but the next one
  *state = if eq_in_window != 0 {
    // there is an equal sign in the window, meaning the chunk ends on a value
    State::Value {
      key_start,
      key_end: offset + eq_in_window.trailing_zeros() as usize,
    }
  } else {
    // there are no equal signs after the latest semicolon
    // meaning the chunk ends on a key
    State::Key { key_start }
  }
}

// I didn't want to use runtime feature detection, or bring in a dependency for this.
//
// This implementation is ported from BurntSushi/memchr to use our vector/mask types:
// https://github.com/BurntSushi/memchr/blob/7fccf70e2a58c1fbedc9b9687c2ba0cf5992537b/src/arch/generic/memchr.rs#L143-L144
//
// The original implementation is licensed under the MIT license.
#[allow(clippy::erasing_op, clippy::identity_op, clippy::needless_range_loop)]
#[inline]
fn find_first(data: &[u8], byte: u8) -> Option<usize> {
  // 1. scalar fallback for small data
  if data.len() < V::SIZE {
    for i in 0..data.len() {
      if data[i] == byte {
        return Some(i);
      }
    }

    return None;
  }

  // 2. read the first chunk unaligned, because we are now
  //    guaranteed to have more than vector-size bytes
  let chunk = V::load_unaligned(data, 0);
  let mask = chunk.eq(byte).movemask();
  if mask.has_match() {
    return Some(mask.first_match() as usize);
  }

  // 3. read the rest of the data in vector-size aligned chunks
  const UNROLLED_BYTES: usize = 4 * V::SIZE;

  // it's fine if we overlap the next vector-size chunk with
  // some part of the first chunk, because we already know
  // that there is no match in the first vector-size bytes.
  let data_addr = data.as_ptr().addr();
  let aligned_start_addr = data_addr + V::SIZE - (data_addr % V::SIZE);
  let aligned_start_offset = aligned_start_addr - data_addr;

  let mut offset = aligned_start_offset;
  while offset + UNROLLED_BYTES < data.len() {
    // do all loads up-front to saturate the pipeline
    let chunk_0 = V::load_aligned(data, offset + V::SIZE * 0).eq(byte);
    let chunk_1 = V::load_aligned(data, offset + V::SIZE * 1).eq(byte);
    let chunk_2 = V::load_aligned(data, offset + V::SIZE * 2).eq(byte);
    let chunk_3 = V::load_aligned(data, offset + V::SIZE * 3).eq(byte);

    // TODO: movemask_will_have_non_zero

    let mask = chunk_0.movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos + 0 * V::SIZE);
    }

    let mask = chunk_1.movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos + 1 * V::SIZE);
    }

    let mask = chunk_2.movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos + 2 * V::SIZE);
    }

    let mask = chunk_3.movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos + 3 * V::SIZE);
    }

    offset += V::SIZE * 4;
  }

  // 4. we may have fewer than UNROLLED_BYTES bytes left, which may
  //    still be enough for one or more vector-size chunks.
  while offset + V::SIZE <= data.len() {
    // the data is still guaranteed to be aligned at this point.
    let chunk = V::load_aligned(data, offset);
    let mask = chunk.eq(byte).movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos);
    }

    offset += V::SIZE;
  }

  // 5. we definitely have fewer than a single vector-size chunk left,
  //    so we have to read the last chunk unaligned.
  //    note that it is fine if it overlaps with the previous chunk,
  //    for the same reason why it's fine in step 3.
  if offset < data.len() {
    let offset = data.len() - V::SIZE;

    let chunk = V::load_unaligned(data, offset);
    let mask = chunk.eq(byte).movemask();
    if mask.has_match() {
      let pos = mask.first_match() as usize;
      return Some(offset + pos);
    }
  }

  None
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn find_first_test() {
    fn a(size: usize, needle_at: usize) -> Vec<u8> {
      let mut data = vec![b'.'; size];
      data[needle_at] = b'x';
      data
    }

    let cases: &[(&[u8], Option<usize>)] = &[
      // sub vector-size chunks
      (b"", None),      // 0 bytes
      (b"x", Some(0)),  // 1 byte
      (b".", None),     // 1 byte
      (b"xx", Some(0)), // 2 bytes
      (b"x.", Some(0)), // 2 bytes
      (b".x", Some(1)), // 2 bytes
      // vector-size chunks
      // 16 bytes
      (b"x...............", Some(0)),
      (b".x..............", Some(1)),
      (b"..............x.", Some(14)),
      (b"...............x", Some(15)),
      // uneven + above vector-size chunks
      // 17 bytes
      (b"x................", Some(0)),
      (b".x...............", Some(1)),
      (b"...............x.", Some(15)),
      (b"................x", Some(16)),
      // 31 bytes
      (b"x...............................", Some(0)),
      (b".x..............................", Some(1)),
      (b"..............................x.", Some(30)),
      (b"...............................x", Some(31)),
      // large chunks
      // 1 KiB
      (&a(1024, 0)[..], Some(0)),
      (&a(1024, 1)[..], Some(1)),
      (&a(1024, 1022)[..], Some(1022)),
      (&a(1024, 1023)[..], Some(1023)),
    ];

    for (i, case) in cases.iter().enumerate() {
      let (data, expected) = *case;
      assert_eq!(find_first(data, b'x'), expected, "case {} failed", i);
    }
  }
}
