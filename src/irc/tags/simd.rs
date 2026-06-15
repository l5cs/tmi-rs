use std::hint::cold_path;

use super::*;
use crate::irc::wide::{Mask, Vector as V};

pub(crate) fn parse(src: &str, pos: &mut usize) -> Option<RawTags> {
  const LEADING_AT_LEN: usize = 1;

  let src = src[*pos..].strip_prefix('@')?.as_bytes();

  // 1. scan for ASCII space to find tags end
  let end = find_first(src, b' ')?;
  *pos += end + LEADING_AT_LEN + 1; // skip '@' + space

  let remainder = &src[..end];
  let mut tags = Array::<128, TagPair>::new();
  let mut offset = 0;

  let mut state = State::Key { key_start: LEADING_AT_LEN };
  while offset + V::SIZE < remainder.len() {
    let chunk = V::load_unaligned(remainder, offset);
    // including the @ symbol in offset
    let src_offset = offset + LEADING_AT_LEN;
    parse_chunk(src_offset, chunk, &mut state, &mut tags);
    offset += V::SIZE;
  }

  if remainder.len() - offset > 0 {
    let chunk = V::load_unaligned_remainder(remainder, offset);
    let src_offset = offset + LEADING_AT_LEN;
    parse_chunk(src_offset, chunk, &mut state, &mut tags);

    if let State::Value { key_start, key_length } = state {
      // value contains whatever is left after key_end
      let pos = remainder.len(); // pos of `;`
      // panic!("{pos} {key_start} {key_length}");
      tags.push(TagPair {
        // relative to original `src`
        key_start: key_start as u32,
        key_length: key_length as u16,
        // starts after `=`
        value_length: (pos - key_start - key_length) as u16,
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
  Value { key_start: usize, key_length: usize },
}

#[inline(always)]
fn parse_chunk(offset: usize, chunk: V, state: &mut State, tags: &mut Array<128, TagPair>) {
  let eq_mask = chunk.eq(b'=').movemask();
  let mut semi_mask = chunk.eq(b';').movemask();

  // finish the state from previous chunks so that we can start from a new key
  let mut chunk_cursor = match *state {
    State::Value { key_start, key_length } => {
      if !semi_mask.has_match() {
        // skip to the next chunk if previous chunk's value doesn't end in this chunk
        return;
      }
      let semi_idx = semi_mask.first_match();
      let pos = offset + semi_idx as usize;
      *state = State::Key { key_start: pos };
      tags.push(TagPair {
        // relative to original `src`
        key_start: key_start as u32,
        key_length: key_length as u16,
        // starts after `=`
        value_length: (pos - (key_start + key_length + 1)) as u16,
      });

      semi_mask.clear_to_first();
      semi_idx + 1
    }
    State::Key { key_start } => {
      if !semi_mask.has_match() && !eq_mask.has_match() {
        // skip to the next chunk if there are no separators at all
        return;
      } else if !semi_mask.has_match() {
        // this chunk has an euqal sign but no tag end
        // meaning we change the state and move next
        let eq_idx = eq_mask.first_match();
        let key_length = ((offset - key_start) as u32 + eq_idx) as usize;
        *state = State::Value { key_start, key_length };
        return;
      }

      // use leading_window to cover from the start of the chunk up to the first semicolon
      // (or the entire chunk if there are no semicolons)
      let eq_in_window = eq_mask.window(semi_mask.leading_window());
      let semi_idx = semi_mask.first_match();

      if eq_in_window.has_match() {
        // HAPPY PATH: key=value
        let eq_idx = eq_in_window.first_match();

        tags.push(TagPair {
          // `State`'s `key_start`s are relative to the tags string (stripped `@` symbol)
          // while the `key_start` in the `TagPair` is relative to the `src` string (with `@` symbol)
          key_start: key_start as u32,
          // offset - key_start = the part of the key in the previous chunk
          // eq_idx = the part of the key in this chunk
          key_length: ((offset - key_start) as u32 + eq_idx) as u16,
          value_length: (semi_idx - (eq_idx + 1)) as u16,
        });
      } else {
        cold_path();
        // VALUELESS PATH: key; (No equal sign)
        tags.push(TagPair {
          key_start: key_start as u32,
          // offset - key_start = the part of the key in the previous chunk
          // semi_idx = the part of the key in this chunk
          key_length: ((offset - key_start) as u32 + semi_idx) as u16,
          value_length: 0, // Explicitly valueless
        });
      }

      // Clear the lowest set bit in the semicolon mask (BLSR instruction or bitwise equivalent)
      if semi_mask.has_match() {
        // and account for case when there are no semicolons in the chunk at all
        semi_mask.clear_to_first();
        semi_idx + 1
      } else {
        0
      }
    }
  };
  // dbg!(semi_mask);

  while semi_mask.has_match() {
    // Find the exact bit position of the first semicolon
    let semi_idx = semi_mask.first_match();

    // Create a bitmask that isolates everything from our current position up to this semicolon
    // Example: if chunk_cursor = 2 and semi_idx = 7, mask is 0001111100
    let bit_window = Mask::between_window(chunk_cursor, semi_idx);

    // Is there an equal sign bit inside this exact window?
    let eq_in_window = eq_mask.window(bit_window);

    // there may be multiple equal signs because values can have it
    // but we only care for the first one since it's the separator
    if eq_in_window.has_match() {
      // HAPPY PATH: key=value
      let eq_idx = eq_in_window.first_match();

      tags.push(TagPair {
        key_start: offset as u32 + chunk_cursor,
        key_length: (eq_idx - chunk_cursor) as u16,
        value_length: (semi_idx - (eq_idx + 1)) as u16,
      });
    } else {
      cold_path();
      // VALUELESS PATH: key; (No equal sign)
      tags.push(TagPair {
        key_start: offset as u32 + chunk_cursor,
        key_length: (semi_idx - chunk_cursor) as u16,
        value_length: 0, // Explicitly valueless
      });
    }

    // Advance our structural cursor past this semicolon
    chunk_cursor = semi_idx + 1;

    // Clear the lowest set bit in the semicolon mask (BLSR instruction or bitwise equivalent)
    semi_mask.clear_to_first();
    // there is no need to mutate the equal mask because we're only interacting with it through the bit window
  }

  let key_start = offset + chunk_cursor as usize;
  // the window over leftovers after the last semicolon
  let bit_window = Mask::trailing_window(chunk_cursor);
  let eq_in_window = eq_mask.window(bit_window);

  // the state only matters cross chunk so we mutate it once we exit
  *state = if eq_in_window.has_match() {
    // there is an equal sign in the window, meaning the chunk ends on a value
    // panic!("{offset} {chunk_cursor}\n{eq_mask:?}\n{bit_window:?}\n{eq_in_window:?}");
    State::Value {
      key_start,
      key_length: (eq_in_window.first_match() - chunk_cursor) as usize,
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
  fn test_tag_parsing() {
    let src = "@a=b;c=d;user-type= ";
    let tags = parse(src, &mut 0).unwrap();
    let a = tags[0];
    assert_eq!(&src[a.key()], "a");
    assert_eq!(&src[a.value()], "b");
    let c = tags[1];
    assert_eq!(&src[c.key()], "c");
    assert_eq!(&src[c.value()], "d");
    let user_type = tags[2];
    assert_eq!(&src[user_type.key()], "user-type");
    assert_eq!(&src[user_type.value()], "");
  }

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