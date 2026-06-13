//! Packed slot allowlists for repeated filtered search.
//!
//! `TurboQuantIndex::search_with_mask` accepts a caller-friendly `&[bool]`,
//! but graph and metadata views tend to reuse the same candidate set across
//! many queries. `SlotMask` stores that set as the same packed `u64` words the
//! search kernel consumes, so callers can build a view once and avoid
//! re-packing a full-length boolean array on every search.

use crate::BLOCK;
use std::iter::FusedIterator;

/// Packed allowlist over positional vector slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMask {
    len: usize,
    words: Vec<u64>,
    count: usize,
    block_counts: Vec<u16>,
    active_blocks: Vec<usize>,
}

impl SlotMask {
    /// Create an empty mask over `len` slots.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            words: vec![0; n_words(len)],
            count: 0,
            block_counts: vec![0; n_blocks(len)],
            active_blocks: Vec::new(),
        }
    }

    /// Create a mask where every slot is allowed.
    pub fn all(len: usize) -> Self {
        let mut words = vec![!0u64; n_words(len)];
        clear_tail_bits(len, &mut words);
        let block_counts = full_block_counts(len);
        let active_blocks = (0..block_counts.len()).collect();
        Self {
            len,
            words,
            count: len,
            block_counts,
            active_blocks,
        }
    }

    /// Build a mask from allowed slot indices. Duplicate slots are ignored.
    ///
    /// # Panics
    ///
    /// Panics if any slot is out of range for this mask length.
    pub fn from_slots<I>(len: usize, slots: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let slots = slots.into_iter();
        let block_count = n_blocks(len);
        let incremental_cutoff = (block_count / 2).max(64);
        if slots
            .size_hint()
            .1
            .is_some_and(|upper| upper <= incremental_cutoff)
        {
            let mut mask = Self::new(len);
            for slot in slots {
                mask.allow(slot);
            }
            return mask;
        }

        let mut mask = Self {
            len,
            words: vec![0; n_words(len)],
            count: 0,
            block_counts: Vec::new(),
            active_blocks: Vec::new(),
        };
        for slot in slots {
            assert!(
                slot < len,
                "slot {slot} out of range for SlotMask length {}",
                len
            );
            let word = slot >> 6;
            let bit = 1u64 << (slot & 63);
            mask.words[word] |= bit;
        }
        mask.rebuild_counts_and_blocks();
        mask
    }

    /// Number of addressable slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of allowed slots.
    pub fn count(&self) -> usize {
        self.count
    }

    /// True when no slots are allowed.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// True when every addressable slot is allowed.
    pub fn is_all(&self) -> bool {
        self.count == self.len
    }

    /// Number of 32-slot SIMD blocks that contain at least one allowed slot.
    pub fn active_block_count(&self) -> usize {
        self.active_blocks.len()
    }

    /// Packed mask words consumed by the search kernel.
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Iterate allowed slot indices in ascending order.
    ///
    /// This walks the packed mask words directly, so sparse masks can expose
    /// their selected slots without scanning every addressable slot.
    pub fn allowed_slots(&self) -> AllowedSlots<'_> {
        self.allowed_slots_with_count(self.count)
    }

    fn allowed_slots_with_count(&self, remaining: usize) -> AllowedSlots<'_> {
        AllowedSlots {
            words: &self.words,
            len: self.len,
            word_idx: 0,
            base_slot: 0,
            current_word: 0,
            remaining,
        }
    }

    /// Allow a slot. Returns `true` if the slot was newly allowed.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= self.len()`.
    pub fn allow(&mut self, slot: usize) -> bool {
        assert!(
            slot < self.len,
            "slot {slot} out of range for SlotMask length {}",
            self.len
        );
        let word = slot >> 6;
        let bit = 1u64 << (slot & 63);
        if self.words[word] & bit == 0 {
            self.words[word] |= bit;
            self.count += 1;
            let block = slot / BLOCK;
            if self.block_counts[block] == 0 {
                insert_sorted(&mut self.active_blocks, block);
            }
            self.block_counts[block] += 1;
            true
        } else {
            false
        }
    }

    /// Disallow a slot. Returns `true` if the slot was previously allowed.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= self.len()`.
    pub fn disallow(&mut self, slot: usize) -> bool {
        assert!(
            slot < self.len,
            "slot {slot} out of range for SlotMask length {}",
            self.len
        );
        let word = slot >> 6;
        let bit = 1u64 << (slot & 63);
        if self.words[word] & bit != 0 {
            self.words[word] &= !bit;
            self.count -= 1;
            let block = slot / BLOCK;
            self.block_counts[block] -= 1;
            if self.block_counts[block] == 0 {
                remove_sorted(&mut self.active_blocks, block);
            }
            true
        } else {
            false
        }
    }

    /// True if `slot` is currently allowed.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= self.len()`.
    pub fn contains(&self, slot: usize) -> bool {
        assert!(
            slot < self.len,
            "slot {slot} out of range for SlotMask length {}",
            self.len
        );
        let word = slot >> 6;
        let bit = 1u64 << (slot & 63);
        self.words[word] & bit != 0
    }

    /// Remove every allowed slot.
    pub fn clear(&mut self) {
        self.words.fill(0);
        self.count = 0;
        self.block_counts.fill(0);
        self.active_blocks.clear();
    }

    /// In-place union with another mask over the same slot domain.
    ///
    /// # Panics
    ///
    /// Panics if the mask lengths differ.
    pub fn union_with(&mut self, other: &Self) {
        assert_eq!(
            self.len, other.len,
            "SlotMask length mismatch: {} != {}",
            self.len, other.len
        );
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
        self.rebuild_counts_and_blocks();
    }

    /// In-place union with many masks over the same slot domain.
    ///
    /// Rebuilds counts and active blocks once after all packed words have been
    /// merged, which is cheaper than repeated [`Self::union_with`] calls for
    /// multi-source metadata filters.
    pub fn union_with_many<'a, I>(&mut self, others: I)
    where
        I: IntoIterator<Item = &'a Self>,
    {
        for other in others {
            assert_eq!(
                self.len, other.len,
                "SlotMask length mismatch: {} != {}",
                self.len, other.len
            );
            for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
                *a |= *b;
            }
        }
        self.rebuild_counts_and_blocks();
    }

    /// In-place intersection with another mask over the same slot domain.
    ///
    /// # Panics
    ///
    /// Panics if the mask lengths differ.
    pub fn intersect_with(&mut self, other: &Self) {
        assert_eq!(
            self.len, other.len,
            "SlotMask length mismatch: {} != {}",
            self.len, other.len
        );
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= *b;
        }
        self.rebuild_counts_and_blocks();
    }

    /// In-place intersection with many masks over the same slot domain.
    ///
    /// Rebuilds counts and active blocks once after all packed words have been
    /// intersected. This keeps repeated graph+tag+source+time view composition
    /// on the packed representation instead of walking selected slots after
    /// every predicate.
    pub fn intersect_with_many<'a, I>(&mut self, others: I)
    where
        I: IntoIterator<Item = &'a Self>,
    {
        for other in others {
            assert_eq!(
                self.len, other.len,
                "SlotMask length mismatch: {} != {}",
                self.len, other.len
            );
            for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
                *a &= *b;
            }
        }
        self.rebuild_counts_and_blocks();
    }

    pub(crate) fn search_parts(&self) -> (&[u64], usize, &[usize]) {
        (&self.words, self.count, &self.active_blocks)
    }

    fn rebuild_counts_and_blocks(&mut self) {
        let mut block_counts = vec![0u16; n_blocks(self.len)];
        let mut active_blocks = Vec::new();
        let mut count = 0usize;
        for (block, block_count) in block_counts.iter_mut().enumerate() {
            let start = block * BLOCK;
            let end = (start + BLOCK).min(self.len);
            let selected = count_bits_in_range(&self.words, start, end);
            if selected > 0 {
                active_blocks.push(block);
                *block_count = selected as u16;
                count += selected;
            }
        }
        self.count = count;
        self.block_counts = block_counts;
        self.active_blocks = active_blocks;
    }
}

/// Iterator over allowed slot indices in a [`SlotMask`].
#[derive(Clone, Debug)]
pub struct AllowedSlots<'a> {
    words: &'a [u64],
    len: usize,
    word_idx: usize,
    base_slot: usize,
    current_word: u64,
    remaining: usize,
}

impl Iterator for AllowedSlots<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            if self.current_word == 0 {
                if self.word_idx >= self.words.len() {
                    self.remaining = 0;
                    return None;
                }
                self.base_slot = self.word_idx * 64;
                self.current_word = self.words[self.word_idx];
                self.word_idx += 1;
                continue;
            }

            let bit = self.current_word.trailing_zeros() as usize;
            self.current_word &= self.current_word - 1;
            let slot = self.base_slot + bit;
            if slot < self.len {
                self.remaining -= 1;
                return Some(slot);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for AllowedSlots<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl FusedIterator for AllowedSlots<'_> {}

fn n_words(len: usize) -> usize {
    (len + 63) / 64
}

fn n_blocks(len: usize) -> usize {
    (len + BLOCK - 1) / BLOCK
}

fn full_block_counts(len: usize) -> Vec<u16> {
    let mut counts = vec![0u16; n_blocks(len)];
    for (block, count) in counts.iter_mut().enumerate() {
        let start = block * BLOCK;
        let end = (start + BLOCK).min(len);
        *count = (end - start) as u16;
    }
    counts
}

fn clear_tail_bits(len: usize, words: &mut [u64]) {
    let rem = len & 63;
    if rem != 0 {
        if let Some(last) = words.last_mut() {
            *last &= (1u64 << rem) - 1;
        }
    }
}

fn count_bits_in_range(words: &[u64], start: usize, end: usize) -> usize {
    if start >= end {
        return 0;
    }
    let start_word = start >> 6;
    let end_word = (end - 1) >> 6;
    let mut count = 0usize;
    for (offset, word) in words[start_word..=end_word].iter().enumerate() {
        let word_idx = start_word + offset;
        let lo = if word_idx == start_word {
            start & 63
        } else {
            0
        };
        let hi = if word_idx == end_word {
            ((end - 1) & 63) + 1
        } else {
            64
        };
        let low_mask = if lo == 0 { 0 } else { (1u64 << lo) - 1 };
        let high_mask = if hi == 64 { !0u64 } else { (1u64 << hi) - 1 };
        count += (*word & (high_mask & !low_mask)).count_ones() as usize;
    }
    count
}

fn insert_sorted(values: &mut Vec<usize>, value: usize) {
    match values.binary_search(&value) {
        Ok(_) => {}
        Err(pos) => values.insert(pos, value),
    }
}

fn remove_sorted(values: &mut Vec<usize>, value: usize) {
    if let Ok(pos) = values.binary_search(&value) {
        values.remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::SlotMask;

    #[test]
    fn from_slots_deduplicates_and_counts() {
        let mask = SlotMask::from_slots(130, [0, 64, 64, 129]);
        assert_eq!(mask.len(), 130);
        assert_eq!(mask.count(), 3);
        assert!(mask.contains(0));
        assert!(mask.contains(64));
        assert!(mask.contains(129));
    }

    #[test]
    fn all_clears_unused_tail_bits() {
        let mask = SlotMask::all(65);
        assert_eq!(mask.count(), 65);
        assert!(mask.is_all());
        assert_eq!(mask.as_words().len(), 2);
        assert_eq!(mask.as_words()[1], 1);
        assert_eq!(mask.active_block_count(), 3);
    }

    #[test]
    fn union_and_intersection_update_counts() {
        let mut a = SlotMask::from_slots(100, [1, 2, 3, 90]);
        let b = SlotMask::from_slots(100, [3, 4, 90, 91]);

        a.union_with(&b);
        assert_eq!(a.count(), 6);
        assert!(a.contains(4));
        assert!(a.contains(91));

        let c = SlotMask::from_slots(100, [2, 4, 91]);
        a.intersect_with(&c);
        assert_eq!(a.count(), 3);
        assert!(a.contains(2));
        assert!(a.contains(4));
        assert!(a.contains(91));
        assert_eq!(a.active_block_count(), 2);
    }

    #[test]
    fn batched_union_and_intersection_update_counts() {
        let mut union = SlotMask::from_slots(130, [0, 64]);
        let b = SlotMask::from_slots(130, [31, 32, 65]);
        let c = SlotMask::from_slots(130, [63, 96, 129]);
        union.union_with_many([&b, &c]);
        assert_eq!(union.count(), 8);
        assert_eq!(union.active_block_count(), 5);
        assert_eq!(
            union.allowed_slots().collect::<Vec<_>>(),
            vec![0, 31, 32, 63, 64, 65, 96, 129]
        );

        let mut intersection = SlotMask::from_slots(130, [0, 31, 32, 63, 64, 65, 96, 129]);
        let keep_a = SlotMask::from_slots(130, [31, 32, 63, 64, 65, 96]);
        let keep_b = SlotMask::from_slots(130, [32, 63, 65, 96, 128]);
        intersection.intersect_with_many([&keep_a, &keep_b]);
        assert_eq!(intersection.count(), 4);
        assert_eq!(intersection.active_block_count(), 3);
        assert_eq!(
            intersection.allowed_slots().collect::<Vec<_>>(),
            vec![32, 63, 65, 96]
        );
    }

    #[test]
    fn active_blocks_track_allow_and_disallow() {
        let mut mask = SlotMask::new(96);
        assert_eq!(mask.active_block_count(), 0);

        mask.allow(40);
        mask.allow(41);
        mask.allow(3);
        assert_eq!(mask.active_block_count(), 2);
        assert_eq!(mask.search_parts().2, &[0, 1]);

        assert!(mask.disallow(40));
        assert_eq!(mask.active_block_count(), 2);
        assert!(mask.disallow(41));
        assert_eq!(mask.active_block_count(), 1);
        assert_eq!(mask.search_parts().2, &[0]);
    }

    #[test]
    fn allowed_slots_iterates_selected_slots_in_order() {
        let empty = SlotMask::new(10);
        assert_eq!(
            empty.allowed_slots().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );

        let all = SlotMask::all(70);
        assert_eq!(
            all.allowed_slots().collect::<Vec<_>>(),
            (0..70).collect::<Vec<_>>()
        );

        let mut sparse = SlotMask::from_slots(100, [99, 3, 64, 65, 3]);
        assert_eq!(
            sparse.allowed_slots().collect::<Vec<_>>(),
            vec![3, 64, 65, 99]
        );
        assert_eq!(sparse.allowed_slots().len(), 4);

        assert!(sparse.disallow(64));
        assert!(sparse.allow(32));
        assert_eq!(
            sparse.allowed_slots().collect::<Vec<_>>(),
            vec![3, 32, 65, 99]
        );
    }
}
