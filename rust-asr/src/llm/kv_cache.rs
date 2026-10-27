//! Key/value cache for autoregressive decoding.
//!
//! Like mlx-lm's `KVCache`, keys and values live in preallocated buffers that
//! grow in `STEP`-row increments, and each step writes its rows in place —
//! instead of concatenating, which copies the whole cache on every token.
//! A sliding-window cache drops the rows no future query can see whenever it
//! has to grow, so it stays bounded by the window.

use anyhow::Result;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::ops::zeros_dtype;
use mlx_rs::Array;

/// Rows added per buffer growth (mlx-lm's step).
const STEP: i32 = 256;

pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Valid rows at the front of the buffers.
    len: i32,
    /// Sliding attention window, if any: a query sees the last `window`
    /// positions, itself included.
    window: Option<i32>,
}

impl KvCache {
    pub fn new(window: Option<i32>) -> Self {
        Self {
            keys: None,
            values: None,
            len: 0,
            window,
        }
    }

    /// Appends `k`, `v` ([B, H, T, D]) and returns the cached keys and values
    /// the new queries may attend to, in temporal order and ending with the
    /// new rows. For a single query under a sliding window that is exactly
    /// the last `window` rows; otherwise every kept row (callers mask).
    pub fn update(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let t = k.shape()[2];
        let capacity = self.keys.as_ref().map_or(0, |a| a.shape()[2]);
        if self.len + t > capacity {
            // Rows older than the window are never attended to again.
            let keep = match self.window {
                Some(w) => self.len.min(w - 1),
                None => self.len,
            };
            let capacity = (keep + t + STEP - 1) / STEP * STEP + STEP;
            self.keys = Some(self.regrow(self.keys.as_ref(), k, keep, capacity)?);
            self.values = Some(self.regrow(self.values.as_ref(), v, keep, capacity)?);
            self.len = keep;
        }

        let (start, end) = (self.len, self.len + t);
        let (keys, values) = match (self.keys.as_mut(), self.values.as_mut()) {
            (Some(keys), Some(values)) => (keys, values),
            _ => unreachable!("buffers allocated above"),
        };
        keys.try_index_mut((.., .., start..end, ..), k)?;
        values.try_index_mut((.., .., start..end, ..), v)?;
        self.len = end;

        let from = match self.window {
            Some(w) if t == 1 => (end - w).max(0),
            _ => 0,
        };
        Ok((
            keys.index((.., .., from..end, ..)),
            values.index((.., .., from..end, ..)),
        ))
    }

    /// A fresh buffer of `capacity` rows shaped like `new`, holding the last
    /// `keep` valid rows of `old` at its front.
    fn regrow(&self, old: Option<&Array>, new: &Array, keep: i32, capacity: i32) -> Result<Array> {
        let s = new.shape();
        let mut buf = zeros_dtype(&[s[0], s[1], capacity, s[3]], new.dtype())?;
        if let Some(old) = old.filter(|_| keep > 0) {
            let rows = old.index((.., .., self.len - keep..self.len, ..));
            buf.try_index_mut((.., .., ..keep, ..), &rows)?;
        }
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::arange;

    /// [1, 1, n, 1] with values start..start + n.
    fn rows(start: i32, n: i32) -> Array {
        arange::<_, f32>(start, start + n, None)
            .unwrap()
            .reshape(&[1, 1, n, 1])
            .unwrap()
    }

    fn values(a: &Array) -> Vec<f32> {
        a.as_slice::<f32>().to_vec()
    }

    #[test]
    fn full_cache_returns_everything_in_order() {
        let mut c = KvCache::new(None);
        c.update(&rows(0, 300), &rows(0, 300)).unwrap();
        for p in 300..310 {
            let (k, _) = c.update(&rows(p, 1), &rows(p, 1)).unwrap();
            assert_eq!(values(&k), (0..=p).map(|x| x as f32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn sliding_cache_keeps_the_window() {
        let w = 8;
        let mut c = KvCache::new(Some(w));
        // A prefill chunk sees every kept row (the caller masks the window).
        let (k, _) = c.update(&rows(0, 5), &rows(0, 5)).unwrap();
        assert_eq!(values(&k), vec![0., 1., 2., 3., 4.]);
        // Single queries see exactly the last `w` positions, across regrowths.
        for p in 5..700 {
            let (k, v) = c.update(&rows(p, 1), &rows(p, 1)).unwrap();
            let want: Vec<f32> = ((p + 1 - w).max(0)..=p).map(|x| x as f32).collect();
            assert_eq!(values(&k), want, "at position {p}");
            assert_eq!(values(&v), want);
        }
        // A later chunk sees the w - 1 rows before it plus itself.
        let (k, _) = c.update(&rows(700, 3), &rows(700, 3)).unwrap();
        let first = values(&k)[0] as i32;
        assert!(first <= 700 - (w - 1), "too few earlier rows kept: starts at {first}");
        assert_eq!(*values(&k).last().unwrap(), 702.);
    }
}
