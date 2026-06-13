//! Thin wrapper over the safetensors weight map with dtype coercion helpers.

use anyhow::{anyhow, Result};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;

pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    pub fn load(path: &str) -> Result<Self> {
        let map = Array::load_safetensors(path).map_err(|e| anyhow!("load_safetensors: {e}"))?;
        Ok(Self { map })
    }

    /// Raw array as stored (BF16 for most cohere tensors).
    pub fn raw(&self, key: &str) -> Result<&Array> {
        self.map
            .get(key)
            .ok_or_else(|| anyhow!("missing weight key: {key}"))
    }

    /// Weight materialized to f32 (we run the POC in f32 for numerical parity
    /// with the Python ground-truth dumps, which are f32).
    pub fn f32(&self, key: &str) -> Result<Array> {
        let a = self.raw(key)?;
        a.as_dtype(Dtype::Float32)
            .map_err(|e| anyhow!("as_dtype f32 for {key}: {e}"))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
}
