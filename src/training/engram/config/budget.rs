//! Capacity changes preserve the graph and serialize the actual prime sizes.
use super::{Config, HEAD_DIM, HEADS, LAYERS, is_prime};
use crate::AppResult;

impl Config {
    pub fn multistream_with_table_budget(
        requested: Option<usize>,
    ) -> AppResult<[Self; LAYERS.len()]> {
        let mut configs = std::array::from_fn(Self::reference_multistream);
        let default = configs.iter().map(Self::table_len).sum::<usize>();
        let Some(requested) = requested else {
            return Ok(configs);
        };
        if !(default..=2_000_000_000).contains(&requested) {
            return Err(format!(
                "Engram table budget must be between {default} and 2000000000 lookup values"
            )
            .into());
        }
        if requested == default {
            return Ok(configs);
        }
        let values_per_capacity = LAYERS.len() * 2 * HEADS * HEAD_DIM;
        let mut prime = (requested / values_per_capacity) as u32;
        for config in &mut configs {
            for slot in &mut config.primes {
                loop {
                    prime += 1;
                    if prime > 1_000_000 {
                        return Err("Engram table budget exceeds checkpoint prime limits".into());
                    }
                    if is_prime(prime) {
                        break;
                    }
                }
                *slot = prime;
            }
        }
        let actual = configs.iter().map(Self::table_len).sum::<usize>();
        if actual < requested || actual - requested >= 1_000_000 {
            return Err("Engram table budget cannot satisfy the prime rounding bound".into());
        }
        Ok(configs)
    }
}
