//! This module defines the WeightTable used in both main and PowerCheck relations
use crate::{
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Weight table in tensor form [e₁ || e₂] with Arc for zero-copy sharing.
///
/// Fresh: e₁ = [1, τ, τ², ...], e₂ = [1, τ^left, τ^{2·left}, ...]
/// Folded: e = (1-r_b)·e_old + r_b·e_new
#[derive(Clone, Debug)]
pub struct WeightTable<E: Engine> {
  /// Weight data [e₁ || e₂], length = left + right
  data: Arc<Vec<E::Scalar>>,

  /// Commitment randomness
  r: E::Scalar,

  /// Split point: e₁ = data[0..left], e₂ = data[left..]
  left: usize,
}

impl<E: Engine> WeightTable<E> {
  /// Create a new weight table
  pub fn new(data: Vec<E::Scalar>, r: E::Scalar, left: usize) -> Self {
    debug_assert!(left <= data.len());
    Self {
      data: Arc::new(data),
      r,
      left,
    }
  }

  /// First half: e₁ = [1, τ, τ², ..., τ^{left-1}] (or folded version)
  pub fn e1(&self) -> &[E::Scalar] {
    &self.data[..self.left]
  }

  /// Second half: e₂ = [1, τ^left, τ^{2·left}, ...] (or folded version)
  pub fn e2(&self) -> &[E::Scalar] {
    &self.data[self.left..]
  }

  /// Get the split point (length of e₁)
  pub fn left(&self) -> usize {
    self.left
  }

  /// Get the length of e₂
  pub fn right(&self) -> usize {
    self.data.len() - self.left
  }

  /// Get the full data as a slice
  pub fn as_slice(&self) -> &[E::Scalar] {
    &self.data
  }

  /// Get the commitment randomness
  pub fn r(&self) -> E::Scalar {
    self.r
  }

  /// Get the total length (left + right)
  pub fn len(&self) -> usize {
    self.data.len()
  }

  /// Check if empty
  pub fn is_empty(&self) -> bool {
    self.data.is_empty()
  }

  /// Fold with another weight table: self + r_b * (other - self)
  pub fn fold(&self, other: &Self, r_b: &E::Scalar) -> Self {
    debug_assert_eq!(self.left, other.left); // Split must match
    let folded: Vec<_> = self
      .data
      .par_iter()
      .zip(other.data.par_iter())
      .map(|(a, b)| *a + *r_b * (*b - *a))
      .collect();
    let r_folded = (E::Scalar::ONE - r_b) * self.r + *r_b * other.r;
    Self::new(folded, r_folded, self.left)
  }

  /// Commit to this weight table
  pub fn commit(&self, ck: &CommitmentKey<E>) -> Commitment<E> {
    E::CE::commit(ck, &self.data, &self.r)
  }
}

// Custom serialization for WeightTable
impl<E: Engine> Serialize for WeightTable<E> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    // Serialize as a tuple of (data, r, left)
    (&*self.data, &self.r, &self.left).serialize(serializer)
  }
}

impl<'de, E: Engine> Deserialize<'de> for WeightTable<E> {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let (data, r, left): (Vec<E::Scalar>, E::Scalar, usize) = Deserialize::deserialize(deserializer)?;
    Ok(Self::new(data, r, left))
  }
}

impl<E: Engine> PartialEq for WeightTable<E> {
  fn eq(&self, other: &Self) -> bool {
    self.left == other.left && self.r == other.r && *self.data == *other.data
  }
}

impl<E: Engine> Eq for WeightTable<E> {}
