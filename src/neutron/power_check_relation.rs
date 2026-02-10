//! This module defines relations used in the PowerCheck folding scheme
use crate::{
  neutron::{relation::Structure, weight_table::WeightTable},
  spartan::math::Math,
  traits::{AbsorbInRO2Trait, Engine, ROTrait},
  Commitment,
};
use ff::Field;
use serde::{Deserialize, Serialize};

/// Metadata for PowerCheck relation (Construction 2).
/// Verifies e = [e₁ || e₂] contains valid powers of τ via F_PC(g₁,g₂,g₃) = g₁ - g₂·g₃ = 0
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerCheckStructure {
  /// Length of e₁. When ℓ is odd, left > right.
  /// Example: ℓ=11 → left=64, right=32
  pub left: usize,

  /// Length of e₂. Invariant: left × right = N (padded constraint count).
  /// Example: ℓ=10 → left=32, right=32 (even case)
  pub right: usize,

  /// Number of PC constraints = left + right
  pub num_cons: usize,

  /// E_pc tensor split for left half. When ⌈log₂(num_cons)⌉ is odd, left_pc > right_pc.
  pub left_pc: usize,

  /// E_pc tensor split for right half. Invariant: left_pc × right_pc ≥ num_cons.
  pub right_pc: usize,
}

impl PowerCheckStructure {
  /// Create a new PowerCheckStructure from split dimensions
  pub fn new(left: usize, right: usize) -> Self {
    let num_cons = left + right;
    let num_cons_padded = num_cons.next_power_of_two();
    let ell_pc = num_cons_padded.log_2();
    let left_pc = 1 << ell_pc.div_ceil(2);
    let right_pc = 1 << (ell_pc / 2);

    Self {
      left,
      right,
      num_cons,
      left_pc,
      right_pc,
    }
  }

  /// Derive from main relation structure
  pub fn from_main<E: Engine>(S: &Structure<E>) -> Self {
    Self::new(S.left, S.right)
  }
}

/// Fresh PowerCheck instance (ZC_PC). All constraints satisfied exactly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PowerCheckInstance<E: Engine> {
  /// Commitment to power table. Equals FoldedInstance.comm_E.
  pub comm_e: Commitment<E>,

  /// Scalar whose powers are in e: e₁[j] = τ^j, e₂[i] = τ^{i·left}
  pub tau: E::Scalar,
}

/// Fresh PowerCheck witness (ZC_PC).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PowerCheckWitness<E: Engine> {
  /// Power table, shared with FoldedWitness.E via Arc
  pub e: WeightTable<E>,
}

/// Folded PowerCheck instance (NSC_PC).
/// Contains commitments to both the original PowerCheck witness and the new E.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedPowerCheckInstance<E: Engine> {
  /// Accumulated error. Fresh: T_pc = 0.
  pub T_pc: E::Scalar,

  /// Commitment to original PowerCheck witness (powers of original tau)
  pub comm_w_pc: Commitment<E>,

  /// Commitment to new power table E (shared with main relation)
  pub comm_e: Commitment<E>,

  /// Original tau (linearly folded: τ_fold = (1-r_b)·τ_old + r_b·τ_new)
  pub tau: E::Scalar,
}

/// Folded PowerCheck witness (NSC_PC).
/// Contains both the original PowerCheck witness and the new E.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedPowerCheckWitness<E: Engine> {
  /// Original PowerCheck witness (powers of original tau)
  pub w_pc: WeightTable<E>,

  /// New power table E (powers of fresh tau), shared with main relation
  pub e: WeightTable<E>,
}

impl<E: Engine> FoldedPowerCheckInstance<E> {
  /// Create a default instance
  pub fn default(S: &PowerCheckStructure) -> Self {
    // Suppress unused warning - S is kept for API consistency
    let _ = S;
    Self {
      T_pc: E::Scalar::ZERO,
      comm_w_pc: Commitment::<E>::default(),
      comm_e: Commitment::<E>::default(),
      tau: E::Scalar::ZERO,
    }
  }

  /// Fold the instance with a fresh instance
  pub fn fold(
    &self,
    u2: &PowerCheckInstance<E>,
    r_b: &E::Scalar,
    comm_e_new: &Commitment<E>,
    T_pc_out: &E::Scalar,
  ) -> Self {
    let one_minus_r = E::Scalar::ONE - r_b;

    Self {
      T_pc: *T_pc_out,
      comm_w_pc: self.comm_w_pc * one_minus_r + u2.comm_e * *r_b,
      comm_e: self.comm_e * one_minus_r + *comm_e_new * *r_b,
      tau: one_minus_r * self.tau + *r_b * u2.tau,
    }
  }
}

impl<E: Engine> FoldedPowerCheckWitness<E> {
  /// Create a default witness
  pub fn default(S: &PowerCheckStructure) -> Self {
    Self {
      w_pc: WeightTable::new(vec![E::Scalar::ZERO; S.left + S.right], E::Scalar::ZERO, S.left),
      e: WeightTable::new(vec![E::Scalar::ZERO; S.left + S.right], E::Scalar::ZERO, S.left),
    }
  }

  /// Fold the witness with a fresh witness
  pub fn fold(&self, w2: &PowerCheckWitness<E>, e_new: &WeightTable<E>, r_b: &E::Scalar) -> Self {
    Self {
      w_pc: self.w_pc.fold(&w2.e, r_b),
      e: self.e.fold(e_new, r_b),
    }
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for FoldedPowerCheckInstance<E> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    ro.absorb(self.T_pc);
    self.comm_w_pc.absorb_in_ro2(ro);
    self.comm_e.absorb_in_ro2(ro);
    ro.absorb(self.tau);
  }
}
