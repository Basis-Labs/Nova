//! Circuit representation of PowerCheck instances for NeutronNova's ZeroFold
#![allow(dead_code)]

use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{ecc::AllocatedNonnativePoint, utils::alloc_zero},
  neutron::power_check_relation::{FoldedPowerCheckInstance, PowerCheckInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// Fresh ZC_PC instance (circuit version)
///
/// Represents the "hanging" PowerCheck that will be checked in the next iteration.
#[derive(Clone, Debug)]
pub struct AllocatedPowerCheckInstance<E: Engine> {
  pub(crate) comm_powers: AllocatedNonnativePoint<E>,
  pub(crate) tau: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedPowerCheckInstance<E> {
  /// Allocates the given `PowerCheckInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&PowerCheckInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    let comm_powers = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "alloc comm_powers"),
      inst.map(|i| i.comm_powers.to_coordinates()),
    )?;

    let tau = AllocatedNum::alloc(cs.namespace(|| "alloc tau"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |i| i.tau))
    })?;

    Ok(Self { comm_powers, tau })
  }

  /// Absorb the instance into a random oracle
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self
      .comm_powers
      .absorb_in_ro(cs.namespace(|| "absorb comm_powers"), ro)?;
    ro.absorb(&self.tau);
    Ok(())
  }
}

/// Folded NSC_PC instance (circuit version)
///
/// Represents the accumulated PowerCheck state with error term.
#[derive(Clone, Debug)]
pub struct AllocatedFoldedPowerCheckInstance<E: Engine> {
  pub(crate) pc_sumcheck_claim: AllocatedNum<E::Scalar>,
  pub(crate) comm_witness: AllocatedNonnativePoint<E>,
  pub(crate) comm_weights: AllocatedNonnativePoint<E>,
  pub(crate) tau: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedFoldedPowerCheckInstance<E> {
  /// Allocates the given `FoldedPowerCheckInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&FoldedPowerCheckInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    let pc_sumcheck_claim = AllocatedNum::alloc(cs.namespace(|| "alloc pc_sumcheck_claim"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |i| i.pc_sumcheck_claim))
    })?;

    let comm_witness = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "alloc comm_witness"),
      inst.map(|i| i.comm_witness.to_coordinates()),
    )?;

    let comm_weights = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "alloc comm_weights"),
      inst.map(|i| i.comm_weights.to_coordinates()),
    )?;

    let tau = AllocatedNum::alloc(cs.namespace(|| "alloc tau"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |i| i.tau))
    })?;

    Ok(Self {
      pc_sumcheck_claim,
      comm_witness,
      comm_weights,
      tau,
    })
  }

  /// Allocates the hardcoded default instance (all zeros)
  pub fn default<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    let pc_sumcheck_claim = alloc_zero(cs.namespace(|| "alloc pc_sumcheck_claim"));
    let comm_witness = AllocatedNonnativePoint::default(cs.namespace(|| "alloc comm_witness"))?;
    let comm_weights = AllocatedNonnativePoint::default(cs.namespace(|| "alloc comm_weights"))?;
    let tau = alloc_zero(cs.namespace(|| "alloc tau"));

    Ok(Self {
      pc_sumcheck_claim,
      comm_witness,
      comm_weights,
      tau,
    })
  }

  /// Absorb the instance into a random oracle
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    ro.absorb(&self.pc_sumcheck_claim);
    self
      .comm_witness
      .absorb_in_ro(cs.namespace(|| "absorb comm_witness"), ro)?;
    self
      .comm_weights
      .absorb_in_ro(cs.namespace(|| "absorb comm_weights"), ro)?;
    ro.absorb(&self.tau);
    Ok(())
  }

  /// Convert a fresh ZC_PC instance to folded form
  ///
  /// The fresh instance has `pc_sumcheck_claim = 0` (exactly satisfied).
  /// `comm_E` becomes `comm_weights` (shared weight table with main NSC).
  pub fn from_fresh_zc_pc(
    zc_pc: &AllocatedPowerCheckInstance<E>,
    comm_E: &AllocatedNonnativePoint<E>,
    zero_claim: &AllocatedNum<E::Scalar>,
  ) -> Self {
    Self {
      pc_sumcheck_claim: zero_claim.clone(),
      comm_witness: zc_pc.comm_powers.clone(),
      comm_weights: comm_E.clone(),
      tau: zc_pc.tau.clone(),
    }
  }

  /// Fold with another instance
  ///
  /// Computes the folded instance using linear folding for scalars,
  /// and uses provided hints for commitments (not verified in-circuit).
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
    r_b: &AllocatedNum<E::Scalar>,
    T_out_pc: &AllocatedNum<E::Scalar>,
    comm_witness_fold: &AllocatedNonnativePoint<E>,
    comm_weights_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<Self, SynthesisError> {
    // tau_fold = self.tau + r_b * (other.tau - self.tau)
    let tau_fold = AllocatedNum::alloc(cs.namespace(|| "alloc tau_fold"), || {
      let tau1 = self
        .tau
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let tau2 = other
        .tau
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(tau1 + r * (tau2 - tau1))
    })?;

    // Enforce: tau_fold - self.tau = r_b * (other.tau - self.tau)
    cs.enforce(
      || "tau_fold = self.tau + r_b * (other.tau - self.tau)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + other.tau.get_variable() - self.tau.get_variable(),
      |lc| lc + tau_fold.get_variable() - self.tau.get_variable(),
    );

    Ok(Self {
      pc_sumcheck_claim: T_out_pc.clone(),       // from sumcheck output
      comm_witness: comm_witness_fold.clone(),   // hint (passed through)
      comm_weights: comm_weights_fold.clone(),   // hint (passed through)
      tau: tau_fold,                             // constrained
    })
  }
}
