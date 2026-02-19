//! Circuit representation of NeutronNova's ZeroFold NIFS
//!
//! This module implements the in-circuit verifier for ZeroFold, which folds
//! both the main NSC relation and the PowerCheck (NSC_PC) relation together.

// Allow dead code: this circuit verifier is WIP and not yet integrated into the main IVC circuit
#![allow(dead_code)]

use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{ecc::AllocatedNonnativePoint, utils::le_bits_to_num},
  neutron::{
    circuit::{
      power_check_relation::{AllocatedFoldedPowerCheckInstance, AllocatedPowerCheckInstance},
      r1cs::AllocatedNonnativeR1CSInstance,
      relation::AllocatedFoldedInstance,
      univariate::AllocatedUniPoly,
    },
    zerofold_nifs::ZeroFoldNIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, RO2ConstantsCircuit, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of ZeroFoldNIFS
///
/// Contains two separate sumcheck polynomials for independent verification
/// of NSC and NSC_PC (PowerCheck) claims.
pub struct AllocatedZeroFoldNIFS<E: Engine> {
  /// Commitment to weight table E
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  /// NSC sumcheck polynomial
  pub(crate) poly_nsc: AllocatedUniPoly<E>,
  /// PowerCheck sumcheck polynomial
  pub(crate) poly_pc: AllocatedUniPoly<E>,
}

impl<E: Engine> AllocatedZeroFoldNIFS<E> {
  /// Allocates the given `ZeroFoldNIFS` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    nifs: Option<&ZeroFoldNIFS<E>>,
    degree: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_E"),
      nifs.map(|nifs| nifs.comm_E.to_coordinates()),
    )?;

    let poly_nsc = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly_nsc"),
      degree,
      nifs.map(|nifs| &nifs.poly_nsc),
    )?;

    let poly_pc = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly_pc"),
      degree,
      nifs.map(|nifs| &nifs.poly_pc),
    )?;

    Ok(Self {
      comm_E,
      poly_nsc,
      poly_pc,
    })
  }

  /// Verify the ZeroFold NIFS proof inside the circuit
  ///
  /// This replays the Fiat-Shamir transcript, verifies sumcheck identities,
  /// and returns the folded instances.
  ///
  /// # Arguments
  /// - `pp_digest`: Public parameters digest (binds the circuit)
  /// - `U1`: Running (accumulated) NSC instance
  /// - `U1_pc`: Running (accumulated) PowerCheck instance
  /// - `U2`: Fresh R1CS instance being folded in
  /// - `U2_pc`: Fresh ZC_PC instance (the "hanging" PowerCheck)
  /// - `comm_W_fold`: Hint for folded comm_W (passed through, not verified in-circuit)
  /// - `comm_E_fold`: Hint for folded comm_E (shared between NSC and PC)
  /// - `comm_witness_fold`: Hint for folded comm_witness (PC's folded power table)
  ///
  /// # Returns
  /// - Folded NSC instance
  /// - Folded PowerCheck instance
  /// - New "hanging" ZC_PC for next iteration
  #[allow(clippy::too_many_arguments)]
  pub fn verify<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    // Running instances (accumulated from prior folds)
    U1: &AllocatedFoldedInstance<E>,
    U1_pc: &AllocatedFoldedPowerCheckInstance<E>,
    // Fresh instances
    U2: &AllocatedNonnativeR1CSInstance<E>,
    U2_pc: &AllocatedPowerCheckInstance<E>,
    // Commitment hints (provided by prover, not verified in-circuit)
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
    comm_witness_fold: &AllocatedNonnativePoint<E>,
    ro_consts: RO2ConstantsCircuit<E>,
  ) -> Result<
    (
      AllocatedFoldedInstance<E>,
      AllocatedFoldedPowerCheckInstance<E>,
      AllocatedPowerCheckInstance<E>,
    ),
    SynthesisError,
  > {
    // ========================================
    // PHASE 1: Replay transcript
    // ========================================
    let mut ro = E::RO2Circuit::new(ro_consts);

    // Step 1: Absorb pp_digest
    ro.absorb(pp_digest);

    // Step 2: Absorb fresh R1CS instance
    // Running instance U1 does not need to be absorbed since U2.X[0] = Hash(vk, U1, i, z0, zi)
    U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

    // Step 3: Absorb fresh ZC_PC instance
    U2_pc.absorb_in_ro(cs.namespace(|| "absorb U2_pc"), &mut ro)?;

    // Step 4: Squeeze τ
    let tau_bits = ro.squeeze(cs.namespace(|| "tau_bits"), NUM_CHALLENGE_BITS, false)?;
    let tau = le_bits_to_num(cs.namespace(|| "tau"), &tau_bits)?;

    // Step 5: Absorb comm_E
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

    // Step 6: Squeeze ρ
    let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
    let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

    // ========================================
    // PHASE 2: Verify sumcheck identities
    // ========================================
    // Claims for the 1-round sumcheck (fresh instances have T=0, pc_sumcheck_claim=0)
    // claim = (1-ρ)·T_running + ρ·T_fresh = (1-ρ)·T_running

    // T_nsc = (1-rho) * U1.T
    let T_nsc = AllocatedNum::alloc(cs.namespace(|| "allocate T_nsc"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(U1_T * (E::Scalar::ONE - rho))
    })?;
    cs.enforce(
      || "enforce T_nsc = (1-rho) * U1.T",
      |lc| lc + U1.T.get_variable(),
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + T_nsc.get_variable(),
    );

    // Verify poly_nsc: poly(0) + poly(1) == T_nsc
    self
      .poly_nsc
      .check_poly_zero_poly_one_with(cs.namespace(|| "check poly_nsc(0) + poly_nsc(1) = T_nsc"), &T_nsc)?;

    // T_pc = (1-rho) * U1_pc.pc_sumcheck_claim
    let T_pc = AllocatedNum::alloc(cs.namespace(|| "allocate T_pc"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let claim = U1_pc
        .pc_sumcheck_claim
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(claim * (E::Scalar::ONE - rho))
    })?;
    cs.enforce(
      || "enforce T_pc = (1-rho) * U1_pc.pc_sumcheck_claim",
      |lc| lc + U1_pc.pc_sumcheck_claim.get_variable(),
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + T_pc.get_variable(),
    );

    // Verify poly_pc: poly(0) + poly(1) == T_pc
    self
      .poly_pc
      .check_poly_zero_poly_one_with(cs.namespace(|| "check poly_pc(0) + poly_pc(1) = T_pc"), &T_pc)?;

    // ========================================
    // PHASE 3: Absorb BOTH polynomials and squeeze r_b
    // ========================================
    // SECURITY: We must absorb both polynomials independently, not just their
    // linear combination. See comment in run_combined_sumfold for the Δ-trick attack.
    self.poly_nsc.absorb_in_ro(&mut ro);
    self.poly_pc.absorb_in_ro(&mut ro);
    let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
    let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

    // ========================================
    // PHASE 4: Compute output claims
    // ========================================
    // eq(ρ, r_b) = (1-ρ)(1-r_b) + ρ·r_b
    let eq_rho_r_b_one = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b_one"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b))
    })?;
    cs.enforce(
      || "check eq_rho_r_b_one = (1-rho) * (1-r_b)",
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + CS::one() - r_b.get_variable(),
      |lc| lc + eq_rho_r_b_one.get_variable(),
    );

    let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b)
    })?;

    // check eq_rho_r_b = (1-rho)(1-r_b) + rho*r_b
    cs.enforce(
      || "check eq_rho_r_b = eq_rho_r_b_one + rho * r_b",
      |lc| lc + rho.get_variable(),
      |lc| lc + r_b.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
    );

    // T_out_nsc = poly_nsc(r_b) / eq(ρ, r_b)
    let eval_nsc = self.poly_nsc.evaluate(cs.namespace(|| "eval_nsc"), &r_b)?;
    let T_out_nsc = AllocatedNum::alloc(cs.namespace(|| "allocate T_out_nsc"), || {
      let eval = eval_nsc.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let eq_inv = eq_rho_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
        .invert()
        .unwrap();
      Ok(eval * eq_inv)
    })?;
    cs.enforce(
      || "enforce T_out_nsc * eq_rho_r_b = eval_nsc",
      |lc| lc + T_out_nsc.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable(),
      |lc| lc + eval_nsc.get_variable(),
    );

    // T_out_pc = poly_pc(r_b) / eq(ρ, r_b)
    let eval_pc = self.poly_pc.evaluate(cs.namespace(|| "eval_pc"), &r_b)?;
    let T_out_pc = AllocatedNum::alloc(cs.namespace(|| "allocate T_out_pc"), || {
      let eval = eval_pc.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let eq_inv = eq_rho_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
        .invert()
        .unwrap();
      Ok(eval * eq_inv)
    })?;
    cs.enforce(
      || "enforce T_out_pc * eq_rho_r_b = eval_pc",
      |lc| lc + T_out_pc.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable(),
      |lc| lc + eval_pc.get_variable(),
    );

    // ========================================
    // PHASE 5: Fold instances
    // ========================================
    // Fold NSC instance
    let folded_nsc = U1.fold(
      cs.namespace(|| "fold NSC"),
      U2,
      &r_b,
      &T_out_nsc,
      comm_W_fold,
      comm_E_fold,
    )?;

    // Convert fresh ZC_PC to folded form for folding
    // Fresh instance has pc_sumcheck_claim = 0 (exactly satisfied)
    // The comm_E from the proof becomes the weights for the fresh PC instance
    let zero_claim = AllocatedNum::alloc(cs.namespace(|| "zero_claim"), || Ok(E::Scalar::ZERO))?;
    cs.enforce(
      || "enforce zero_claim = 0",
      |lc| lc + zero_claim.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    let U2_pc_folded =
      AllocatedFoldedPowerCheckInstance::from_fresh_zc_pc(U2_pc, &self.comm_E, &zero_claim);

    // Fold PC instance
    // Note: comm_E_fold is reused as comm_weights_fold (shared weight table)
    let folded_pc = U1_pc.fold(
      cs.namespace(|| "fold PC"),
      &U2_pc_folded,
      &r_b,
      &T_out_pc,
      comm_witness_fold,
      comm_E_fold,
    )?;

    // Construct new "hanging" ZC_PC for next iteration
    let new_zc_pc = AllocatedPowerCheckInstance {
      comm_powers: self.comm_E.clone(),
      tau,
    };

    Ok((folded_nsc, folded_pc, new_zc_pc))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      Circuit, ConstraintSystem,
    },
    neutron::{
      circuit::power_check_relation::AllocatedFoldedPowerCheckInstance,
      power_check_relation::{PowerCheckInstance, PowerCheckStructure},
      relation::Structure,
      zerofold_nifs::{setup_nsc, setup_nsc_pc, setup_zc_pc, ZeroFoldNIFS},
    },
    provider::{hyperkzg::EvaluationEngine as HyperKZGEE, Bn256EngineKZG},
    r1cs::R1CSShape,
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    traits::{
      circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait,
      RO2Constants, RO2ConstantsCircuit,
    },
    Commitment, CommitmentKey,
  };

  /// A test circuit that verifies a ZeroFold NIFS proof in-circuit
  struct ZeroFoldVerifyCircuit<E: Engine> {
    // Public parameters
    pp_digest: E::Scalar,
    ro_consts: RO2ConstantsCircuit<E>,

    // Running instances (before fold)
    U1: crate::neutron::relation::FoldedInstance<E>,
    U1_pc: crate::neutron::power_check_relation::FoldedPowerCheckInstance<E>,

    // Fresh instances
    U2: crate::r1cs::R1CSInstance<E>,
    U2_pc: PowerCheckInstance<E>,

    // Proof
    nifs: ZeroFoldNIFS<E>,
    degree: usize,

    // Commitment hints
    comm_W_fold: Commitment<E>,
    comm_E_fold: Commitment<E>,
    comm_witness_fold: Commitment<E>,
  }

  impl<E: Engine> Circuit<E::Scalar> for ZeroFoldVerifyCircuit<E> {
    fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      // Allocate pp_digest
      let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || Ok(self.pp_digest))?;

      // Allocate running NSC instance
      let U1 = AllocatedFoldedInstance::alloc(cs.namespace(|| "U1"), Some(&self.U1))?;

      // Allocate running PC instance
      let U1_pc =
        AllocatedFoldedPowerCheckInstance::alloc(cs.namespace(|| "U1_pc"), Some(&self.U1_pc))?;

      // Allocate fresh R1CS instance
      let U2 = AllocatedNonnativeR1CSInstance::alloc(cs.namespace(|| "U2"), Some(&self.U2))?;

      // Allocate fresh ZC_PC instance
      let U2_pc = AllocatedPowerCheckInstance::alloc(cs.namespace(|| "U2_pc"), Some(&self.U2_pc))?;

      // Allocate NIFS proof
      let nifs =
        AllocatedZeroFoldNIFS::alloc(cs.namespace(|| "nifs"), Some(&self.nifs), self.degree)?;

      // Allocate commitment hints
      let comm_W_fold = AllocatedNonnativePoint::alloc(
        cs.namespace(|| "comm_W_fold"),
        Some(self.comm_W_fold.to_coordinates()),
      )?;
      let comm_E_fold = AllocatedNonnativePoint::alloc(
        cs.namespace(|| "comm_E_fold"),
        Some(self.comm_E_fold.to_coordinates()),
      )?;
      let comm_witness_fold = AllocatedNonnativePoint::alloc(
        cs.namespace(|| "comm_witness_fold"),
        Some(self.comm_witness_fold.to_coordinates()),
      )?;

      // Run in-circuit verification
      let (_folded_nsc, _folded_pc, _new_zc_pc) = nifs.verify(
        cs.namespace(|| "verify"),
        &pp_digest,
        &U1,
        &U1_pc,
        &U2,
        &U2_pc,
        &comm_W_fold,
        &comm_E_fold,
        &comm_witness_fold,
        self.ro_consts,
      )?;

      Ok(())
    }
  }

  /// Generate a test circuit shape and commitment key
  fn generate_test_circuit<E: Engine, S: RelaxedR1CSSNARKTrait<E>>(
    num_cons: usize,
  ) -> (R1CSShape<E>, CommitmentKey<E>) {
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    let shape = shape.pad();
    (shape, ck)
  }

  /// Generate a satisfying R1CS witness
  fn generate_satisfying_witness<E: Engine>(
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    input: u64,
    num_cons: usize,
  ) -> (crate::r1cs::R1CSInstance<E>, crate::r1cs::R1CSWitness<E>) {
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(input)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );

    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs.r1cs_instance_and_witness(shape, ck).unwrap();
    let w = w.pad(shape);
    (u, w)
  }

  /// Test that the ZeroFold circuit verifier produces a satisfying circuit
  fn test_zerofold_circuit_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();
    let ro_consts_circuit = RO2ConstantsCircuit::<E>::default();
    let pp_digest = E::Scalar::ZERO;
    let num_cons = 32usize;

    // Setup test circuit
    let (shape, ck) = generate_test_circuit::<E, S>(num_cons);
    let str = Structure::new(&shape);
    let S_pc = PowerCheckStructure::from_main(&str);

    // Polynomial degree is 5 (6 evaluations: e0, e1, e2, e3, e4, e5)
    let degree = 5usize;

    // Initialize with zeros
    let (nsc_instance, nsc_witness, abc) = setup_nsc::<E>(&str);
    let (nsc_pc_instance, nsc_pc_witness) = setup_nsc_pc::<E>(&ck, &S_pc);
    let (zc_pc_instance, zc_pc_witness) = setup_zc_pc::<E>(&ck, &S_pc);

    // Generate fresh R1CS witness
    let (u_fresh, w_fresh) = generate_satisfying_witness::<E>(&shape, &ck, 2, num_cons);

    // Run ZeroFoldNIFS::prove
    let result = ZeroFoldNIFS::prove(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &S_pc,
      (&nsc_instance, &nsc_witness),
      (&abc.0, &abc.1, &abc.2),
      (&nsc_pc_instance, &nsc_pc_witness),
      (&u_fresh, &w_fresh),
      (&zc_pc_instance, &zc_pc_witness),
    )
    .expect("prove should succeed");

    // Verify native verification succeeds
    let (_verified_nsc, _verified_pc, _verified_zc_pc) = result
      .nifs
      .verify(
        &ro_consts,
        &pp_digest,
        &nsc_instance,
        &nsc_pc_instance,
        &u_fresh,
        &zc_pc_instance,
      )
      .expect("native verify should succeed");

    // Create test circuit for in-circuit verification
    let test_circuit = ZeroFoldVerifyCircuit::<E> {
      pp_digest,
      ro_consts: ro_consts_circuit,
      U1: nsc_instance,
      U1_pc: nsc_pc_instance,
      U2: u_fresh,
      U2_pc: zc_pc_instance,
      nifs: result.nifs,
      degree,
      comm_W_fold: result.folded.nsc_instance.comm_W,
      comm_E_fold: result.folded.nsc_instance.comm_E,
      comm_witness_fold: result.folded.pc_instance.comm_witness,
    };

    // First, get the shape of the verification circuit
    let mut shape_cs: ShapeCS<E> = ShapeCS::new();
    let _ = test_circuit.clone().synthesize(&mut shape_cs);
    let verify_shape = shape_cs.r1cs_shape().unwrap();
    let verify_ck =
      R1CSShape::commitment_key(&[&verify_shape], &[&*S::ck_floor()]).unwrap();

    // Now synthesize with actual values and check satisfiability
    let mut cs = SatisfyingAssignment::<E>::new();
    test_circuit
      .synthesize(&mut cs)
      .expect("circuit synthesis should succeed");

    let (inst, witness) = cs
      .r1cs_instance_and_witness(&verify_shape, &verify_ck)
      .expect("should produce valid instance/witness");

    // Check that the circuit is satisfied
    verify_shape
      .is_sat(&verify_ck, &inst, &witness)
      .expect("circuit should be satisfied");
  }

  // We need to implement Clone for the test circuit
  impl<E: Engine> Clone for ZeroFoldVerifyCircuit<E> {
    fn clone(&self) -> Self {
      Self {
        pp_digest: self.pp_digest,
        ro_consts: self.ro_consts.clone(),
        U1: self.U1.clone(),
        U1_pc: self.U1_pc.clone(),
        U2: self.U2.clone(),
        U2_pc: self.U2_pc.clone(),
        nifs: self.nifs.clone(),
        degree: self.degree,
        comm_W_fold: self.comm_W_fold,
        comm_E_fold: self.comm_E_fold,
        comm_witness_fold: self.comm_witness_fold,
      }
    }
  }

  #[test]
  fn test_zerofold_circuit() {
    test_zerofold_circuit_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
  }
}
