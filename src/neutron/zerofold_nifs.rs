//! ZeroFold NIFS: Combined NSC + NSC_PC (PowerCheck) folding scheme
//!
//! This module implements the ZeroFold prover which folds an R1CS instance-witness pair
//! into the accumulating (NSC, NSC_PC) state.
//!
//! ## Security Design: Two Separate Polynomials
//!
//! The proof contains two separate sumcheck polynomials (`poly_nsc` and `poly_pc`)
//! rather than a combined polynomial with a hint. This provides stronger security:
//!
//! 1. **Independent verification**: Each polynomial can be verified against its claim
//!    - NSC: `poly_nsc(0) + poly_nsc(1) == T_nsc`
//!    - NSC_PC: `poly_pc(0) + poly_pc(1) == T_pc`
//!
//! 2. **No hints**: Verifier derives output claims directly from polynomials
//!    - `T_out_nsc = poly_nsc(r_b) / eq(ρ, r_b)`
//!    - `T_out_pc = poly_pc(r_b) / eq(ρ, r_b)`
//!
//! 3. **Stronger local soundness**: Cheating requires forging a polynomial, not just a scalar
//!
//! The γ challenge is still used for transcript binding (Fiat-Shamir) - a combined polynomial
//! `poly_nsc + γ·poly_pc` is absorbed in the transcript before squeezing `r_b`.

#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  neutron::{
    nested_sumcheck::convert_to_nsc,
    power_check_relation::{
      FoldedPowerCheckInstance, FoldedPowerCheckWitness, PowerCheckInstance, PowerCheckStructure,
      PowerCheckWitness,
    },
    relation::{FoldedInstance, FoldedWitness, Structure},
    sumcheck::run_combined_sumfold,
    weight_table::WeightTable,
  },
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::polys::univariate::UniPoly,
  traits::{AbsorbInRO2Trait, Engine, RO2Constants, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// ZeroFold NIFS proof (NSC + NSC_PC)
///
/// Contains two separate sumcheck polynomials for independent verification.
/// Verifier derives output claims directly: `T_out = poly(r_b) / eq(ρ, r_b)`
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ZeroFoldNIFS<E: Engine> {
  /// Commitment to weight table E
  pub(crate) comm_E: Commitment<E>,
  /// NSC sumcheck polynomial
  pub(crate) poly_nsc: UniPoly<E::Scalar>,
  /// PowerCheck sumcheck polynomial
  pub(crate) poly_pc: UniPoly<E::Scalar>,
}

/// Complete folded state: NSC + NSC_PC + cached matrix-vector products
///
/// Returned by the unified `fold` function which folds both NSC and NSC_PC together.
/// The weight table E is folded once and shared between `nsc_witness.E` and `pc_witness.witness`.
#[derive(Clone, Debug)]
pub struct FoldedState<E: Engine> {
  // === Main NSC ===
  /// Folded NSC instance
  pub nsc_instance: FoldedInstance<E>,
  /// Folded NSC witness
  pub nsc_witness: FoldedWitness<E>,
  /// Folded Az = A·z vector (cached for next iteration)
  pub Az: Vec<E::Scalar>,
  /// Folded Bz = B·z vector (cached for next iteration)
  pub Bz: Vec<E::Scalar>,
  /// Folded Cz = C·z vector (cached for next iteration)
  pub Cz: Vec<E::Scalar>,
  // === PowerCheck NSC_PC ===
  /// Folded PowerCheck instance
  pub pc_instance: FoldedPowerCheckInstance<E>,
  /// Folded PowerCheck witness
  pub pc_witness: FoldedPowerCheckWitness<E>,
}

/// Output of ZeroFold prove (handles both NSC and NSC_PC)
///
/// Per NeutronNova Construction 4: The proof contains a SINGLE combined polynomial
/// `poly = poly_nsc + γ·poly_pc` where γ is a random challenge.
#[derive(Clone, Debug)]
pub struct ZeroFoldOutput<E: Engine> {
  // === Proof ===
  /// ZeroFold proof (comm_E, combined poly, T_out_pc)
  pub nifs: ZeroFoldNIFS<E>,
  /// γ challenge used to combine NSC and NSC_PC polynomials
  pub gamma: E::Scalar,

  // === Folded results ===
  /// Folded state containing NSC + NSC_PC + cached Az/Bz/Cz
  pub folded: FoldedState<E>,
  /// New ZC_PC (hanging check for next iteration)
  pub new_zc_pc: (PowerCheckInstance<E>, PowerCheckWitness<E>),

  // === Challenges (needed for verification/reconstruction) ===
  /// τ used to generate E
  pub tau: E::Scalar,
  /// ρ (RLC challenge)
  pub rho: E::Scalar,
  /// r_b (folding challenge)
  pub r_b: E::Scalar,
}

/// Construction 4: Fold both NSC and NSC_PC together
///
/// Folds all linear objects using: a₁ + r_b · (a₂ - a₁)
/// The weight table E is folded once and shared between NSC witness and PowerCheck witness.
///
/// Returns `FoldedState` containing:
/// - Folded NSC instance and witness
/// - Cached Az/Bz/Cz for next iteration (saves one sparse matmul)
/// - Folded PowerCheck instance and witness
pub fn fold<E: Engine>(
  // NSC (1, 2)
  nsc1: (&FoldedInstance<E>, &FoldedWitness<E>),
  nsc2: (&FoldedInstance<E>, &FoldedWitness<E>),
  abc1: (&[E::Scalar], &[E::Scalar], &[E::Scalar]),
  abc2: (&[E::Scalar], &[E::Scalar], &[E::Scalar]),
  // PowerCheck (1, 2)
  pc1: (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
  pc2: (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
  // Challenges and claims
  r_b: &E::Scalar,
  sumcheck_claim_out_nsc: &E::Scalar,
  sumcheck_claim_out_pc: &E::Scalar,
) -> FoldedState<E> {
  // Destructure for clarity
  let (U1, W1) = nsc1;
  let (U2, W2) = nsc2;
  let (Az1, Bz1, Cz1) = abc1;
  let (Az2, Bz2, Cz2) = abc2;
  let (U1_pc, W1_pc) = pc1;
  let (U2_pc, W2_pc) = pc2;

  // Helper: fold two vectors element-wise using a + r_b*(b-a)
  let fold_vec = |v1: &[E::Scalar], v2: &[E::Scalar]| -> Vec<E::Scalar> {
    v1.par_iter()
      .zip(v2.par_iter())
      .map(|(a, b)| *a + *r_b * (*b - *a))
      .collect()
  };

  // ===== Fold E once (shared between NSC witness and PC witness) =====
  let E = W1.E.fold(&W2.E, r_b);

  // ===== Main NSC =====
  let nsc_instance = U1.fold_with(U2, r_b, sumcheck_claim_out_nsc);
  let nsc_witness = W1.fold_with(W2, E.clone(), r_b);

  // Fold Az/Bz/Cz in parallel (caching optimization - saves one sparse matmul per iteration)
  let ((Az, Bz), Cz) = rayon::join(
    || rayon::join(|| fold_vec(Az1, Az2), || fold_vec(Bz1, Bz2)),
    || fold_vec(Cz1, Cz2),
  );

  // ===== PowerCheck NSC_PC =====
  let pc_instance = U1_pc.fold(U2_pc, r_b, sumcheck_claim_out_pc);
  let pc_witness = W1_pc.fold(W2_pc, r_b);

  FoldedState {
    nsc_instance,
    nsc_witness,
    Az,
    Bz,
    Cz,
    pc_instance,
    pc_witness,
  }
}

// ============================================================================
// Setup Functions (Step 0 Initialization)
// ============================================================================

/// Initialize NSC relation with zero/default instances
///
/// Returns (instance, witness, (Az, Bz, Cz)) all initialized to zero.
/// The zero witness satisfies the zero instance with T=0.
pub fn setup_nsc<E: Engine>(
  S: &Structure<E>,
) -> (
  FoldedInstance<E>,
  FoldedWitness<E>,
  (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>),
) {
  // All zeros: W=0, u=0, X=0, T=0, E=0
  let instance = FoldedInstance::default(S);
  let witness = FoldedWitness::default(S);

  // z = [W, u, X] = [zeros, 0, zeros] → Az = Bz = Cz = zeros
  let num_cons = S.left * S.right;
  let Az = vec![E::Scalar::ZERO; num_cons];
  let Bz = vec![E::Scalar::ZERO; num_cons];
  let Cz = vec![E::Scalar::ZERO; num_cons];

  (instance, witness, (Az, Bz, Cz))
}

/// Initialize NSC_PC relation with zero/default instances
///
/// Returns (instance, witness) with pc_sumcheck_claim=0, tau=0, and properly committed witness tables.
pub fn setup_nsc_pc<E: Engine>(
  ck: &CommitmentKey<E>,
  S_pc: &PowerCheckStructure,
) -> (FoldedPowerCheckInstance<E>, FoldedPowerCheckWitness<E>) {
  let witness = FoldedPowerCheckWitness::default(S_pc);
  let instance = FoldedPowerCheckInstance::from_witness(ck, &witness);
  (instance, witness)
}

/// Initialize ZC_PC (PowerCheck) for step 0
///
/// With tau=0, the power table is [1, 0, 0, ...] ∥ [1, 0, 0, ...]
/// which satisfies all PowerCheck constraints since e[i] = e[i-1]·τ = e[i-1]·0 = 0.
pub fn setup_zc_pc<E: Engine>(
  ck: &CommitmentKey<E>,
  S_pc: &PowerCheckStructure,
) -> (PowerCheckInstance<E>, PowerCheckWitness<E>) {
  let left = S_pc.left;
  let right = S_pc.right;

  // tau=0, so e₁ = [1, 0, 0, ...] and e₂ = [1, 0, 0, ...]
  let mut powers_vec = vec![E::Scalar::ZERO; left + right];
  powers_vec[0] = E::Scalar::ONE; // e₁[0] = 1 (base case)
  powers_vec[left] = E::Scalar::ONE; // e₂[0] = 1 (base case)

  let powers = WeightTable::new(powers_vec, E::Scalar::ZERO, left);
  let comm_powers = powers.commit(ck);

  let instance = PowerCheckInstance {
    comm_powers,
    tau: E::Scalar::ZERO,
  };
  let witness = PowerCheckWitness { powers };

  (instance, witness)
}

impl<E: Engine> ZeroFoldNIFS<E> {
  /// ZeroFold prover: folds an R1CS instance-witness pair into the accumulating (NSC, NSC_PC) state.
  ///
  /// Recurring state: (NSC, NSC_PC, ZC_PC) with cached (Az, Bz, Cz)
  ///
  /// Input:
  /// - (NSC, NSC_PC, ZC_PC) with cached (Az, Bz, Cz)
  /// - Fresh zero-check R1CS relation (ZC)
  ///
  /// Output:
  /// - (NSC', NSC_PC', ZC_PC') with new (Az', Bz', Cz')
  #[allow(clippy::too_many_arguments)]
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    S_pc: &PowerCheckStructure,
    // Accumulated NSC
    nsc: (&FoldedInstance<E>, &FoldedWitness<E>),
    abc: (&[E::Scalar], &[E::Scalar], &[E::Scalar]),
    // Accumulated NSC_PC
    nsc_pc: (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
    // Fresh R1CS (ZC)
    r1cs: (&R1CSInstance<E>, &R1CSWitness<E>),
    // Fresh ZC_PC (from previous iteration)
    zc_pc: (&PowerCheckInstance<E>, &PowerCheckWitness<E>),
  ) -> Result<ZeroFoldOutput<E>, NovaError> {
    // === PHASE 1: Transcript Setup ===
    // Per paper Construction 3: Only absorb FRESH instances before τ.
    // Accumulated instances are implicitly bound through prior fold transcripts.
    let mut ro = E::RO2::new(ro_consts.clone());
    ro.absorb(*pp_digest);

    // === PHASE 2: ZC to NSC conversion (Construction 3) ===
    let conv = convert_to_nsc(ck, S, r1cs, zc_pc, &mut ro)?;

    // Rebind for fold (1 = accumulated, 2 = converted fresh)
    let (U1, W1) = nsc;
    let nsc2 = conv.nsc;
    let abc1 = abc;
    let abc2 = (conv.Az.as_slice(), conv.Bz.as_slice(), conv.Cz.as_slice());
    let (U1_pc, W1_pc) = nsc_pc;
    let (U2_pc, W2_pc) = conv.nsc_pc;

    // === PHASE 3: Squeeze ρ (RLC challenge) ===
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // === PHASE 4: Run combined sumfold (Construction 4) ===
    // This samples γ, computes poly_nsc and poly_pc, combines them as poly = poly_nsc + γ·poly_pc,
    // absorbs the combined poly, squeezes r_b, and computes output claims.
    let sumcheck_out = run_combined_sumfold::<E>(
      &rho,
      // NSC: (1 = accumulated, 2 = fresh)
      U1.T,
      (S.left, S.right),
      W1.E.as_slice(),
      abc1,
      conv.E.as_slice(),
      abc2,
      // NSC_PC: (1 = accumulated, 2 = fresh)
      U1_pc.pc_sumcheck_claim,
      S_pc,
      (W1_pc.weights.e1(), W1_pc.weights.e2()),
      (W2_pc.weights.e1(), W2_pc.weights.e2()),
      (W1_pc.witness.e1(), W1_pc.witness.e2()),
      (W2_pc.witness.e1(), W2_pc.witness.e2()),
      &U1_pc.tau,
      &U2_pc.tau,
      &mut ro,
    )?;

    let gamma = sumcheck_out.gamma;
    let r_b = sumcheck_out.r_b;

    // === PHASE 5: Fold both NSC and NSC_PC ===
    let folded = fold::<E>(
      nsc,
      (&nsc2.0, &nsc2.1),
      abc1,
      (abc2.0, abc2.1, abc2.2),
      nsc_pc,
      (&U2_pc, &W2_pc),
      &r_b,
      &sumcheck_out.sumcheck_claim_out_nsc,
      &sumcheck_out.sumcheck_claim_out_pc,
    );

    // === PHASE 6: Return ZeroFold output ===
    Ok(ZeroFoldOutput {
      nifs: ZeroFoldNIFS {
        comm_E: conv.comm_E,
        poly_nsc: sumcheck_out.poly_nsc,
        poly_pc: sumcheck_out.poly_pc,
      },
      gamma,
      folded,
      new_zc_pc: conv.zc_pc,
      tau: conv.tau,
      rho,
      r_b,
    })
  }

  /// Verify a ZeroFold proof and return the folded instances.
  ///
  /// This replays the Fiat-Shamir transcript from `prove()` to derive the same
  /// challenges (τ, ρ, γ, r_b), then verifies the sumcheck identities and
  /// computes the folded instances.
  ///
  /// # Arguments
  /// - `ro_consts`: Random oracle constants (same as prover)
  /// - `pp_digest`: Public parameters digest (binds the circuit)
  /// - `U1`: Running (accumulated) NSC instance from prior folds
  /// - `U1_pc`: Running (accumulated) PowerCheck instance from prior folds
  /// - `U2`: Fresh R1CS instance being folded in
  /// - `U2_pc`: Fresh ZC_PC instance (the "hanging" PowerCheck from prior iteration)
  ///
  /// # Returns
  /// - `FoldedInstance`: New folded NSC instance
  /// - `FoldedPowerCheckInstance`: New folded PowerCheck instance
  /// - `PowerCheckInstance`: New "hanging" ZC_PC for next iteration
  #[allow(clippy::type_complexity)]
  pub fn verify(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    // Running instances (accumulated from prior folds)
    U1: &FoldedInstance<E>,
    U1_pc: &FoldedPowerCheckInstance<E>,
    // Fresh instances (public inputs for this folding step)
    U2: &R1CSInstance<E>,
    U2_pc: &PowerCheckInstance<E>,
  ) -> Result<(FoldedInstance<E>, FoldedPowerCheckInstance<E>, PowerCheckInstance<E>), NovaError> {
    // ========================================
    // PHASE 1: Replay transcript (must match prove exactly)
    // ========================================
    let mut ro = E::RO2::new(ro_consts.clone());

    // Step 1: Absorb pp_digest
    ro.absorb(*pp_digest);

    // Step 2: Absorb fresh R1CS instance
    U2.absorb_in_ro2(&mut ro);

    // Step 3: Absorb fresh ZC_PC instance
    U2_pc.absorb_in_ro2(&mut ro);

    // Step 4: Squeeze τ (used to construct E; verifier gets comm_E from proof)
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // Step 5: Absorb comm_E
    self.comm_E.absorb_in_ro2(&mut ro);

    // Step 6: Squeeze ρ
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let one_minus_rho = E::Scalar::ONE - rho;

    // Step 7: Squeeze γ
    let gamma = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // ========================================
    // PHASE 2: Verify sumcheck identities
    // ========================================
    // Claims for the 1-round sumcheck (fresh instances have T=0, pc_sumcheck_claim=0)
    // claim = (1-ρ)·T_running + ρ·T_fresh = (1-ρ)·T_running
    let claim_nsc = one_minus_rho * U1.T;
    let claim_pc = one_minus_rho * U1_pc.pc_sumcheck_claim;

    // Verify poly_nsc: poly(0) + poly(1) == claim_nsc
    if self.poly_nsc.eval_at_zero() + self.poly_nsc.eval_at_one() != claim_nsc {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Verify poly_pc: poly(0) + poly(1) == claim_pc
    if self.poly_pc.eval_at_zero() + self.poly_pc.eval_at_one() != claim_pc {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // ========================================
    // PHASE 3: Absorb combined polynomial and squeeze r_b
    // ========================================
    // Prover absorbs poly_combined = poly_nsc + γ·poly_pc
    let poly_combined = {
      let poly_pc_scaled = self.poly_pc.scaled(&gamma);
      self.poly_nsc.add(&poly_pc_scaled)
    };
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly_combined, &mut ro);

    // Squeeze r_b
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // ========================================
    // PHASE 4: Compute output claims
    // ========================================
    // eq(ρ, r_b) = (1-ρ)(1-r_b) + ρ·r_b
    let eq_rho_r_b = one_minus_rho * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_inv: E::Scalar =
      Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;

    // T_out = poly(r_b) / eq(ρ, r_b)
    let T_out_nsc = self.poly_nsc.evaluate(&r_b) * eq_inv;
    let T_out_pc = self.poly_pc.evaluate(&r_b) * eq_inv;

    // ========================================
    // PHASE 5: Fold instances
    // ========================================
    // Fold NSC instance
    let folded_nsc = U1.fold(U2, &self.comm_E, &r_b, &T_out_nsc)?;

    // Convert fresh ZC_PC to FoldedPowerCheckInstance for folding
    // Fresh instance has pc_sumcheck_claim = 0 (exactly satisfied)
    let U2_pc_folded = FoldedPowerCheckInstance::from_fresh_zc_pc(U2_pc, self.comm_E);

    // Fold PC instance
    let folded_pc = U1_pc.fold(&U2_pc_folded, &r_b, &T_out_pc);

    // Construct new "hanging" ZC_PC for next iteration
    let new_zc_pc = PowerCheckInstance {
      comm_powers: self.comm_E,
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
    neutron::nested_sumcheck::{
      pc_residual_at, reconstruct_fresh_from_tau, verify_nsc_claim_bruteforce,
      verify_nsc_pc_claim_bruteforce,
    },
    provider::{
      hyperkzg::EvaluationEngine as HyperKZGEE, ipa_pc::EvaluationEngine, Bn256EngineKZG,
      PallasEngine,
    },
    r1cs::R1CSShape,
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    traits::{
      circuit::NonTrivialCircuit, commitment::CommitmentEngineTrait, snark::RelaxedR1CSSNARKTrait,
      Engine, RO2Constants,
    },
  };

  // ============================================================================
  // Test Verification Helpers
  // ============================================================================

  /// Verify that fresh ZC_PC has all residuals = 0
  fn verify_zc_pc_valid<E: Engine>(
    S_pc: &PowerCheckStructure,
    instance: &PowerCheckInstance<E>,
    witness: &PowerCheckWitness<E>,
  ) -> Result<(), NovaError> {
    let e1 = witness.powers.e1();
    let e2 = witness.powers.e2();
    let tau = instance.tau;
    let left = S_pc.left;

    for i in 0..S_pc.num_cons {
      let residual = pc_residual_at(i, left, e1, e2, tau);
      if residual != E::Scalar::ZERO {
        return Err(NovaError::UnSat {
          reason: format!("ZC_PC residual nonzero at index {}: {:?}", i, residual),
        });
      }
    }
    Ok(())
  }

  /// Verify linear folding correctness for NSC
  fn verify_nsc_linear_folding<E: Engine>(
    (U1, W1): (&FoldedInstance<E>, &FoldedWitness<E>),
    (U2, W2): (&FoldedInstance<E>, &FoldedWitness<E>),
    (U_folded, W_folded): (&FoldedInstance<E>, &FoldedWitness<E>),
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check W folding
    for (i, w_f) in W_folded.W.iter().enumerate() {
      let expected = W1.W[i] * one_minus_r + W2.W[i] * *r_b;
      if *w_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("W folding mismatch at index {}", i),
        });
      }
    }

    // Check E folding
    for (i, e_f) in W_folded.E.as_slice().iter().enumerate() {
      let expected = W1.E.as_slice()[i] * one_minus_r + W2.E.as_slice()[i] * *r_b;
      if *e_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("E folding mismatch at index {}", i),
        });
      }
    }

    // Check u folding
    let expected_u = U1.u * one_minus_r + U2.u * *r_b;
    if U_folded.u != expected_u {
      return Err(NovaError::UnSat {
        reason: format!("u folding mismatch: {:?} != {:?}", U_folded.u, expected_u),
      });
    }

    // Check X folding
    for (i, x_f) in U_folded.X.iter().enumerate() {
      let expected = U1.X[i] * one_minus_r + U2.X[i] * *r_b;
      if *x_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("X folding mismatch at index {}", i),
        });
      }
    }

    Ok(())
  }

  /// Verify linear folding for NSC_PC
  fn verify_nsc_pc_linear_folding<E: Engine>(
    (U1, W1): (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
    (U2, W2): (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
    folded: &FoldedState<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check tau folding
    let expected_tau = U1.tau * one_minus_r + U2.tau * *r_b;
    if folded.pc_instance.tau != expected_tau {
      return Err(NovaError::UnSat {
        reason: format!(
          "tau folding mismatch: {:?} != {:?}",
          folded.pc_instance.tau, expected_tau
        ),
      });
    }

    // Check witness (power table) folding
    for (i, w_f) in folded.pc_witness.witness.as_slice().iter().enumerate() {
      let expected = W1.witness.as_slice()[i] * one_minus_r + W2.witness.as_slice()[i] * *r_b;
      if *w_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("PC witness folding mismatch at index {}", i),
        });
      }
    }

    // Check weights folding (should equal folded E)
    for (i, w_f) in folded.pc_witness.weights.as_slice().iter().enumerate() {
      let expected = W1.weights.as_slice()[i] * one_minus_r + W2.weights.as_slice()[i] * *r_b;
      if *w_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("PC weights folding mismatch at index {}", i),
        });
      }
    }

    Ok(())
  }

  /// Verify commitment homomorphism
  fn verify_commitment_homomorphism<E: Engine>(
    U1: &FoldedInstance<E>,
    U2: &FoldedInstance<E>,
    U_folded: &FoldedInstance<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check comm_W homomorphism
    let expected_comm_W = U1.comm_W * one_minus_r + U2.comm_W * *r_b;
    if U_folded.comm_W != expected_comm_W {
      return Err(NovaError::UnSat {
        reason: "comm_W homomorphism violated".to_string(),
      });
    }

    // Check comm_E homomorphism
    let expected_comm_E = U1.comm_E * one_minus_r + U2.comm_E * *r_b;
    if U_folded.comm_E != expected_comm_E {
      return Err(NovaError::UnSat {
        reason: "comm_E homomorphism violated".to_string(),
      });
    }

    Ok(())
  }

  /// Verify sumcheck polynomial identity: poly(0) + poly(1) == T_claim
  fn verify_sumcheck_identity<E: Engine>(
    poly: &UniPoly<E::Scalar>,
    T_claim: &E::Scalar,
  ) -> Result<(), NovaError> {
    let sum = poly.eval_at_zero() + poly.eval_at_one();
    if sum != *T_claim {
      return Err(NovaError::InvalidSumcheckProof);
    }
    Ok(())
  }

  /// Verify NSC witness commitment: recompute commitment from witness and check it matches instance
  fn verify_nsc_witness_commitment<E: Engine>(
    ck: &CommitmentKey<E>,
    U: &FoldedInstance<E>,
    W: &FoldedWitness<E>,
  ) -> Result<(), NovaError> {
    let comm_W = E::CE::commit(ck, &W.W, &W.r_W);
    let comm_E = W.E.commit(ck);

    if comm_W != U.comm_W {
      return Err(NovaError::UnSat {
        reason: "NSC comm_W mismatch: recomputed != claimed".to_string(),
      });
    }
    if comm_E != U.comm_E {
      return Err(NovaError::UnSat {
        reason: "NSC comm_E mismatch: recomputed != claimed".to_string(),
      });
    }
    Ok(())
  }

  /// Verify NSC_PC witness commitment: recompute commitment from witness and check it matches instance
  fn verify_nsc_pc_witness_commitment<E: Engine>(
    ck: &CommitmentKey<E>,
    U: &FoldedPowerCheckInstance<E>,
    W: &FoldedPowerCheckWitness<E>,
  ) -> Result<(), NovaError> {
    let comm_witness = W.witness.commit(ck);
    let comm_weights = W.weights.commit(ck);

    if comm_witness != U.comm_witness {
      return Err(NovaError::UnSat {
        reason: "NSC_PC comm_witness mismatch: recomputed != claimed".to_string(),
      });
    }
    if comm_weights != U.comm_weights {
      return Err(NovaError::UnSat {
        reason: "NSC_PC comm_weights mismatch: recomputed != claimed".to_string(),
      });
    }
    Ok(())
  }

  /// Generate a test circuit and commitment key
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
  ) -> (R1CSInstance<E>, R1CSWitness<E>) {
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

  // ============================================================================
  // ZeroFold Tests
  // ============================================================================

  /// Test a single fold from zero initialization
  fn test_single_fold_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = E::Scalar::ZERO;
    let num_cons = 32usize;

    // Setup
    let (shape, ck) = generate_test_circuit::<E, S>(num_cons);
    let str = Structure::new(&shape);
    let S_pc = PowerCheckStructure::from_main(&str);

    // Initialize with zeros
    let (nsc_instance, nsc_witness, abc) = setup_nsc::<E>(&str);
    let (nsc_pc_instance, nsc_pc_witness) = setup_nsc_pc::<E>(&ck, &S_pc);
    let (zc_pc_instance, zc_pc_witness) = setup_zc_pc::<E>(&ck, &S_pc);

    // Verify initial ZC_PC is valid
    verify_zc_pc_valid::<E>(&S_pc, &zc_pc_instance, &zc_pc_witness)
      .expect("Initial ZC_PC should be valid");

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
    .expect("ZeroFoldNIFS::prove should succeed");

    // === VERIFY PROOF ===
    let (verified_nsc, verified_pc, verified_zc_pc) = result
      .nifs
      .verify(
        &ro_consts,
        &pp_digest,
        &nsc_instance,
        &nsc_pc_instance,
        &u_fresh,
        &zc_pc_instance,
      )
      .expect("ZeroFoldNIFS::verify should succeed");

    // Check verifier output matches prover output
    assert_eq!(
      verified_nsc, result.folded.nsc_instance,
      "Verified NSC instance should match prover's"
    );
    assert_eq!(
      verified_pc, result.folded.pc_instance,
      "Verified PC instance should match prover's"
    );
    assert_eq!(
      verified_zc_pc, result.new_zc_pc.0,
      "Verified ZC_PC should match prover's"
    );

    // === VERIFY FINAL STATE ===

    // 1. Verify Structure::is_sat
    str
      .is_sat(&ck, &result.folded.nsc_instance, &result.folded.nsc_witness)
      .expect("Folded NSC should satisfy Structure::is_sat");

    // 2. Verify NSC claim via brute force
    verify_nsc_claim_bruteforce::<E>(
      &str,
      &result.folded.nsc_instance,
      &result.folded.nsc_witness,
    )
    .expect("NSC claim should match brute force computation");

    // 3. Verify NSC_PC claim via brute force
    verify_nsc_pc_claim_bruteforce::<E>(
      &S_pc,
      &result.folded.pc_instance,
      &result.folded.pc_witness,
    )
    .expect("NSC_PC claim should match brute force computation");

    // 4. Verify new ZC_PC is valid
    verify_zc_pc_valid::<E>(&S_pc, &result.new_zc_pc.0, &result.new_zc_pc.1)
      .expect("New ZC_PC should be valid");

    // === VERIFY LINEAR FOLDING ===

    // Reconstruct fresh NSC/NSC_PC from tau for verification
    let fresh_nsc = reconstruct_fresh_from_tau::<E>(
      &str,
      (&u_fresh, &w_fresh),
      (&zc_pc_instance, &zc_pc_witness),
      &result.tau,
      &result.nifs.comm_E,
    )
    .expect("Reconstruction should succeed");

    // 5. Verify NSC linear folding
    verify_nsc_linear_folding::<E>(
      (&nsc_instance, &nsc_witness),
      (&fresh_nsc.nsc.0, &fresh_nsc.nsc.1),
      (&result.folded.nsc_instance, &result.folded.nsc_witness),
      &result.r_b,
    )
    .expect("NSC linear folding should be correct");

    // 6. Verify NSC_PC linear folding
    verify_nsc_pc_linear_folding::<E>(
      (&nsc_pc_instance, &nsc_pc_witness),
      (&fresh_nsc.nsc_pc.0, &fresh_nsc.nsc_pc.1),
      &result.folded,
      &result.r_b,
    )
    .expect("NSC_PC linear folding should be correct");

    // 7. Verify commitment homomorphism
    verify_commitment_homomorphism::<E>(
      &nsc_instance,
      &fresh_nsc.nsc.0,
      &result.folded.nsc_instance,
      &result.r_b,
    )
    .expect("Commitment homomorphism should hold");

    // 8. Verify separate sumcheck identities
    let T_nsc_claim = (E::Scalar::ONE - result.rho) * nsc_instance.T;
    let T_pc_claim = (E::Scalar::ONE - result.rho) * nsc_pc_instance.pc_sumcheck_claim;
    verify_sumcheck_identity::<E>(&result.nifs.poly_nsc, &T_nsc_claim)
      .expect("NSC sumcheck identity should hold");
    verify_sumcheck_identity::<E>(&result.nifs.poly_pc, &T_pc_claim)
      .expect("PowerCheck sumcheck identity should hold");

    // 9. Verify witness commitments (recompute and check match)
    verify_nsc_witness_commitment::<E>(
      &ck,
      &result.folded.nsc_instance,
      &result.folded.nsc_witness,
    )
    .expect("NSC witness commitment should match");
    verify_nsc_pc_witness_commitment::<E>(
      &ck,
      &result.folded.pc_instance,
      &result.folded.pc_witness,
    )
    .expect("NSC_PC witness commitment should match");
  }

  #[test]
  fn test_single_fold() {
    test_single_fold_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
  }

  /// Test two sequential folds
  fn test_two_sequential_folds_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = E::Scalar::ZERO;
    let num_cons = 32usize;

    // Setup
    let (shape, ck) = generate_test_circuit::<E, S>(num_cons);
    let str = Structure::new(&shape);
    let S_pc = PowerCheckStructure::from_main(&str);

    // Initialize with zeros
    let (mut nsc_instance, mut nsc_witness, mut abc) = setup_nsc::<E>(&str);
    let (mut nsc_pc_instance, mut nsc_pc_witness) = setup_nsc_pc::<E>(&ck, &S_pc);
    let (mut zc_pc_instance, mut zc_pc_witness) = setup_zc_pc::<E>(&ck, &S_pc);

    // Generate two different witnesses
    let (u1, w1) = generate_satisfying_witness::<E>(&shape, &ck, 2, num_cons);
    let (u2, w2) = generate_satisfying_witness::<E>(&shape, &ck, 3, num_cons);

    // === FOLD 1 ===
    // Save state before fold for verify
    let nsc_instance_before_fold1 = nsc_instance.clone();
    let nsc_pc_instance_before_fold1 = nsc_pc_instance.clone();
    let zc_pc_instance_before_fold1 = zc_pc_instance.clone();

    let result1 = ZeroFoldNIFS::prove(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &S_pc,
      (&nsc_instance, &nsc_witness),
      (&abc.0, &abc.1, &abc.2),
      (&nsc_pc_instance, &nsc_pc_witness),
      (&u1, &w1),
      (&zc_pc_instance, &zc_pc_witness),
    )
    .expect("Fold 1 should succeed");

    // Verify fold 1 proof
    let (verified_nsc1, verified_pc1, verified_zc_pc1) = result1
      .nifs
      .verify(
        &ro_consts,
        &pp_digest,
        &nsc_instance_before_fold1,
        &nsc_pc_instance_before_fold1,
        &u1,
        &zc_pc_instance_before_fold1,
      )
      .expect("Fold 1: verify should succeed");
    assert_eq!(verified_nsc1, result1.folded.nsc_instance, "Fold 1: verified NSC should match");
    assert_eq!(verified_pc1, result1.folded.pc_instance, "Fold 1: verified PC should match");
    assert_eq!(verified_zc_pc1, result1.new_zc_pc.0, "Fold 1: verified ZC_PC should match");

    // Verify fold 1
    str
      .is_sat(
        &ck,
        &result1.folded.nsc_instance,
        &result1.folded.nsc_witness,
      )
      .expect("Fold 1: NSC should satisfy is_sat");
    verify_nsc_claim_bruteforce::<E>(
      &str,
      &result1.folded.nsc_instance,
      &result1.folded.nsc_witness,
    )
    .expect("Fold 1: NSC claim should match");
    verify_nsc_pc_claim_bruteforce::<E>(
      &S_pc,
      &result1.folded.pc_instance,
      &result1.folded.pc_witness,
    )
    .expect("Fold 1: NSC_PC claim should match");
    verify_zc_pc_valid::<E>(&S_pc, &result1.new_zc_pc.0, &result1.new_zc_pc.1)
      .expect("Fold 1: New ZC_PC should be valid");

    // Update running state
    nsc_instance = result1.folded.nsc_instance.clone();
    nsc_witness = result1.folded.nsc_witness.clone();
    abc = (
      result1.folded.Az.clone(),
      result1.folded.Bz.clone(),
      result1.folded.Cz.clone(),
    );
    nsc_pc_instance = result1.folded.pc_instance.clone();
    nsc_pc_witness = result1.folded.pc_witness.clone();
    zc_pc_instance = result1.new_zc_pc.0.clone();
    zc_pc_witness = result1.new_zc_pc.1.clone();

    // === FOLD 2 ===
    // Save state before fold for verify
    let nsc_instance_before_fold2 = nsc_instance.clone();
    let nsc_pc_instance_before_fold2 = nsc_pc_instance.clone();
    let zc_pc_instance_before_fold2 = zc_pc_instance.clone();

    let result2 = ZeroFoldNIFS::prove(
      &ck,
      &ro_consts,
      &pp_digest,
      &str,
      &S_pc,
      (&nsc_instance, &nsc_witness),
      (&abc.0, &abc.1, &abc.2),
      (&nsc_pc_instance, &nsc_pc_witness),
      (&u2, &w2),
      (&zc_pc_instance, &zc_pc_witness),
    )
    .expect("Fold 2 should succeed");

    // Verify fold 2 proof
    let (verified_nsc2, verified_pc2, verified_zc_pc2) = result2
      .nifs
      .verify(
        &ro_consts,
        &pp_digest,
        &nsc_instance_before_fold2,
        &nsc_pc_instance_before_fold2,
        &u2,
        &zc_pc_instance_before_fold2,
      )
      .expect("Fold 2: verify should succeed");
    assert_eq!(verified_nsc2, result2.folded.nsc_instance, "Fold 2: verified NSC should match");
    assert_eq!(verified_pc2, result2.folded.pc_instance, "Fold 2: verified PC should match");
    assert_eq!(verified_zc_pc2, result2.new_zc_pc.0, "Fold 2: verified ZC_PC should match");

    // Verify fold 2
    str
      .is_sat(
        &ck,
        &result2.folded.nsc_instance,
        &result2.folded.nsc_witness,
      )
      .expect("Fold 2: NSC should satisfy is_sat");
    verify_nsc_claim_bruteforce::<E>(
      &str,
      &result2.folded.nsc_instance,
      &result2.folded.nsc_witness,
    )
    .expect("Fold 2: NSC claim should match");
    verify_nsc_pc_claim_bruteforce::<E>(
      &S_pc,
      &result2.folded.pc_instance,
      &result2.folded.pc_witness,
    )
    .expect("Fold 2: NSC_PC claim should match");
    verify_zc_pc_valid::<E>(&S_pc, &result2.new_zc_pc.0, &result2.new_zc_pc.1)
      .expect("Fold 2: New ZC_PC should be valid");
  }

  #[test]
  fn test_two_sequential_folds() {
    test_two_sequential_folds_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
  }

  /// Test three-step folding with multiple engines
  fn test_three_step_folding_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();
    let pp_digest = E::Scalar::ZERO;
    let num_cons = 32usize;

    // Setup
    let (shape, ck) = generate_test_circuit::<E, S>(num_cons);
    let str = Structure::new(&shape);
    let S_pc = PowerCheckStructure::from_main(&str);

    // Initialize with zeros
    let (mut nsc_instance, mut nsc_witness, mut abc) = setup_nsc::<E>(&str);
    let (mut nsc_pc_instance, mut nsc_pc_witness) = setup_nsc_pc::<E>(&ck, &S_pc);
    let (mut zc_pc_instance, mut zc_pc_witness) = setup_zc_pc::<E>(&ck, &S_pc);

    // Generate three different witnesses
    let witnesses: Vec<_> = (2..5)
      .map(|i| generate_satisfying_witness::<E>(&shape, &ck, i, num_cons))
      .collect();

    for (step, (u, w)) in witnesses.iter().enumerate() {
      // Save state before fold for verify
      let nsc_instance_before = nsc_instance.clone();
      let nsc_pc_instance_before = nsc_pc_instance.clone();
      let zc_pc_instance_before = zc_pc_instance.clone();

      let result = ZeroFoldNIFS::prove(
        &ck,
        &ro_consts,
        &pp_digest,
        &str,
        &S_pc,
        (&nsc_instance, &nsc_witness),
        (&abc.0, &abc.1, &abc.2),
        (&nsc_pc_instance, &nsc_pc_witness),
        (u, w),
        (&zc_pc_instance, &zc_pc_witness),
      )
      .unwrap_or_else(|e| panic!("Step {}: ZeroFoldNIFS::prove failed: {:?}", step, e));

      // Verify the proof
      let (verified_nsc, verified_pc, verified_zc_pc) = result
        .nifs
        .verify(
          &ro_consts,
          &pp_digest,
          &nsc_instance_before,
          &nsc_pc_instance_before,
          u,
          &zc_pc_instance_before,
        )
        .unwrap_or_else(|e| panic!("Step {}: ZeroFoldNIFS::verify failed: {:?}", step, e));
      assert_eq!(
        verified_nsc, result.folded.nsc_instance,
        "Step {}: verified NSC should match",
        step
      );
      assert_eq!(
        verified_pc, result.folded.pc_instance,
        "Step {}: verified PC should match",
        step
      );
      assert_eq!(
        verified_zc_pc, result.new_zc_pc.0,
        "Step {}: verified ZC_PC should match",
        step
      );

      // Verify all invariants
      str
        .is_sat(&ck, &result.folded.nsc_instance, &result.folded.nsc_witness)
        .unwrap_or_else(|e| panic!("Step {}: is_sat failed: {:?}", step, e));
      verify_nsc_claim_bruteforce::<E>(
        &str,
        &result.folded.nsc_instance,
        &result.folded.nsc_witness,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC claim failed: {:?}", step, e));
      verify_nsc_pc_claim_bruteforce::<E>(
        &S_pc,
        &result.folded.pc_instance,
        &result.folded.pc_witness,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC_PC claim failed: {:?}", step, e));
      verify_zc_pc_valid::<E>(&S_pc, &result.new_zc_pc.0, &result.new_zc_pc.1)
        .unwrap_or_else(|e| panic!("Step {}: ZC_PC validity failed: {:?}", step, e));

      // Update running state
      nsc_instance = result.folded.nsc_instance;
      nsc_witness = result.folded.nsc_witness;
      abc = (result.folded.Az, result.folded.Bz, result.folded.Cz);
      nsc_pc_instance = result.folded.pc_instance;
      nsc_pc_witness = result.folded.pc_witness;
      zc_pc_instance = result.new_zc_pc.0;
      zc_pc_witness = result.new_zc_pc.1;
    }
  }

  #[test]
  fn test_three_step_folding() {
    test_three_step_folding_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
    test_three_step_folding_with::<PallasEngine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
  }
}
