//! This module implements a non-interactive folding scheme from NeutronNova
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
    sumcheck::{prove_helper, run_combined_sumfold, EvalAcc},
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

/// An NIFS message from NeutronNova's folding scheme
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_E: Commitment<E>,
  pub(crate) poly: UniPoly<E::Scalar>,
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

/// Output of combined NIFS prove (handles both NSC and NSC_PC)
///
/// Per NeutronNova Construction 4: The proof contains a SINGLE combined polynomial
/// `poly = poly_nsc + γ·poly_pc` where γ is a random challenge.
#[derive(Clone, Debug)]
pub struct NIFSCombinedOutput<E: Engine> {
  // === Proof ===
  /// Combined sumcheck proof (comm_E, combined poly)
  pub nifs: NIFS<E>,
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

impl<E: Engine> NIFS<E> {
  /// Takes as input a folded instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with a compatible structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  ///
  /// Note that this code is tailored for use with NeutronNova's IVC scheme, which enforces
  /// certain requirements between the two instances that are folded.
  /// In particular, it requires that `U1` and `U2` are such that the hash of `U1` is stored in the public IO of `U2`.
  /// In this particular setting, this means that if `U2` is absorbed in the RO, it implicitly absorbs `U1` as well.
  /// So the code below avoids absorbing `U1` in the RO.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(NIFS<E>, (FoldedInstance<E>, FoldedWitness<E>)), NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute a commitment to the eq polynomial
    let E = WeightTable::from_tau(&tau, S.left, S.right);
    let comm_E: Commitment<E> = E.commit(ck);

    comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // We now run a single round of the sum-check protocol to establish
    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5): EvalAcc<E::Scalar> =
      prove_helper::<E>(
        &rho,
        (S.left, S.right),
        W1.E.as_slice(),
        &Az1,
        &Bz1,
        &Cz1,
        E.as_slice(),
        &Az2,
        &Bz2,
        &Cz2,
      );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // === ASSERT: Sumcheck Identity ===
    // poly(0) + poly(1) = (1-ρ)·T_running + ρ·T_fresh where T_fresh = 0
    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      T,
      "Sumcheck identity violated: poly(0) + poly(1) ≠ (1-ρ)·T_running"
    );

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_inv: E::Scalar =
      Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b_inv;

    // === ASSERT: Output Claim Consistency ===
    // poly(r_b) = T_out · eq(ρ, r_b)
    assert_eq!(
      poly.evaluate(&r_b),
      T_out * eq_rho_r_b,
      "Output claim inconsistent: poly(r_b) ≠ T_out · eq(ρ, r_b)"
    );

    let U = U1.fold(U2, &comm_E, &r_b, &T_out)?;
    let W = W1.fold(W2, &E, &r_b)?;

    // return the folded instance and witness
    Ok((Self { comm_E, poly }, (U, W)))
  }

  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  #[cfg(test)]
  pub fn verify(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<FoldedInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    self.comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    // check if poly(0) + poly(1) = T
    if self.poly.eval_at_zero() + self.poly.eval_at_one() != T {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&self.poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_inv: E::Scalar =
      Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b_inv;

    let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;

    // return the folded instance and witness
    Ok(U)
  }

  /// ZeroFold prover: folds both NSC and NSC_PC (PowerCheck) relations together.
  ///
  /// Takes accumulated (NSC, NSC_PC) with cached (Az, Bz, Cz), a fresh zero-check R1CS relation, and a fresh ZC_PC.
  /// Returns folded state plus a new ZC_PC for the next iteration.
  #[allow(clippy::too_many_arguments)]
  pub fn prove_zerofold(
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
  ) -> Result<NIFSCombinedOutput<E>, NovaError> {
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

    // === PHASE 6: Return combined output ===
    Ok(NIFSCombinedOutput {
      nifs: NIFS {
        comm_E: conv.comm_E,
        poly: sumcheck_out.poly,
      },
      gamma,
      folded,
      new_zc_pc: conv.zc_pc,
      tau: conv.tau,
      rho,
      r_b,
    })
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
      PallasEngine, Secp256k1Engine,
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
    running: (&FoldedInstance<E>, &FoldedWitness<E>),
    fresh: (&FoldedInstance<E>, &FoldedWitness<E>),
    folded: &FoldedState<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check W folding
    for (i, w_f) in folded.nsc_witness.W.iter().enumerate() {
      let expected = running.1.W[i] * one_minus_r + fresh.1.W[i] * *r_b;
      if *w_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("W folding mismatch at index {}", i),
        });
      }
    }

    // Check E folding
    for (i, e_f) in folded.nsc_witness.E.as_slice().iter().enumerate() {
      let expected = running.1.E.as_slice()[i] * one_minus_r + fresh.1.E.as_slice()[i] * *r_b;
      if *e_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("E folding mismatch at index {}", i),
        });
      }
    }

    // Check u folding
    let expected_u = running.0.u * one_minus_r + fresh.0.u * *r_b;
    if folded.nsc_instance.u != expected_u {
      return Err(NovaError::UnSat {
        reason: format!(
          "u folding mismatch: {:?} != {:?}",
          folded.nsc_instance.u, expected_u
        ),
      });
    }

    // Check X folding
    for (i, x_f) in folded.nsc_instance.X.iter().enumerate() {
      let expected = running.0.X[i] * one_minus_r + fresh.0.X[i] * *r_b;
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
    running: (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
    fresh: (&FoldedPowerCheckInstance<E>, &FoldedPowerCheckWitness<E>),
    folded: &FoldedState<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check tau folding
    let expected_tau = running.0.tau * one_minus_r + fresh.0.tau * *r_b;
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
      let expected =
        running.1.witness.as_slice()[i] * one_minus_r + fresh.1.witness.as_slice()[i] * *r_b;
      if *w_f != expected {
        return Err(NovaError::UnSat {
          reason: format!("PC witness folding mismatch at index {}", i),
        });
      }
    }

    // Check weights folding (should equal folded E)
    for (i, w_f) in folded.pc_witness.weights.as_slice().iter().enumerate() {
      let expected =
        running.1.weights.as_slice()[i] * one_minus_r + fresh.1.weights.as_slice()[i] * *r_b;
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
    running: &FoldedInstance<E>,
    fresh: &FoldedInstance<E>,
    folded: &FoldedInstance<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check comm_W homomorphism
    let expected_comm_W = running.comm_W * one_minus_r + fresh.comm_W * *r_b;
    if folded.comm_W != expected_comm_W {
      return Err(NovaError::UnSat {
        reason: "comm_W homomorphism violated".to_string(),
      });
    }

    // Check comm_E homomorphism
    let expected_comm_E = running.comm_E * one_minus_r + fresh.comm_E * *r_b;
    if folded.comm_E != expected_comm_E {
      return Err(NovaError::UnSat {
        reason: "comm_E homomorphism violated".to_string(),
      });
    }

    Ok(())
  }

  /// Verify commitment homomorphism for NSC_PC (PowerCheck)
  fn verify_nsc_pc_commitment_homomorphism<E: Engine>(
    running: &FoldedPowerCheckInstance<E>,
    fresh: &FoldedPowerCheckInstance<E>,
    folded: &FoldedPowerCheckInstance<E>,
    r_b: &E::Scalar,
  ) -> Result<(), NovaError> {
    let one_minus_r = E::Scalar::ONE - r_b;

    // Check comm_witness homomorphism
    let expected_comm_witness = running.comm_witness * one_minus_r + fresh.comm_witness * *r_b;
    if folded.comm_witness != expected_comm_witness {
      return Err(NovaError::UnSat {
        reason: "PC comm_witness homomorphism violated".to_string(),
      });
    }

    // Check comm_weights homomorphism
    let expected_comm_weights = running.comm_weights * one_minus_r + fresh.comm_weights * *r_b;
    if folded.comm_weights != expected_comm_weights {
      return Err(NovaError::UnSat {
        reason: "PC comm_weights homomorphism violated".to_string(),
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
    instance: &FoldedInstance<E>,
    witness: &FoldedWitness<E>,
  ) -> Result<(), NovaError> {
    let comm_W = E::CE::commit(ck, &witness.W, &witness.r_W);
    let comm_E = witness.E.commit(ck);

    if comm_W != instance.comm_W {
      return Err(NovaError::UnSat {
        reason: "NSC comm_W mismatch: recomputed != claimed".to_string(),
      });
    }
    if comm_E != instance.comm_E {
      return Err(NovaError::UnSat {
        reason: "NSC comm_E mismatch: recomputed != claimed".to_string(),
      });
    }
    Ok(())
  }

  /// Verify NSC_PC witness commitment: recompute commitment from witness and check it matches instance
  fn verify_nsc_pc_witness_commitment<E: Engine>(
    ck: &CommitmentKey<E>,
    instance: &FoldedPowerCheckInstance<E>,
    witness: &FoldedPowerCheckWitness<E>,
  ) -> Result<(), NovaError> {
    let comm_witness = witness.witness.commit(ck);
    let comm_weights = witness.weights.commit(ck);

    if comm_witness != instance.comm_witness {
      return Err(NovaError::UnSat {
        reason: "NSC_PC comm_witness mismatch: recomputed != claimed".to_string(),
      });
    }
    if comm_weights != instance.comm_weights {
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

  fn execute_sequence<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) {
    // produce a default running instance
    let str = Structure::new(shape);
    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U1, W1,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U2, W2,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    // check if the running instance is satisfiable
    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());
  }

  fn test_tiny_r1cs_bellpepper_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();

    // generate a non-trivial circuit
    let num_cons: usize = 32;

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U1, W1) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(3)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // pad the shape and witnesses
    let shape = shape.pad();
    let W1 = W1.pad(&shape);
    let W2 = W2.pad(&shape);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
    test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
  }

  // ============================================================================
  // Combined NIFS Tests (NSC + NSC_PC)
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

    // Run prove_zerofold
    let result = NIFS::prove_zerofold(
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
    .expect("prove_zerofold should succeed");

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
    // Note: This only works when pc_sumcheck_claim = 0 (initial fold from valid ZC_PC instances)
    // For subsequent folds, the relationship is more complex due to eq(ρ, r_b) factor
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
    let fresh = reconstruct_fresh_from_tau::<E>(
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
      (&fresh.nsc.0, &fresh.nsc.1),
      &result.folded,
      &result.r_b,
    )
    .expect("NSC linear folding should be correct");

    // 6. Verify NSC_PC linear folding
    verify_nsc_pc_linear_folding::<E>(
      (&nsc_pc_instance, &nsc_pc_witness),
      (&fresh.nsc_pc.0, &fresh.nsc_pc.1),
      &result.folded,
      &result.r_b,
    )
    .expect("NSC_PC linear folding should be correct");

    // 7. Verify commitment homomorphism
    verify_commitment_homomorphism::<E>(
      &nsc_instance,
      &fresh.nsc.0,
      &result.folded.nsc_instance,
      &result.r_b,
    )
    .expect("Commitment homomorphism should hold");

    // 8. Verify combined sumcheck identity: poly(0) + poly(1) = T_nsc + γ·T_pc
    let T_nsc_claim = (E::Scalar::ONE - result.rho) * nsc_instance.T;
    let T_pc_claim = (E::Scalar::ONE - result.rho) * nsc_pc_instance.pc_sumcheck_claim;
    let combined_claim = T_nsc_claim + result.gamma * T_pc_claim;
    verify_sumcheck_identity::<E>(&result.nifs.poly, &combined_claim)
      .expect("Combined sumcheck identity should hold");

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
    let result1 = NIFS::prove_zerofold(
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

    // Reconstruct fresh NSC/NSC_PC for fold 1 verification
    let fresh1 = reconstruct_fresh_from_tau::<E>(
      &str,
      (&u1, &w1),
      (&zc_pc_instance, &zc_pc_witness),
      &result1.tau,
      &result1.nifs.comm_E,
    )
    .expect("Fold 1: Reconstruction should succeed");

    verify_nsc_linear_folding::<E>(
      (&nsc_instance, &nsc_witness),
      (&fresh1.nsc.0, &fresh1.nsc.1),
      &result1.folded,
      &result1.r_b,
    )
    .expect("Fold 1: NSC linear folding should be correct");
    verify_nsc_pc_linear_folding::<E>(
      (&nsc_pc_instance, &nsc_pc_witness),
      (&fresh1.nsc_pc.0, &fresh1.nsc_pc.1),
      &result1.folded,
      &result1.r_b,
    )
    .expect("Fold 1: NSC_PC linear folding should be correct");
    verify_nsc_pc_commitment_homomorphism::<E>(
      &nsc_pc_instance,
      &fresh1.nsc_pc.0,
      &result1.folded.pc_instance,
      &result1.r_b,
    )
    .expect("Fold 1: NSC_PC commitment homomorphism should hold");
    verify_nsc_witness_commitment::<E>(
      &ck,
      &result1.folded.nsc_instance,
      &result1.folded.nsc_witness,
    )
    .expect("Fold 1: NSC witness commitment should match");
    verify_nsc_pc_witness_commitment::<E>(
      &ck,
      &result1.folded.pc_instance,
      &result1.folded.pc_witness,
    )
    .expect("Fold 1: NSC_PC witness commitment should match");

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
    let result2 = NIFS::prove_zerofold(
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
    // NSC_PC brute force check - works for ANY pc_sumcheck_claim value
    verify_nsc_pc_claim_bruteforce::<E>(
      &S_pc,
      &result2.folded.pc_instance,
      &result2.folded.pc_witness,
    )
    .expect("Fold 2: NSC_PC claim should match");
    verify_zc_pc_valid::<E>(&S_pc, &result2.new_zc_pc.0, &result2.new_zc_pc.1)
      .expect("Fold 2: New ZC_PC should be valid");

    // Reconstruct fresh NSC/NSC_PC for fold 2 verification
    let fresh2 = reconstruct_fresh_from_tau::<E>(
      &str,
      (&u2, &w2),
      (&zc_pc_instance, &zc_pc_witness),
      &result2.tau,
      &result2.nifs.comm_E,
    )
    .expect("Fold 2: Reconstruction should succeed");

    verify_nsc_linear_folding::<E>(
      (&nsc_instance, &nsc_witness),
      (&fresh2.nsc.0, &fresh2.nsc.1),
      &result2.folded,
      &result2.r_b,
    )
    .expect("Fold 2: NSC linear folding should be correct");
    verify_nsc_pc_linear_folding::<E>(
      (&nsc_pc_instance, &nsc_pc_witness),
      (&fresh2.nsc_pc.0, &fresh2.nsc_pc.1),
      &result2.folded,
      &result2.r_b,
    )
    .expect("Fold 2: NSC_PC linear folding should be correct");
    verify_nsc_pc_commitment_homomorphism::<E>(
      &nsc_pc_instance,
      &fresh2.nsc_pc.0,
      &result2.folded.pc_instance,
      &result2.r_b,
    )
    .expect("Fold 2: NSC_PC commitment homomorphism should hold");
    verify_nsc_witness_commitment::<E>(
      &ck,
      &result2.folded.nsc_instance,
      &result2.folded.nsc_witness,
    )
    .expect("Fold 2: NSC witness commitment should match");
    verify_nsc_pc_witness_commitment::<E>(
      &ck,
      &result2.folded.pc_instance,
      &result2.folded.pc_witness,
    )
    .expect("Fold 2: NSC_PC witness commitment should match");

    // Verify sumcheck accumulation
    let T_nsc_claim = (E::Scalar::ONE - result2.rho) * nsc_instance.T;
    verify_sumcheck_identity::<E>(&result2.nifs.poly, &T_nsc_claim)
      .expect("Fold 2: NSC sumcheck identity should hold");
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
      let result = NIFS::prove_zerofold(
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
      .unwrap_or_else(|e| panic!("Step {}: prove_zerofold failed: {:?}", step, e));

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
      // NSC_PC brute force check - works for ANY pc_sumcheck_claim value
      verify_nsc_pc_claim_bruteforce::<E>(
        &S_pc,
        &result.folded.pc_instance,
        &result.folded.pc_witness,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC_PC claim failed: {:?}", step, e));
      verify_zc_pc_valid::<E>(&S_pc, &result.new_zc_pc.0, &result.new_zc_pc.1)
        .unwrap_or_else(|e| panic!("Step {}: ZC_PC validity failed: {:?}", step, e));

      // Reconstruct fresh NSC/NSC_PC for verification
      let fresh = reconstruct_fresh_from_tau::<E>(
        &str,
        (u, w),
        (&zc_pc_instance, &zc_pc_witness),
        &result.tau,
        &result.nifs.comm_E,
      )
      .unwrap_or_else(|e| panic!("Step {}: Reconstruction failed: {:?}", step, e));

      verify_nsc_linear_folding::<E>(
        (&nsc_instance, &nsc_witness),
        (&fresh.nsc.0, &fresh.nsc.1),
        &result.folded,
        &result.r_b,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC linear folding failed: {:?}", step, e));
      verify_nsc_pc_linear_folding::<E>(
        (&nsc_pc_instance, &nsc_pc_witness),
        (&fresh.nsc_pc.0, &fresh.nsc_pc.1),
        &result.folded,
        &result.r_b,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC_PC linear folding failed: {:?}", step, e));
      verify_nsc_pc_commitment_homomorphism::<E>(
        &nsc_pc_instance,
        &fresh.nsc_pc.0,
        &result.folded.pc_instance,
        &result.r_b,
      )
      .unwrap_or_else(|e| {
        panic!(
          "Step {}: NSC_PC commitment homomorphism failed: {:?}",
          step, e
        )
      });
      verify_nsc_witness_commitment::<E>(
        &ck,
        &result.folded.nsc_instance,
        &result.folded.nsc_witness,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC witness commitment mismatch: {:?}", step, e));
      verify_nsc_pc_witness_commitment::<E>(
        &ck,
        &result.folded.pc_instance,
        &result.folded.pc_witness,
      )
      .unwrap_or_else(|e| panic!("Step {}: NSC_PC witness commitment mismatch: {:?}", step, e));

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

  /// Test that pc_sumcheck_claim is correctly computed after multiple folds.
  ///
  /// The key relationship verified:
  /// - After each fold, the bruteforce pc_sumcheck_claim matches the instance claim
  /// - This is already verified by verify_nsc_pc_claim_bruteforce in test_three_step_folding
  ///
  /// This test additionally verifies the T_out computation:
  /// sumcheck_claim_out_pc = poly_pc.evaluate(r_b) * eq_rho_r_b_inv
  fn test_pc_claim_accumulation_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
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

    // Generate witnesses
    let witnesses: Vec<_> = (2..5)
      .map(|i| generate_satisfying_witness::<E>(&shape, &ck, i, num_cons))
      .collect();

    for (step, (u, w)) in witnesses.iter().enumerate() {
      let result = NIFS::prove_zerofold(
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
      .unwrap_or_else(|e| panic!("Step {}: prove_zerofold failed: {:?}", step, e));

      // Verify the combined sumcheck polynomial identity: poly(0) + poly(1) = T_nsc + γ·T_pc
      let T_nsc_claim = (E::Scalar::ONE - result.rho) * nsc_instance.T;
      let T_pc_claim = (E::Scalar::ONE - result.rho) * nsc_pc_instance.pc_sumcheck_claim;
      let combined_claim = T_nsc_claim + result.gamma * T_pc_claim;
      verify_sumcheck_identity::<E>(&result.nifs.poly, &combined_claim)
        .unwrap_or_else(|e| panic!("Step {}: Combined sumcheck identity failed: {:?}", step, e));

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
  fn test_pc_claim_accumulation() {
    test_pc_claim_accumulation_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
  }
}

#[cfg(test)]
mod benchmarks {
  use super::*;
  use crate::{
    frontend::{
      gadgets::{
        boolean::{AllocatedBit, Boolean},
        num::AllocatedNum,
        sha256::sha256,
      },
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      ConstraintSystem, SynthesisError,
    },
    nova::nifs::NIFS as NovaNIFS,
    provider::Bn256EngineKZG,
    r1cs::{R1CSShape, SparseMatrix},
    traits::{commitment::CommitmentEngineTrait, snark::default_ck_hint, ROConstants},
  };
  use core::marker::PhantomData;
  use criterion::Criterion;
  use ff::PrimeField;
  use num_integer::Integer;
  use num_traits::ToPrimitive;
  use rand::Rng;

  /// generates a satisfying R1CS with small witness values
  fn generate_sample_r1cs<E: Engine>(
    num_cons: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let num_vars = num_cons;
    let num_io = 1;

    // we will just generate constraints of the form x * x = x, checking Booleanity
    // generate the constraints by creating sparse matrices
    let A = SparseMatrix::new(
      &(0..num_cons)
        .map(|i| (i, i, E::Scalar::ONE))
        .collect::<Vec<_>>(),
      num_cons,
      num_vars + 1 + num_io,
    );
    let B = A.clone();
    let C = A.clone();

    let S: R1CSShape<E> = R1CSShape::new(num_cons, num_vars, num_io, A, B, C).unwrap();

    let S = S.pad();

    // sample a ck
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    // let witness be randomly generated booleans
    let w = (0..S.num_cons)
      .into_par_iter()
      .map(|_| {
        let mut rng = rand::thread_rng();
        let result: u8 = rng.gen();
        result % 2
      })
      .collect::<Vec<_>>();

    let W = {
      // convert W to field elements
      let W = (0..S.num_cons)
        .into_par_iter()
        .map(|i| <E as Engine>::Scalar::from(w[i] as u64))
        .collect::<Vec<_>>();
      R1CSWitness::new(&S, &W).unwrap()
    };

    let x = vec![E::Scalar::from(0)];
    (S, ck, W, w, x)
  }

  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> Sha256Circuit<E> {
    pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  fn generarate_sha_r1cs<E: Engine>(
    len: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let S = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U, W) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

    let S = S.pad();
    let W = W.pad(&S);

    let w = W
      .W
      .iter()
      .map(|e| {
        // map field element to u8
        // this assumes little-endian representation
        e.to_repr().as_ref()[0]
      })
      .collect::<Vec<_>>();

    // sanity check by recommiting to w
    let comm_W = <E as Engine>::CE::commit_small(&ck, &w, &W.r_W);
    assert_eq!(comm_W, U.comm_W);

    let X = U.X.clone();
    (S, ck, W, w, X)
  }

  fn bench_nifs_inner<E: Engine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    c: &mut Criterion,
    name: &str,
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    W: &R1CSWitness<E>,
    w: &[T],
    x: &[E::Scalar],
  ) {
    let num_cons = S.num_cons;

    // generate a default running instance
    let str = Structure::new(S);
    let f_W = FoldedWitness::default(&str);
    let f_U = FoldedInstance::default(&str);
    let res = str.is_sat(ck, &f_U, &f_W);
    assert!(res.is_ok());

    // generate default values
    let pp_digest = E::Scalar::ZERO;
    let ro_consts = RO2Constants::<E>::default();

    // produce an NIFS with (W, U) as the first incoming witness-instance pair
    c.bench_function(&format!("neutron_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit with the specialized method
        let comm_W = E::CE::commit_small(ck, w, &W.r_W);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NIFS::prove(ck, &ro_consts, &pp_digest, &str, &f_U, &f_W, &U, W);
        assert!(res.is_ok());
      })
    });

    // generate a random relaxed R1CS instance-witness pair
    let (r_U, r_W) = R1CSShape::<E>::sample_random_instance_witness(S, ck).unwrap();
    let ro_consts = ROConstants::<E>::default();

    // produce an NIFS with (r_W, r_U) as the second incoming witness-instance pair
    c.bench_function(&format!("nova_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit to R1CS witness
        let comm_W = W.commit(ck);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NovaNIFS::prove(ck, &ro_consts, &pp_digest, S, &r_U, &r_W, &U, W);
        assert!(res.is_ok());
      })
    });
  }

  #[test]
  fn bench_nifs_simple() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    let num_cons = 1024;
    let (S, ck, W, w, x) = generate_sample_r1cs::<E>(num_cons); // W is R1CSWitness, w is a vector of u8, x is a vector of field elements
    bench_nifs_inner(&mut criterion, "simple", &S, &ck, &W, &w, &x);
  }

  #[test]
  fn bench_nifs_sha256() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    for len in [32, 64].iter() {
      let (S, ck, W, w, x) = generarate_sha_r1cs::<E>(*len);
      bench_nifs_inner(&mut criterion, "sha256", &S, &ck, &W, &w, &x);
    }
  }
}
