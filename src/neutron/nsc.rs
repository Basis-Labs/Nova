//! Construction 3: Convert zero-check to nested sum-check
#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  neutron::{
    power_check_relation::{
      fresh_power_check, FoldedPowerCheckInstance, FoldedPowerCheckWitness, PowerCheckInstance,
      PowerCheckWitness,
    },
    relation::{FoldedInstance, FoldedWitness, Structure},
    weight_table::WeightTable,
  },
  r1cs::{R1CSInstance, R1CSWitness},
  traits::{AbsorbInRO2Trait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;

/// Output of Construction 3: Zero-check conversion
///
/// Converts ZC × ZC_PC → NSC × NSC_PC × ZC_PC
pub struct NSCConversionOutput<E: Engine> {
  /// Challenge τ for power polynomial (used to construct E)
  pub tau: E::Scalar,

  /// Shared weight table [e₁ || e₂] for both main NSC and NSC_PC
  /// Uses dimensions (left, right) from main relation; PowerCheck
  /// constraints are padded to this domain (indices >= num_cons skipped)
  pub E: WeightTable<E>,

  /// Commitment to E
  pub comm_E: Commitment<E>,

  /// Matrix-vector product Az for fresh R1CS
  pub Az: Vec<E::Scalar>,
  /// Matrix-vector product Bz for fresh R1CS
  pub Bz: Vec<E::Scalar>,
  /// Matrix-vector product Cz for fresh R1CS
  pub Cz: Vec<E::Scalar>,

  /// Fresh R1CS converted to NSC form
  pub nsc: (FoldedInstance<E>, FoldedWitness<E>),

  /// Input ZC_PC converted to NSC_PC form (attaches E as weights)
  pub nsc_pc: (FoldedPowerCheckInstance<E>, FoldedPowerCheckWitness<E>),

  /// Fresh PowerCheck instance (the "hanging" ZC_PC to be checked later)
  pub fresh_pc: (PowerCheckInstance<E>, PowerCheckWitness<E>),
}

/// Construction 3: Convert zero-check to nested sum-check
///
/// Converts fresh ZC (R1CS) and existing ZC_PC (PowerCheck) instances
/// into NSC and NSC_PC form, plus outputs a new fresh ZC_PC.
///
/// Type signature: ZC × ZC_PC → NSC × NSC_PC × ZC_PC
pub fn convert_to_nsc<E: Engine>(
  ck: &CommitmentKey<E>,
  S: &Structure<E>,
  zc: (&R1CSInstance<E>, &R1CSWitness<E>),
  zc_pc: (&PowerCheckInstance<E>, &PowerCheckWitness<E>),
  transcript: &mut E::RO2,
) -> Result<NSCConversionOutput<E>, NovaError> {
  // Step 1: Absorb fresh R1CS instance into transcript
  let (U2, W2) = zc;
  U2.absorb_in_ro2(transcript);

  // Step 2: Absorb ZC_PC instance into transcript
  let (U_pc, W_pc) = zc_pc;
  U_pc.absorb_in_ro2(transcript);

  // Step 3: Squeeze τ from transcript
  let tau = transcript.squeeze(NUM_CHALLENGE_BITS, false);

  // Step 4: Create fresh ZC_PC for E (the hanging PowerCheck)
  // This creates E = [1, τ, τ², ...] and commits to it.
  // IMPORTANT: E has main domain dimensions (left × right) for BOTH
  // the main NSC and NSC_PC sumchecks. PowerCheck constraints are padded.
  let (fresh_pc_instance, fresh_pc_witness) = fresh_power_check(&tau, S.left, S.right, ck);
  let E = fresh_pc_witness.powers.clone();
  let comm_E: Commitment<E> = fresh_pc_instance.comm_powers;

  // Step 5: Absorb comm_E into transcript (for downstream challenges)
  comm_E.absorb_in_ro2(transcript);

  // Step 6: Compute (Az, Bz, Cz) for fresh R1CS
  let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
  let (Az, Bz, Cz) = S.S.multiply_vec(&z2)?;

  // Step 7: Convert input ZC_PC to NSC_PC form
  // The NSC_PC uses:
  //   - witness: the original power table being checked (from pc_wit)
  //   - weights: same E table as main NSC (padded domain, skip indices >= num_cons)
  let nsc_pc_instance = FoldedPowerCheckInstance::from_fresh_zc_pc(U_pc, comm_E);
  let nsc_pc_witness = FoldedPowerCheckWitness::from_fresh_zc_pc(W_pc, E.clone());

  // Step 8: Create fresh NSC form from R1CS
  let nsc_instance = FoldedInstance {
    comm_W: U2.comm_W,
    comm_E,
    T: E::Scalar::ZERO,
    u: E::Scalar::ONE,
    X: U2.X.clone(),
  };
  let nsc_witness = FoldedWitness::from_r1cs(W2, E.clone());

  Ok(NSCConversionOutput {
    tau,
    E,
    comm_E,
    Az,
    Bz,
    Cz,
    nsc: (nsc_instance, nsc_witness),
    nsc_pc: (nsc_pc_instance, nsc_pc_witness),
    fresh_pc: (fresh_pc_instance, fresh_pc_witness),
  })
}

// ============================================================================
// Test Verification Helpers
// ============================================================================

#[cfg(test)]
use super::power_check_relation::{pc_g_at, PowerCheckStructure};

/// Compute residual F_PC = g₁ - g₂·g₃ at index i.
///
/// For a valid power table, all residuals are zero.
/// For a corrupted table, at least one residual will be non-zero.
#[cfg(test)]
pub(crate) fn pc_residual_at<F: Field>(i: usize, left: usize, e1: &[F], e2: &[F], tau: F) -> F {
  debug_assert!(
    i < left + e2.len(),
    "PC constraint index {i} out of bounds (num_cons = {})",
    left + e2.len()
  );
  let (g1, g2, g3) = pc_g_at(i, left, e1, e2, tau);
  g1 - g2 * g3
}

/// Brute-force verify the NSC sumcheck claim
/// sum = Σᵢ E[i] · (Az[i]·Bz[i] - Cz[i]) should equal U.T
#[cfg(test)]
pub(crate) fn verify_nsc_claim_bruteforce<E: Engine>(
  S: &Structure<E>,
  U: &FoldedInstance<E>,
  W: &FoldedWitness<E>,
) -> Result<(), NovaError> {
  // Compute z = [W, u, X]
  let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
  let (Az, Bz, Cz) = S.S.multiply_vec(&z)?;

  // Compute full E (outer product of E1 and E2)
  let (E1, E2) = (W.E.e1(), W.E.e2());
  let mut full_E = vec![E::Scalar::ZERO; S.left * S.right];
  for i in 0..S.right {
    for j in 0..S.left {
      full_E[i * S.left + j] = E2[i] * E1[j];
    }
  }

  // Compute weighted sum
  let sum: E::Scalar = full_E
    .iter()
    .zip(Az.iter())
    .zip(Bz.iter())
    .zip(Cz.iter())
    .map(|(((e, a), b), c)| *e * (*a * *b - *c))
    .fold(E::Scalar::ZERO, |acc, x| acc + x);

  if sum != U.T {
    return Err(NovaError::UnSat {
      reason: format!(
        "NSC claim mismatch: computed {:?} != claimed {:?}",
        sum, U.T
      ),
    });
  }
  Ok(())
}

/// Brute-force verify the NSC_PC sumcheck claim
/// pc_sumcheck_claim = Σᵢ E_pc[i] · pc_residual_at(i) should equal U.pc_sumcheck_claim
#[cfg(test)]
pub(crate) fn verify_nsc_pc_claim_bruteforce<E: Engine>(
  S_pc: &PowerCheckStructure,
  U: &FoldedPowerCheckInstance<E>,
  W: &FoldedPowerCheckWitness<E>,
) -> Result<(), NovaError> {
  let e1 = W.witness.e1();
  let e2 = W.witness.e2();
  let w_left = W.weights.e1();
  let w_right = W.weights.e2();
  let tau = U.tau;
  let left = S_pc.left; // Main relation's left (also used for weight indexing)

  // pc_sumcheck_claim = Σᵢ weight(i) · residual(i)
  let sum: E::Scalar = (0..S_pc.num_cons)
    .map(|i| {
      let residual = pc_residual_at(i, left, e1, e2, tau);
      let row = i / left;
      let col = i % left;
      w_right[row] * w_left[col] * residual
    })
    .fold(E::Scalar::ZERO, |acc, x| acc + x);

  if sum != U.pc_sumcheck_claim {
    return Err(NovaError::UnSat {
      reason: format!(
        "NSC_PC claim mismatch: computed {:?} != claimed {:?}",
        sum, U.pc_sumcheck_claim
      ),
    });
  }
  Ok(())
}
