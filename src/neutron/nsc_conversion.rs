//! Construction 3: Convert zero-check to nested sum-check
#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  neutron::{
    power_check_relation::{
      FoldedPowerCheckInstance, FoldedPowerCheckWitness, PowerCheckInstance, PowerCheckStructure,
      PowerCheckWitness,
    },
    relation::Structure,
    weight_table::WeightTable,
  },
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::polys::power::PowPolynomial,
  traits::{AbsorbInRO2Trait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rand_core::OsRng;

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
  _S_pc: &PowerCheckStructure,
  zc: (&R1CSInstance<E>, &R1CSWitness<E>),
  zc_pc: (&PowerCheckInstance<E>, &PowerCheckWitness<E>),
  transcript: &mut E::RO2,
) -> Result<NSCConversionOutput<E>, NovaError> {
  // Step 1: Absorb fresh R1CS instance into transcript
  let (U2, W2) = zc;
  U2.absorb_in_ro2(transcript);

  // Step 2: Absorb ZC_PC instance into transcript
  let (pc_inst, pc_wit) = zc_pc;
  pc_inst.absorb_in_ro2(transcript);

  // Step 3: Squeeze τ from transcript
  let tau = transcript.squeeze(NUM_CHALLENGE_BITS, false);

  // Step 4: Create power table E = [e₁ || e₂] from τ
  // e₁ = [1, τ, τ², ..., τ^(left-1)]
  // e₂ = [1, τ^left, τ^(2·left), ...]
  //
  // IMPORTANT: We use a SINGLE weight table E with main domain dimensions
  // for BOTH the main NSC and NSC_PC sumchecks. PowerCheck constraints
  // (num_cons = left + right) are padded to the main domain (left × right).
  // The sumcheck skips indices >= num_cons.
  let E_vec = PowPolynomial::new(&tau, S.ell).split_evals(S.left, S.right);
  let r_E = E::Scalar::random(&mut OsRng);
  let E = WeightTable::new(E_vec, r_E, S.left);

  // Step 5: Commit to E (shared by both NSC and NSC_PC)
  let comm_E: Commitment<E> = E.commit(ck);

  // Step 6: Absorb comm_E into transcript (for downstream challenges)
  comm_E.absorb_in_ro2(transcript);

  // Step 7: Compute (Az, Bz, Cz) for fresh R1CS
  let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
  let (Az, Bz, Cz) = S.S.multiply_vec(&z2)?;

  // Step 8: Convert input ZC_PC to NSC_PC form
  // The NSC_PC uses:
  //   - witness: the original power table being checked (from pc_wit)
  //   - weights: same E table as main NSC (padded domain, skip indices >= num_cons)
  let nsc_pc_instance = FoldedPowerCheckInstance::from_fresh_zc_pc(pc_inst, comm_E);
  let nsc_pc_witness = FoldedPowerCheckWitness::from_fresh_zc_pc(pc_wit, E.clone());

  // Step 9: Create fresh PowerCheck instance/witness (the hanging ZC_PC)
  // This checks that E is a valid power table
  let fresh_pc_instance = PowerCheckInstance {
    comm_powers: comm_E,
    tau,
  };
  let fresh_pc_witness = PowerCheckWitness { powers: E.clone() };

  Ok(NSCConversionOutput {
    tau,
    E,
    comm_E,
    Az,
    Bz,
    Cz,
    nsc_pc: (nsc_pc_instance, nsc_pc_witness),
    fresh_pc: (fresh_pc_instance, fresh_pc_witness),
  })
}
