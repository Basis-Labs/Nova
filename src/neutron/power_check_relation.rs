//! This module defines relations used in the PowerCheck folding scheme
use crate::{
  neutron::{relation::Structure, weight_table::WeightTable},
  traits::{AbsorbInRO2Trait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use serde::{Deserialize, Serialize};

/// Metadata for PowerCheck relation (Construction 2).
/// Verifies e = [e₁ || e₂] contains valid powers of τ via F_PC(g₁,g₂,g₃) = g₁ - g₂·g₃ = 0
///
/// # Sumcheck Domain
///
/// The PowerCheck sumcheck uses the SAME domain as the main NSC sumcheck (left × right).
/// This allows using a single weight table E for both sumchecks, which is essential for
/// soundness: E is checked via ZC_PC, and using the same E for NSC_PC ensures the
/// PowerCheck constraints are properly weighted.
///
/// The actual number of PowerCheck constraints is only `left + right`, so indices
/// `i >= num_cons` are skipped in `prove_helper_pc` (they contribute zero).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerCheckStructure {
  /// Length of e₁. When ℓ is odd, left > right.
  /// Example: ℓ=11 → left=64, right=32
  pub left: usize,

  /// Length of e₂. Invariant: left × right = N (main relation constraint count).
  /// Example: ℓ=10 → left=32, right=32 (even case)
  pub right: usize,

  /// Actual number of PowerCheck constraints = left + right.
  /// NOT necessarily a power of two (e.g., 96 for ℓ=11).
  /// Indices >= num_cons are padding and skipped in prove_helper_pc.
  pub num_cons: usize,
}

impl PowerCheckStructure {
  /// Create a new PowerCheckStructure from split dimensions.
  ///
  /// For odd ℓ, left > right (e.g., ℓ=11 → left=64, right=32 → num_cons=96).
  /// The sumcheck uses the main domain (left × right), with indices >= num_cons skipped.
  pub fn new(left: usize, right: usize) -> Self {
    let num_cons = left + right;
    Self {
      left,
      right,
      num_cons,
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
  pub comm_powers: Commitment<E>,

  /// Scalar whose powers are in powers: powers₁[j] = τ^j, powers₂[i] = τ^{i·left}
  pub tau: E::Scalar,
}

/// Fresh PowerCheck witness (ZC_PC).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PowerCheckWitness<E: Engine> {
  /// Power table, shared with FoldedWitness.E via Arc
  pub powers: WeightTable<E>,
}

/// Create a fresh ZC_PC (zero-check PowerCheck) instance and witness.
///
/// This is the entry point for creating a PowerCheck that verifies E is a valid power table.
/// The returned instance/witness can then be converted to NSC_PC form via
/// `FoldedPowerCheckInstance::from_fresh_zc_pc` and `FoldedPowerCheckWitness::from_fresh_zc_pc`.
pub fn fresh_power_check<E: Engine>(
  tau: &E::Scalar,
  left: usize,
  right: usize,
  ck: &CommitmentKey<E>,
) -> (PowerCheckInstance<E>, PowerCheckWitness<E>) {
  let witness = PowerCheckWitness {
    powers: WeightTable::from_tau(tau, left, right),
  };
  let instance = PowerCheckInstance {
    comm_powers: witness.powers.commit(ck),
    tau: *tau,
  };
  (instance, witness)
}

/// Folded PowerCheck instance (NSC_PC).
/// Contains commitments to both the original PowerCheck witness and the new E.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedPowerCheckInstance<E: Engine> {
  /// Accumulated error. Fresh: pc_sumcheck_claim = 0.
  pub pc_sumcheck_claim: E::Scalar,

  /// Commitment to accumulated PowerCheck witness (powers of original tau)
  pub comm_witness: Commitment<E>,

  /// Commitment to weights (E vector, shared with main relation)
  pub comm_weights: Commitment<E>,

  /// Original tau (linearly folded: τ_fold = (1-r_b)·τ_old + r_b·τ_new)
  pub tau: E::Scalar,
}

/// Folded PowerCheck witness (NSC_PC).
/// Contains both the accumulated PowerCheck witness and the weights (E vector).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedPowerCheckWitness<E: Engine> {
  /// Accumulated PowerCheck witness (powers of original tau)
  pub witness: WeightTable<E>,

  /// Weights (E vector, powers of fresh tau), shared with main relation
  pub weights: WeightTable<E>,
}

impl<E: Engine> FoldedPowerCheckInstance<E> {
  /// Create a fresh NSC_PC instance from a ZC_PC instance.
  ///
  /// This converts a PowerCheck instance into the folded form for the NSC_PC sumcheck.
  /// The `comm_E` parameter MUST be the same commitment used for the main NSC sumcheck,
  /// ensuring both sumchecks use the same (checked) weight table.
  ///
  /// Fresh instances have pc_sumcheck_claim = 0 since the PowerCheck is exactly satisfied.
  pub fn from_fresh_zc_pc(zc_pc: &PowerCheckInstance<E>, comm_E: Commitment<E>) -> Self {
    Self {
      pc_sumcheck_claim: E::Scalar::ZERO,
      comm_witness: zc_pc.comm_powers,
      comm_weights: comm_E,
      tau: zc_pc.tau,
    }
  }

  /// Create an instance from a witness with proper commitments
  pub fn from_witness(
    ck: &CommitmentKey<E>,
    w: &FoldedPowerCheckWitness<E>,
  ) -> Self {
    Self {
      pc_sumcheck_claim: E::Scalar::ZERO,
      comm_witness: w.witness.commit(ck),
      comm_weights: w.weights.commit(ck),
      tau: E::Scalar::ZERO,
    }
  }

  /// Create a default instance (zero commitments, only valid if witness is also all zeros with zero randomness)
  pub fn default(S: &PowerCheckStructure) -> Self {
    // Suppress unused warning - S is kept for API consistency
    let _ = S;
    Self {
      pc_sumcheck_claim: E::Scalar::ZERO,
      comm_witness: Commitment::<E>::default(),
      comm_weights: Commitment::<E>::default(),
      tau: E::Scalar::ZERO,
    }
  }

  /// Fold the instance with a fresh instance
  pub fn fold(
    &self,
    u2: &PowerCheckInstance<E>,
    r_b: &E::Scalar,
    comm_weights_new: &Commitment<E>,
    pc_sumcheck_claim_out: &E::Scalar,
  ) -> Self {
    let one_minus_r = E::Scalar::ONE - r_b;

    Self {
      pc_sumcheck_claim: *pc_sumcheck_claim_out,
      comm_witness: self.comm_witness * one_minus_r + u2.comm_powers * *r_b,
      comm_weights: self.comm_weights * one_minus_r + *comm_weights_new * *r_b,
      tau: one_minus_r * self.tau + *r_b * u2.tau,
    }
  }
}

impl<E: Engine> FoldedPowerCheckWitness<E> {
  /// Create a default witness for tau=0
  ///
  /// Both witness and weights have dimensions (left, right) matching the main domain.
  /// Using the same dimensions allows sharing a single weight table E for both
  /// the main NSC sumcheck and the NSC_PC sumcheck.
  ///
  /// The witness is a valid power table for tau=0: [1, 0, 0, ...] || [1, 0, 0, ...]
  /// This satisfies all PowerCheck constraints since e[i] = e[i-1]·τ = e[i-1]·0 = 0.
  pub fn default(S: &PowerCheckStructure) -> Self {
    // witness: power table for tau=0, with e1[0]=1 and e2[0]=1
    let mut witness_vec = vec![E::Scalar::ZERO; S.left + S.right];
    witness_vec[0] = E::Scalar::ONE; // e₁[0] = 1 (base case)
    witness_vec[S.left] = E::Scalar::ONE; // e₂[0] = 1 (base case)

    Self {
      witness: WeightTable::new(witness_vec, E::Scalar::ZERO, S.left),
      // weights: same dimensions as main E (left, right), NOT (left_pc, right_pc)
      weights: WeightTable::new(
        vec![E::Scalar::ZERO; S.left + S.right],
        E::Scalar::ZERO,
        S.left,
      ),
    }
  }

  /// Create a fresh NSC_PC witness from a ZC_PC witness.
  ///
  /// This converts a PowerCheck witness into the folded form for the NSC_PC sumcheck.
  /// The `E` parameter MUST be the same weight table used for the main NSC sumcheck,
  /// ensuring both sumchecks use the same (checked) weight table.
  pub fn from_fresh_zc_pc(zc_pc: &PowerCheckWitness<E>, E: WeightTable<E>) -> Self {
    Self {
      witness: zc_pc.powers.clone(),
      weights: E,
    }
  }

  /// Fold the witness with a fresh witness
  pub fn fold(&self, w2: &PowerCheckWitness<E>, weights_new: &WeightTable<E>, r_b: &E::Scalar) -> Self {
    Self {
      witness: self.witness.fold(&w2.powers, r_b),
      weights: self.weights.fold(weights_new, r_b),
    }
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for PowerCheckInstance<E> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    self.comm_powers.absorb_in_ro2(ro);
    ro.absorb(self.tau);
  }
}

impl<E: Engine> AbsorbInRO2Trait<E> for FoldedPowerCheckInstance<E> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    ro.absorb(self.pc_sumcheck_claim);
    self.comm_witness.absorb_in_ro2(ro);
    self.comm_weights.absorb_in_ro2(ro);
    ro.absorb(self.tau);
  }
}

/// Compute (g₁, g₂, g₃) at index i per Construction 2 selection pattern.
///
/// The selection pattern verifies that e = [e₁ || e₂] contains valid powers of τ:
/// - e₁ = [1, τ, τ², ..., τ^{left-1}]
/// - e₂ = [1, τ^left, τ^{2·left}, ..., τ^{(right-1)·left}]
///
/// Each constraint checks: g₁[i] = g₂[i] · g₃[i]
///
/// | Index Range              | g₁         | g₂         | g₃         | Meaning                    |
/// |--------------------------|------------|------------|------------|----------------------------|
/// | i = 0                    | e[0]       | 1          | 1          | e[0] = 1 (base case)       |
/// | 1 ≤ i < left             | e[i]       | e[i-1]     | τ          | e[i] = e[i-1]·τ (chain)    |
/// | i = left                 | e[left]    | 1          | 1          | e[left] = 1 (2nd base)     |
/// | i = left+1               | e[left+1]  | e[left-1]  | τ          | e[left+1] = τ^{left-1}·τ   |
/// | i = left+2               | e[left+2]  | e[left+1]  | e[left+1]  | e[left+2] = (τ^left)²      |
/// | left+3 ≤ i < left+right  | e[i]       | e[left+1]  | e[i-1]     | e[i] = τ^left · e[i-1]     |
#[inline(always)]
pub fn pc_g_at<F: Field>(
  i: usize,
  left: usize,
  e1: &[F], // table[0..left]
  e2: &[F], // table[left..]
  tau: F,
) -> (F, F, F) {
  debug_assert!(
    i < left + e2.len(),
    "PC constraint index {i} out of bounds (num_cons = {})",
    left + e2.len()
  );

  // g₁ is always e[i]
  let g1 = if i < left { e1[i] } else { e2[i - left] };

  // g₂, g₃ depend on region (selection pattern)
  let (g2, g3) = if i == 0 {
    // Region 0: base case e[0] = 1
    (F::ONE, F::ONE)
  } else if i < left {
    // Region 1: first half chain e[i] = e[i-1]·τ
    (e1[i - 1], tau)
  } else if i == left {
    // Region 2: second half base e[left] = 1
    (F::ONE, F::ONE)
  } else if i == left + 1 {
    // Region 3: link constraint e[left+1] = e[left-1]·τ = τ^left
    (e1[left - 1], tau)
  } else if i == left + 2 {
    // Region 4: squaring e[left+2] = e[left+1]² = τ^{2·left}
    (e2[1], e2[1])
  } else {
    // Region 5: second half chain e[i] = τ^left · e[i-1]
    (e2[1], e2[i - left - 1])
  };

  (g1, g2, g3)
}

/// Compute residual F_PC = g₁ - g₂·g₃ at index i.
///
/// For a valid power table, all residuals are zero.
/// For a corrupted table, at least one residual will be non-zero.
#[inline(always)]
pub fn pc_residual_at<F: Field>(i: usize, left: usize, e1: &[F], e2: &[F], tau: F) -> F {
  debug_assert!(
    i < left + e2.len(),
    "PC constraint index {i} out of bounds (num_cons = {})",
    left + e2.len()
  );
  let (g1, g2, g3) = pc_g_at(i, left, e1, e2, tau);
  g1 - g2 * g3
}

/// Brute-force computation of PowerCheck weighted sum for testing.
/// Computes: pc_sumcheck_claim = Σᵢ w_right[row] · w_left[col] · (g₁[i] - g₂[i]·g₃[i])
///
/// Uses main domain dimensions (left, right) for weights, matching the main NSC sumcheck.
/// Indices >= num_cons are skipped (they contribute zero since F_PC = 0 for padding).
///
/// This is a test utility that can be imported by other test modules.
#[cfg(test)]
pub(crate) fn compute_pc_weighted_sum_bruteforce<F: Field>(
  e1: &[F],      // First half of power table (length = left)
  e2: &[F],      // Second half of power table (length = right)
  tau: &F,
  w_left: &[F],  // E left weights (length = left, same as main E)
  w_right: &[F], // E right weights (length = right, same as main E)
) -> F {
  let left = e1.len();
  let right = e2.len();
  let num_cons = left + right;

  // Iterate over main domain (left × right), skip padding (i >= num_cons)
  let mut sum = F::ZERO;
  for row in 0..right {
    for col in 0..left {
      let i = row * left + col;
      if i >= num_cons {
        continue; // Skip padding
      }
      let (g1, g2, g3) = pc_g_at(i, left, e1, e2, *tau);
      let residual = g1 - g2 * g3;
      sum += w_right[row] * w_left[col] * residual;
    }
  }
  sum
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{provider::Bn256EngineKZG, r1cs::R1CSShape, spartan::polys::power::PowPolynomial};
  use ff::Field;
  use rand::{rngs::StdRng, RngCore, SeedableRng};

  type E = Bn256EngineKZG;

  /// Create a fresh valid power table [e₁ || e₂] with random tau.
  /// Returns (table, tau) where table satisfies all PC constraints.
  fn fresh_power_table(
    rng: &mut impl RngCore,
    left: usize,
    right: usize,
  ) -> (WeightTable<E>, <E as Engine>::Scalar) {
    let tau = <E as Engine>::Scalar::random(&mut *rng);
    let ell = ((left * right) as u32).ilog2() as usize;
    let vec = PowPolynomial::new(&tau, ell).split_evals(left, right);
    let r = <E as Engine>::Scalar::random(&mut *rng);
    (WeightTable::new(vec, r, left), tau)
  }

  /// Create random weights with main domain dimensions (left, right).
  /// This matches the dimensions of the main E weight table.
  fn random_weights(rng: &mut impl RngCore, left: usize, right: usize) -> WeightTable<E> {
    let vec: Vec<_> = (0..(left + right))
      .map(|_| <E as Engine>::Scalar::random(&mut *rng))
      .collect();
    let r = <E as Engine>::Scalar::random(&mut *rng);
    WeightTable::new(vec, r, left)
  }

  /// Compute pc_sumcheck_claim using WeightTable inputs.
  /// Delegates to the shared `compute_pc_weighted_sum_bruteforce` function.
  fn compute_power_check_sumcheck_claim_naive(
    _S_pc: &PowerCheckStructure,
    weights: &WeightTable<E>,
    table: &WeightTable<E>,
    tau: <E as Engine>::Scalar,
  ) -> <E as Engine>::Scalar {
    compute_pc_weighted_sum_bruteforce(
      table.e1(),
      table.e2(),
      &tau,
      weights.e1(),
      weights.e2(),
    )
  }

  // ============================================================================
  // Test 1: Fresh power table ⇒ all residuals are zero
  // ============================================================================
  #[test]
  fn test_pc_residuals_zero_for_fresh_table() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    // Test both even and odd ell cases
    let test_cases = [
      (32, 32), // ell=10, even case
      (64, 32), // ell=11, odd case
    ];

    for (left, right) in test_cases {
      let (table, tau) = fresh_power_table(&mut rng, left, right);
      let e1 = table.e1();
      let e2 = table.e2();

      for i in 0..(left + right) {
        let residual = pc_residual_at(i, left, e1, e2, tau);
        assert_eq!(
          residual,
          <E as Engine>::Scalar::ZERO,
          "Residual nonzero at i={} for left={}, right={}",
          i,
          left,
          right
        );
      }
    }
  }

  // ============================================================================
  // Test 2: Corrupt one entry ⇒ detect nonzero residual
  // ============================================================================
  #[test]
  fn test_pc_detects_corruption() {
    let mut rng = StdRng::seed_from_u64(123);

    let left = 64;
    let right = 32;
    let tau = <E as Engine>::Scalar::random(&mut rng);
    let ell = ((left * right) as u32).ilog2() as usize;

    // Create a valid table then corrupt it
    let mut vec = PowPolynomial::new(&tau, ell).split_evals(left, right);
    vec[10] += <E as Engine>::Scalar::ONE; // corrupt entry 10
    let table = WeightTable::<E>::new(vec, <E as Engine>::Scalar::ZERO, left);
    let e1 = table.e1();
    let e2 = table.e2();

    // At least one residual should be nonzero
    let any_nonzero = (0..(left + right)).any(|i| {
      let residual = pc_residual_at(i, left, e1, e2, tau);
      residual != <E as Engine>::Scalar::ZERO
    });

    assert!(any_nonzero, "Corruption was not detected (unexpected)");
  }

  // ============================================================================
  // Test 3: Boundary cases - test each region's boundary indices
  // ============================================================================
  #[test]
  fn test_pc_residuals_boundary_cases() {
    let mut rng = StdRng::seed_from_u64(456);

    let left = 64;
    let right = 32;
    let (table, tau) = fresh_power_table(&mut rng, left, right);
    let e1 = table.e1();
    let e2 = table.e2();

    // Test boundary indices for each region
    let boundary_indices = [
      0,                  // Region 0: base case
      1,                  // Region 1: first element of chain
      left - 1,           // Region 1: last element of first half
      left,               // Region 2: second half base
      left + 1,           // Region 3: link constraint
      left + 2,           // Region 4: squaring
      left + 3,           // Region 5: first element of second half chain
      left + right - 1,   // Region 5: last element
    ];

    for &i in &boundary_indices {
      let residual = pc_residual_at(i, left, e1, e2, tau);
      assert_eq!(
        residual,
        <E as Engine>::Scalar::ZERO,
        "Residual nonzero at boundary index i={}",
        i
      );
    }
  }

  // ============================================================================
  // Test 4: NSC_PC weighted sum is zero for valid table
  // ============================================================================
  #[test]
  fn test_nsc_pc_weighted_sum_zero_for_valid_table() {
    let mut rng = StdRng::seed_from_u64(789);

    let left = 64;
    let right = 32;
    let S_pc = PowerCheckStructure::new(left, right);

    let (table, tau) = fresh_power_table(&mut rng, left, right);
    // Use main domain dimensions (left, right) for weights
    let weights = random_weights(&mut rng, left, right);

    let pc_claim = compute_power_check_sumcheck_claim_naive(&S_pc, &weights, &table, tau);

    assert_eq!(
      pc_claim,
      <E as Engine>::Scalar::ZERO,
      "Weighted sum should be zero for valid table"
    );
  }

  // ============================================================================
  // Test 5: Folding closure - after folding, pc_sumcheck_claim computed naively matches
  // ============================================================================
  #[test]
  fn test_nsc_pc_closed_under_folding_when_pc_claim_is_correct() {
    let mut rng = StdRng::seed_from_u64(999);

    let left = 64;
    let right = 32;
    let S_pc = PowerCheckStructure::new(left, right);

    // Create two fresh valid power tables
    let (table1, tau1) = fresh_power_table(&mut rng, left, right);
    let (table2, tau2) = fresh_power_table(&mut rng, left, right);

    // Create two weight tables with main domain dimensions
    let weights1 = random_weights(&mut rng, left, right);
    let weights2 = random_weights(&mut rng, left, right);

    // Pick a random folding challenge
    let r_b = <E as Engine>::Scalar::random(&mut rng);

    // Verify that fresh tables have pc_sumcheck_claim = 0
    let pc_claim_1 = compute_power_check_sumcheck_claim_naive(&S_pc, &weights1, &table1, tau1);
    let pc_claim_2 = compute_power_check_sumcheck_claim_naive(&S_pc, &weights2, &table2, tau2);

    assert_eq!(
      pc_claim_1,
      <E as Engine>::Scalar::ZERO,
      "Fresh table 1 should have pc_sumcheck_claim = 0"
    );
    assert_eq!(
      pc_claim_2,
      <E as Engine>::Scalar::ZERO,
      "Fresh table 2 should have pc_sumcheck_claim = 0"
    );

    // Fold everything
    let folded_table = table1.fold(&table2, &r_b);
    let folded_weights = weights1.fold(&weights2, &r_b);
    let folded_tau = (<E as Engine>::Scalar::ONE - r_b) * tau1 + r_b * tau2;

    // Compute pc_sumcheck_claim for the folded instance
    let folded_pc_claim = compute_power_check_sumcheck_claim_naive(&S_pc, &folded_weights, &folded_table, folded_tau);

    // A. Folding two DIFFERENT valid tables produces non-zero pc_sumcheck_claim (cross-terms exist)
    assert_ne!(
      folded_pc_claim,
      <E as Engine>::Scalar::ZERO,
      "Folding different valid tables should produce non-zero pc_sumcheck_claim due to cross-terms"
    );

    // B. Folding a table with ITSELF should preserve pc_sumcheck_claim = 0 (no cross-terms)
    let self_folded_table = table1.fold(&table1, &r_b);
    let self_folded_weights = weights1.fold(&weights1, &r_b);
    let self_folded_tau = tau1; // (1-r_b)*tau1 + r_b*tau1 = tau1
    let self_folded_pc_claim =
      compute_power_check_sumcheck_claim_naive(&S_pc, &self_folded_weights, &self_folded_table, self_folded_tau);
    assert_eq!(
      self_folded_pc_claim,
      <E as Engine>::Scalar::ZERO,
      "Self-fold should preserve pc_sumcheck_claim = 0 (no cross-terms)"
    );
  }

  // ============================================================================
  // Test 6: Commitment homomorphism matches folding
  // ============================================================================
  #[test]
  fn test_commitment_homomorphism() {
    use crate::r1cs::SparseMatrix;
    use crate::traits::snark::default_ck_hint;

    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let left = 64;
    let right = 32;
    let table_size = left + right;

    // Create a minimal R1CS shape with enough variables for the weight tables
    // We need at least table_size variables (columns = num_vars + 1 + num_io)
    let num_vars = table_size;
    let num_cons = 1;
    let num_io = 1;
    let one = <E as Engine>::Scalar::ONE;

    // Minimal constraint: x * 1 = x (trivially satisfiable)
    let A = SparseMatrix::new(&[(0, 0, one)], num_cons, num_vars + 1 + num_io);
    let B = SparseMatrix::new(&[(0, num_vars, one)], num_cons, num_vars + 1 + num_io); // u column
    let C = SparseMatrix::new(&[(0, 0, one)], num_cons, num_vars + 1 + num_io);

    let shape: R1CSShape<E> = R1CSShape::new(num_cons, num_vars, num_io, A, B, C).unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

    // Create two tables
    let (table1, _tau1) = fresh_power_table(&mut rng, left, right);
    let (table2, _tau2) = fresh_power_table(&mut rng, left, right);

    // Pick a random folding challenge
    let r_b = <E as Engine>::Scalar::random(&mut rng);

    // Fold
    let table_f = table1.fold(&table2, &r_b);

    // Commit to each table
    let comm1 = table1.commit(&ck);
    let comm2 = table2.commit(&ck);
    let comm_f = table_f.commit(&ck);

    // Check homomorphism: Commit(fold) = (1-r_b)*Commit(1) + r_b*Commit(2)
    let expected = comm1 * (<E as Engine>::Scalar::ONE - r_b) + comm2 * r_b;

    assert_eq!(
      comm_f, expected,
      "Commitment homomorphism violated: Commit(fold) ≠ (1-r_b)·Commit(1) + r_b·Commit(2)"
    );
  }
}
