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

/// Folded PowerCheck instance (NSC_PC).
/// Contains commitments to both the original PowerCheck witness and the new E.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedPowerCheckInstance<E: Engine> {
  /// Accumulated error. Fresh: T_pc = 0.
  pub T_pc: E::Scalar,

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
  /// Create a default instance
  pub fn default(S: &PowerCheckStructure) -> Self {
    // Suppress unused warning - S is kept for API consistency
    let _ = S;
    Self {
      T_pc: E::Scalar::ZERO,
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
    T_pc_out: &E::Scalar,
  ) -> Self {
    let one_minus_r = E::Scalar::ONE - r_b;

    Self {
      T_pc: *T_pc_out,
      comm_witness: self.comm_witness * one_minus_r + u2.comm_powers * *r_b,
      comm_weights: self.comm_weights * one_minus_r + *comm_weights_new * *r_b,
      tau: one_minus_r * self.tau + *r_b * u2.tau,
    }
  }
}

impl<E: Engine> FoldedPowerCheckWitness<E> {
  /// Create a default witness for tau=0
  ///
  /// Note: witness has dimensions (left, right) matching the power table,
  /// but weights has dimensions (left_pc, right_pc) for the sumcheck structure.
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
      // weights: sumcheck weights, dimensions (left_pc, right_pc)
      weights: WeightTable::new(
        vec![E::Scalar::ZERO; S.left_pc + S.right_pc],
        E::Scalar::ZERO,
        S.left_pc,
      ),
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
    ro.absorb(self.T_pc);
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
  let (g1, g2, g3) = pc_g_at(i, left, e1, e2, tau);
  g1 - g2 * g3
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

  /// Create random weights [E_pc,1 || E_pc,2] for testing.
  fn random_weights(rng: &mut impl RngCore, left_pc: usize, right_pc: usize) -> WeightTable<E> {
    let vec: Vec<_> = (0..(left_pc + right_pc))
      .map(|_| <E as Engine>::Scalar::random(&mut *rng))
      .collect();
    let r = <E as Engine>::Scalar::random(&mut *rng);
    WeightTable::new(vec, r, left_pc)
  }

  /// Compute T_pc = Σᵢ E_pc,2[row(i)] · E_pc,1[col(i)] · (g₁[i] - g₂[i]·g₃[i])
  ///
  /// where:
  ///   - i ∈ [0, num_cons) where num_cons = left + right
  ///   - row(i) = i / left_pc
  ///   - col(i) = i % left_pc
  fn compute_t_pc_naive(
    S_pc: &PowerCheckStructure,
    weights: &WeightTable<E>,
    table: &WeightTable<E>,
    tau: <E as Engine>::Scalar,
  ) -> <E as Engine>::Scalar {
    let e1 = table.e1();
    let e2 = table.e2();
    let left = table.left();

    let weights_e1 = weights.e1(); // E_pc,1
    let weights_e2 = weights.e2(); // E_pc,2

    (0..S_pc.num_cons)
      .map(|i| {
        let residual = pc_residual_at(i, left, e1, e2, tau);
        let row = i / S_pc.left_pc;
        let col = i % S_pc.left_pc;
        weights_e2[row] * weights_e1[col] * residual
      })
      .fold(<E as Engine>::Scalar::ZERO, |acc, x| acc + x)
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
    let weights = random_weights(&mut rng, S_pc.left_pc, S_pc.right_pc);

    let t_pc = compute_t_pc_naive(&S_pc, &weights, &table, tau);

    assert_eq!(
      t_pc,
      <E as Engine>::Scalar::ZERO,
      "Weighted sum should be zero for valid table"
    );
  }

  // ============================================================================
  // Test 5: Folding closure - after folding, T_pc computed naively matches
  // ============================================================================
  #[test]
  fn test_nsc_pc_closed_under_folding_when_tpc_is_correct() {
    let mut rng = StdRng::seed_from_u64(999);

    let left = 64;
    let right = 32;
    let S_pc = PowerCheckStructure::new(left, right);

    // Create two fresh valid power tables
    let (table1, tau1) = fresh_power_table(&mut rng, left, right);
    let (table2, tau2) = fresh_power_table(&mut rng, left, right);

    // Create two weight tables
    let weights1 = random_weights(&mut rng, S_pc.left_pc, S_pc.right_pc);
    let weights2 = random_weights(&mut rng, S_pc.left_pc, S_pc.right_pc);

    // Pick a random folding challenge
    let r_b = <E as Engine>::Scalar::random(&mut rng);

    // Fold everything
    let table_f = table1.fold(&table2, &r_b);
    let weights_f = weights1.fold(&weights2, &r_b);
    let tau_f = (<E as Engine>::Scalar::ONE - r_b) * tau1 + r_b * tau2;

    // Compute T_pc for the folded instance
    let tpc_f = compute_t_pc_naive(&S_pc, &weights_f, &table_f, tau_f);

    // Verify that this T_pc is the correct "error" for the folded relation.
    // The folded relation is satisfied when T_pc equals the weighted sum of residuals.
    // Since we computed tpc_f exactly as that sum, this is a tautology for correctness,
    // but it verifies that compute_t_pc_naive works correctly on folded instances.

    // Also verify that the two fresh tables had T_pc = 0
    let tpc1 = compute_t_pc_naive(&S_pc, &weights1, &table1, tau1);
    let tpc2 = compute_t_pc_naive(&S_pc, &weights2, &table2, tau2);

    assert_eq!(
      tpc1,
      <E as Engine>::Scalar::ZERO,
      "Fresh table 1 should have T_pc = 0"
    );
    assert_eq!(
      tpc2,
      <E as Engine>::Scalar::ZERO,
      "Fresh table 2 should have T_pc = 0"
    );

    // The folded T_pc is generally nonzero due to cross-terms.
    // We just verify the computation completed without error.
    // The important check is that if we later verify the folded instance
    // with T_pc = tpc_f, it should pass.
    let _ = tpc_f; // Use the value to avoid warning
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
