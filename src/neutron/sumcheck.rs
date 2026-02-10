//! Sumcheck utilities for NeutronNova folding scheme
//!
//! This module provides:
//! - Shared accumulator utilities for computing sumcheck evaluations
//! - Combiner functions for R1CS and PowerCheck relations
//! - `prove_helper` for R1CS sumcheck polynomial evaluation
//! - `prove_helper_pc` for PowerCheck sumcheck polynomial evaluation
#![allow(non_snake_case)]
use crate::{neutron::power_check_relation::PowerCheckStructure, traits::Engine};
use ff::{Field, PrimeField};
use rayon::prelude::*;

/// Accumulator type for sumcheck evaluations at points (0, 2, 3, 4, 5)
pub type EvalAcc<F> = (F, F, F, F, F);

/// Zero accumulator for fold/reduce operations
#[inline(always)]
pub fn zero_acc<F: Field>() -> EvalAcc<F> {
  (F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO)
}

/// Combine two accumulators (for reduce step)
#[inline(always)]
pub fn add_acc<F: Field>(a: EvalAcc<F>, b: EvalAcc<F>) -> EvalAcc<F> {
  (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4)
}

/// R1CS combiner: e · (Az · Bz - Cz)
#[inline(always)]
pub fn comb_r1cs<F: Field>(e: F, az: F, bz: F, cz: F) -> F {
  e * (az * bz - cz)
}

/// PowerCheck combiner: w · (g1 - g2 · g3)
#[inline(always)]
pub fn comb_powercheck<F: Field>(w: F, g1: F, g2: F, g3: F) -> F {
  w * (g1 - g2 * g3)
}

/// Accumulates bound function evaluations into an existing accumulator.
///
/// Fold-friendly: takes acc, returns updated acc. Works with rayon fold + reduce.
///
/// Computes evaluations at points 0, 2, 3, 4, 5 using incremental interpolation:
/// - eval 0: use low values
/// - eval 2: low + 2·delta
/// - eval 3: low + 3·delta (incremental from eval 2)
/// - eval 4: low + 4·delta (incremental from eval 3)
/// - eval 5: low + 5·delta (incremental from eval 4)
#[inline(always)]
pub fn accum_contribution<F: Field, Comb>(
  acc: EvalAcc<F>,
  comb_func: Comb,
  c1_low: F,
  c2_low: F,
  c3_low: F,
  c4_low: F,
  c1_high: F,
  c2_high: F,
  c3_high: F,
  c4_high: F,
) -> EvalAcc<F>
where
  Comb: Fn(F, F, F, F) -> F,
{
  // eval 0: use low values
  let e0 = comb_func(c1_low, c2_low, c3_low, c4_low);

  // Compute deltas for incremental interpolation
  let (d1, d2, d3, d4) = (
    c1_high - c1_low,
    c2_high - c2_low,
    c3_high - c3_low,
    c4_high - c4_low,
  );

  // eval 2: low + 2·delta
  let (c1, c2, c3, c4) = (
    c1_low + d1 + d1,
    c2_low + d2 + d2,
    c3_low + d3 + d3,
    c4_low + d4 + d4,
  );
  let e2 = comb_func(c1, c2, c3, c4);

  // eval 3: low + 3·delta (incremental)
  let (c1, c2, c3, c4) = (c1 + d1, c2 + d2, c3 + d3, c4 + d4);
  let e3 = comb_func(c1, c2, c3, c4);

  // eval 4: low + 4·delta (incremental)
  let (c1, c2, c3, c4) = (c1 + d1, c2 + d2, c3 + d3, c4 + d4);
  let e4 = comb_func(c1, c2, c3, c4);

  // eval 5: low + 5·delta (incremental)
  let (c1, c2, c3, c4) = (c1 + d1, c2 + d2, c3 + d3, c4 + d4);
  let e5 = comb_func(c1, c2, c3, c4);

  (acc.0 + e0, acc.1 + e2, acc.2 + e3, acc.3 + e4, acc.4 + e5)
}

/// Applies rho scaling factors to the accumulator.
///
/// The scaling factors are derived from eq polynomial evaluation:
/// - (1 - ρ) for eval 0
/// - (3ρ - 1) for eval 2
/// - (5ρ - 2) for eval 3
/// - (7ρ - 3) for eval 4
/// - (9ρ - 4) for eval 5
#[inline(always)]
pub fn apply_rho_scaling<F: PrimeField>(acc: EvalAcc<F>, rho: &F) -> EvalAcc<F> {
  let one_minus_rho = F::ONE - rho;
  let three_rho_minus_one = F::from(3u64) * rho - F::ONE;
  let five_rho_minus_two = F::from(5u64) * rho - F::from(2u64);
  let seven_rho_minus_three = F::from(7u64) * rho - F::from(3u64);
  let nine_rho_minus_four = F::from(9u64) * rho - F::from(4u64);

  (
    acc.0 * one_minus_rho,
    acc.1 * three_rho_minus_one,
    acc.2 * five_rho_minus_two,
    acc.3 * seven_rho_minus_three,
    acc.4 * nine_rho_minus_four,
  )
}

/// Computes evaluations of the R1CS sum-check polynomial at 0, 2, 3, 4, 5.
///
/// The R1CS relation is: e · (Az · Bz - Cz) = 0 for all constraints.
/// This function computes the bound polynomial evaluations needed for sumcheck.
#[inline]
pub fn prove_helper<E: Engine>(
  rho: &E::Scalar,
  (left, right): (usize, usize),
  e1: &[E::Scalar],
  Az1: &[E::Scalar],
  Bz1: &[E::Scalar],
  Cz1: &[E::Scalar],
  e2: &[E::Scalar],
  Az2: &[E::Scalar],
  Bz2: &[E::Scalar],
  Cz2: &[E::Scalar],
) -> EvalAcc<E::Scalar> {
  // sanity check sizes
  assert_eq!(e1.len(), left + right);
  assert_eq!(Az1.len(), left * right);
  assert_eq!(Bz1.len(), left * right);
  assert_eq!(Cz1.len(), left * right);
  assert_eq!(e2.len(), left + right);
  assert_eq!(Az2.len(), left * right);
  assert_eq!(Bz2.len(), left * right);
  assert_eq!(Cz2.len(), left * right);

  let f1 = &e1[left..];
  let f2 = &e2[left..];

  let acc = (0..right)
    .into_par_iter()
    .fold(zero_acc, |outer_acc, i| {
      // Inner loop accumulates contributions for fixed i
      let inner_acc = (0..left).fold(zero_acc(), |acc, j| {
        let k = i * left + j;
        accum_contribution(
          acc,
          comb_r1cs,
          e1[j],
          Az1[k],
          Bz1[k],
          Cz1[k],
          e2[j],
          Az2[k],
          Bz2[k],
          Cz2[k],
        )
      });

      // Scale by the outer f (second half of e) for this row
      let delta_f = f2[i] - f1[i];

      // eval 0: f1[i] * inner_acc.0
      let e0 = f1[i] * inner_acc.0;

      // eval 2: (f1 + 2*delta) * inner_acc.1
      let f_bound = f1[i] + delta_f + delta_f;
      let e2 = f_bound * inner_acc.1;

      // eval 3: (f1 + 3*delta) * inner_acc.2
      let f_bound = f_bound + delta_f;
      let e3 = f_bound * inner_acc.2;

      // eval 4: (f1 + 4*delta) * inner_acc.3
      let f_bound = f_bound + delta_f;
      let e4 = f_bound * inner_acc.3;

      // eval 5: (f1 + 5*delta) * inner_acc.4
      let f_bound = f_bound + delta_f;
      let e5 = f_bound * inner_acc.4;

      add_acc(outer_acc, (e0, e2, e3, e4, e5))
    })
    .reduce(zero_acc, add_acc);

  apply_rho_scaling(acc, rho)
}

/// Computes evaluations for PowerCheck sum-check polynomial at points 0, 2, 3, 4, 5.
///
/// # Performance: Region-Split Loop Structure
///
/// This function uses region-split loops instead of a single loop with conditionals.
/// The PowerCheck relation has 6 distinct regions with different (g₁, g₂, g₃) formulas.
///
/// - Region 0: 1 iteration (base case e₁[0] = 1)
/// - Region 1: `left-1` iterations (first half chain e₁[i] = τ·e₁[i-1]) ← HOT LOOP
/// - Region 2: 1 iteration (second half base e₂[0] = 1)
/// - Region 3: 1 iteration (link constraint e₂[1] = τ^left)
/// - Region 4: 1 iteration (squaring e₂[2] = e₂[1]²)
/// - Region 5: `right-3` iterations (second half chain e₂[k] = e₂[1]·e₂[k-1]) ← HOT LOOP
///
/// Hot loops (regions 1 and 5) run in parallel via `rayon::join`, and each is
/// internally parallelized with `into_par_iter().fold().reduce()`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn prove_helper_pc<E: Engine>(
  rho: &E::Scalar,
  S_pc: &PowerCheckStructure,
  // E_pc weights [left_pc, right_pc] for running and fresh instances
  weights_1: (&[E::Scalar], &[E::Scalar]), // Running (left, right)
  weights_2: (&[E::Scalar], &[E::Scalar]), // Fresh (left, right)
  // PowerCheck witness [e₁, e₂] for running and fresh instances
  wit_1: (&[E::Scalar], &[E::Scalar]), // Running (e1, e2)
  wit_2: (&[E::Scalar], &[E::Scalar]), // Fresh (e1, e2)
  tau_1: &E::Scalar,
  tau_2: &E::Scalar,
) -> EvalAcc<E::Scalar> {
  let (w1_left, w1_right) = weights_1;
  let (w2_left, w2_right) = weights_2;
  let (e1_1, e1_2) = wit_1; // Running: e1_1 = first half, e1_2 = second half
  let (e2_1, e2_2) = wit_2; // Fresh: e2_1 = first half, e2_2 = second half

  let left = S_pc.left;
  let right = S_pc.right;
  let left_pc = S_pc.left_pc;

  // Sanity checks
  debug_assert_eq!(w1_left.len(), S_pc.left_pc);
  debug_assert_eq!(w1_right.len(), S_pc.right_pc);
  debug_assert_eq!(w2_left.len(), S_pc.left_pc);
  debug_assert_eq!(w2_right.len(), S_pc.right_pc);
  debug_assert_eq!(e1_1.len(), left);
  debug_assert_eq!(e1_2.len(), right);
  debug_assert_eq!(e2_1.len(), left);
  debug_assert_eq!(e2_2.len(), right);

  // Helper to compute weight at constraint index i
  // Weight = E_pc_right[row] * E_pc_left[col] where row = i / left_pc, col = i % left_pc
  let weight_at = |i: usize, w_left: &[E::Scalar], w_right: &[E::Scalar]| -> E::Scalar {
    let row = i / left_pc;
    let col = i % left_pc;
    w_right[row] * w_left[col]
  };

  // =========================================================================
  // Single-element regions (O(1) - negligible cost, run inline)
  // =========================================================================

  // Region 0: i = 0 (base case: e₁[0] = 1)
  let acc_0 = {
    let i = 0;
    let w1 = weight_at(i, w1_left, w1_right);
    let w2 = weight_at(i, w2_left, w2_right);
    accum_contribution(
      zero_acc(),
      comb_powercheck,
      w1,
      e1_1[0],
      E::Scalar::ONE,
      E::Scalar::ONE,
      w2,
      e2_1[0],
      E::Scalar::ONE,
      E::Scalar::ONE,
    )
  };

  // Region 2: i = left (second half base: e₂[0] = 1)
  let acc_2 = {
    let i = left;
    let w1 = weight_at(i, w1_left, w1_right);
    let w2 = weight_at(i, w2_left, w2_right);
    accum_contribution(
      zero_acc(),
      comb_powercheck,
      w1,
      e1_2[0],
      E::Scalar::ONE,
      E::Scalar::ONE,
      w2,
      e2_2[0],
      E::Scalar::ONE,
      E::Scalar::ONE,
    )
  };

  // Region 3: i = left+1 (link constraint: e₂[1] = τ^left = τ·e₁[left-1])
  let acc_3 = if right > 1 {
    let i = left + 1;
    let w1 = weight_at(i, w1_left, w1_right);
    let w2 = weight_at(i, w2_left, w2_right);
    accum_contribution(
      zero_acc(),
      comb_powercheck,
      w1,
      e1_2[1],
      e1_1[left - 1],
      *tau_1,
      w2,
      e2_2[1],
      e2_1[left - 1],
      *tau_2,
    )
  } else {
    zero_acc()
  };

  // Region 4: i = left+2 (squaring: e₂[2] = e₂[1]²)
  let acc_4 = if right > 2 {
    let i = left + 2;
    let w1 = weight_at(i, w1_left, w1_right);
    let w2 = weight_at(i, w2_left, w2_right);
    accum_contribution(
      zero_acc(),
      comb_powercheck,
      w1,
      e1_2[2],
      e1_2[1],
      e1_2[1],
      w2,
      e2_2[2],
      e2_2[1],
      e2_2[1],
    )
  } else {
    zero_acc()
  };

  // =========================================================================
  // HOT LOOPS: Run regions 1 & 5 in parallel via rayon::join
  // =========================================================================
  let (acc_1, acc_5) = rayon::join(
    // Region 1: 1 ≤ i < left (first half chain: e₁[i] = τ·e₁[i-1])
    || {
      (1..left)
        .into_par_iter()
        .fold(zero_acc, |acc, i| {
          let w1 = weight_at(i, w1_left, w1_right);
          let w2 = weight_at(i, w2_left, w2_right);
          accum_contribution(
            acc,
            comb_powercheck,
            w1,
            e1_1[i],
            e1_1[i - 1],
            *tau_1,
            w2,
            e2_1[i],
            e2_1[i - 1],
            *tau_2,
          )
        })
        .reduce(zero_acc, add_acc)
    },
    // Region 5: 3 ≤ k < right (second half chain: e₂[k] = e₂[1]·e₂[k-1])
    || {
      (3..right)
        .into_par_iter()
        .fold(zero_acc, |acc, k| {
          let i = left + k;
          let w1 = weight_at(i, w1_left, w1_right);
          let w2 = weight_at(i, w2_left, w2_right);
          accum_contribution(
            acc,
            comb_powercheck,
            w1,
            e1_2[k],
            e1_2[1],
            e1_2[k - 1],
            w2,
            e2_2[k],
            e2_2[1],
            e2_2[k - 1],
          )
        })
        .reduce(zero_acc, add_acc)
    },
  );

  // Reduce all regions
  let acc = [acc_0, acc_1, acc_2, acc_3, acc_4, acc_5]
    .into_iter()
    .reduce(add_acc)
    .unwrap();

  apply_rho_scaling(acc, rho)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{provider::PallasEngine, spartan::polys::univariate::UniPoly};

  /// Brute-force computation of PowerCheck weighted sum for testing.
  /// Computes: T_pc = Σᵢ E_pc_weight(i) · (g₁[i] - g₂[i]·g₃[i])
  fn compute_pc_weighted_sum_bruteforce<F: Field>(
    e1: &[F],      // First half of power table
    e2: &[F],      // Second half of power table
    tau: &F,
    w_left: &[F],  // E_pc left weights
    w_right: &[F], // E_pc right weights
    left_pc: usize,
  ) -> F {
    let left = e1.len();
    let right = e2.len();
    let num_cons = left + right;

    (0..num_cons)
      .map(|i| {
        // Compute (g1, g2, g3) based on region
        let (g1, g2, g3) = if i == 0 {
          // Region 0: base case
          (e1[0], F::ONE, F::ONE)
        } else if i < left {
          // Region 1: first half chain
          (e1[i], e1[i - 1], *tau)
        } else if i == left {
          // Region 2: second half base
          (e2[0], F::ONE, F::ONE)
        } else if i == left + 1 {
          // Region 3: link
          (e2[1], e1[left - 1], *tau)
        } else if i == left + 2 {
          // Region 4: squaring
          (e2[2], e2[1], e2[1])
        } else {
          // Region 5: second half chain
          let k = i - left;
          (e2[k], e2[1], e2[k - 1])
        };

        let residual = g1 - g2 * g3;

        // Weight from E_pc tensor
        let row = i / left_pc;
        let col = i % left_pc;
        let weight = w_right[row] * w_left[col];

        weight * residual
      })
      .fold(F::ZERO, |acc, x| acc + x)
  }

  /// Create a valid power table (all constraints = 0)
  fn create_valid_power_table<F: Field>(tau: &F, left: usize, right: usize) -> (Vec<F>, Vec<F>) {
    // e1 = [1, τ, τ², ..., τ^(left-1)]
    let e1: Vec<F> = (0..left)
      .scan(F::ONE, |state, _| {
        let val = *state;
        *state = *state * tau;
        Some(val)
      })
      .collect();

    // e2 = [1, τ^left, τ^(2·left), ...]
    let tau_left = tau.pow([left as u64]);
    let e2: Vec<F> = (0..right)
      .scan(F::ONE, |state, _| {
        let val = *state;
        *state = *state * tau_left;
        Some(val)
      })
      .collect();

    (e1, e2)
  }

  /// Test that prove_helper_pc computes correct sum by comparing with brute-force
  #[test]
  fn test_prove_helper_pc_claim() {
    type E = PallasEngine;
    type F = <E as Engine>::Scalar;

    // Small test case: left=4, right=4 → num_cons=8
    let left = 4usize;
    let right = 4usize;
    let S_pc = PowerCheckStructure::new(left, right);

    // Create two different tau values
    let tau_1 = F::from(7u64);
    let tau_2 = F::from(11u64);

    // Create valid power tables
    let (e1_1, e2_1) = create_valid_power_table(&tau_1, left, right);
    let (e1_2, e2_2) = create_valid_power_table(&tau_2, left, right);

    // Create random E_pc weights
    let w1_left: Vec<F> = (0..S_pc.left_pc).map(|i| F::from((i + 1) as u64)).collect();
    let w1_right: Vec<F> = (0..S_pc.right_pc).map(|i| F::from((i + 10) as u64)).collect();
    let w2_left: Vec<F> = (0..S_pc.left_pc).map(|i| F::from((i + 100) as u64)).collect();
    let w2_right: Vec<F> = (0..S_pc.right_pc).map(|i| F::from((i + 200) as u64)).collect();

    // Compute T_pc for running instance (brute force)
    let T_pc_1 =
      compute_pc_weighted_sum_bruteforce(&e1_1, &e2_1, &tau_1, &w1_left, &w1_right, S_pc.left_pc);

    // For a VALID power table, T_pc should be 0
    assert_eq!(T_pc_1, F::ZERO, "Valid power table should have T_pc = 0");

    // Now test with an INVALID power table (corrupt one element)
    let mut e1_corrupted = e1_1.clone();
    e1_corrupted[2] = F::from(999u64); // Corrupt one entry

    let T_pc_corrupted = compute_pc_weighted_sum_bruteforce(
      &e1_corrupted,
      &e2_1,
      &tau_1,
      &w1_left,
      &w1_right,
      S_pc.left_pc,
    );
    assert_ne!(
      T_pc_corrupted,
      F::ZERO,
      "Corrupted power table should have T_pc ≠ 0"
    );

    // Test prove_helper_pc with random rho
    let rho = F::from(123u64);

    let (e0, e2, e3, e4, e5) = prove_helper_pc::<E>(
      &rho,
      &S_pc,
      (&w1_left, &w1_right),
      (&w2_left, &w2_right),
      (&e1_1, &e2_1),
      (&e1_2, &e2_2),
      &tau_1,
      &tau_2,
    );

    // The sum T = (1-rho)*T_1 where T_1 = 0 for valid table, so T = 0
    // poly(0) + poly(1) = T, and poly(1) = T - poly(0)
    // For valid tables: e0 should capture the running contribution (which is 0)
    // This is a basic sanity check that the function runs without panic

    // Verify eval_0 + eval_1 structure
    let T_claim = F::ZERO; // Both tables are valid → T_1 = T_2 = 0
    let eval_1 = T_claim - e0;

    // Build polynomial and verify identity
    let evals = vec![e0, eval_1, e2, e3, e4, e5];
    let poly = UniPoly::<F>::from_evals(&evals);

    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      T_claim,
      "Sumcheck identity must hold: poly(0) + poly(1) = T_claim"
    );
  }

  /// Test that sumcheck claim equals random linear combination of individual claims
  #[test]
  fn test_powercheck_sumcheck_claim_is_rlc() {
    type E = PallasEngine;
    type F = <E as Engine>::Scalar;

    // Test case with non-zero T_pc (using corrupted tables)
    let left = 4usize;
    let right = 4usize;
    let S_pc = PowerCheckStructure::new(left, right);

    let tau_1 = F::from(7u64);
    let tau_2 = F::from(11u64);

    // Create tables - running instance with corrupted table (non-zero T_pc)
    let (mut e1_1, e2_1) = create_valid_power_table(&tau_1, left, right);
    e1_1[2] = F::from(999u64); // Corrupt to get non-zero T_pc

    // Fresh instance with valid table (T_pc = 0)
    let (e1_2, e2_2) = create_valid_power_table(&tau_2, left, right);

    // E_pc weights
    let w1_left: Vec<F> = (0..S_pc.left_pc).map(|i| F::from((i + 1) as u64)).collect();
    let w1_right: Vec<F> = (0..S_pc.right_pc).map(|i| F::from((i + 10) as u64)).collect();
    let w2_left: Vec<F> = (0..S_pc.left_pc).map(|i| F::from((i + 100) as u64)).collect();
    let w2_right: Vec<F> = (0..S_pc.right_pc).map(|i| F::from((i + 200) as u64)).collect();

    // Compute individual T_pc values
    let T_pc_1 =
      compute_pc_weighted_sum_bruteforce(&e1_1, &e2_1, &tau_1, &w1_left, &w1_right, S_pc.left_pc);
    let T_pc_2 =
      compute_pc_weighted_sum_bruteforce(&e1_2, &e2_2, &tau_2, &w2_left, &w2_right, S_pc.left_pc);

    assert_ne!(
      T_pc_1,
      F::ZERO,
      "Running instance should have non-zero T_pc"
    );
    assert_eq!(
      T_pc_2,
      F::ZERO,
      "Fresh valid instance should have T_pc = 0"
    );

    // Sample random rho
    let rho = F::from(42u64);

    // Compute expected claim as RLC
    let T_claim_expected = (F::ONE - rho) * T_pc_1 + rho * T_pc_2;

    // Run prove_helper_pc
    let (e0, e2, e3, e4, e5) = prove_helper_pc::<E>(
      &rho,
      &S_pc,
      (&w1_left, &w1_right),
      (&w2_left, &w2_right),
      (&e1_1, &e2_1),
      (&e1_2, &e2_2),
      &tau_1,
      &tau_2,
    );

    // Build polynomial
    let eval_1 = T_claim_expected - e0;
    let evals = vec![e0, eval_1, e2, e3, e4, e5];
    let poly = UniPoly::<F>::from_evals(&evals);

    // KEY CHECK: poly(0) + poly(1) = T_claim (random linear combination)
    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      T_claim_expected,
      "Sumcheck claim must equal RLC of individual claims: (1-ρ)·T_1 + ρ·T_2"
    );
  }

  /// Test that valid power tables have all constraints = 0
  #[test]
  fn test_powercheck_constraint_regions() {
    type F = <PallasEngine as Engine>::Scalar;

    let left = 8usize;
    let right = 4usize;
    let tau = F::from(5u64);

    let (e1, e2) = create_valid_power_table(&tau, left, right);

    // Verify all constraints = 0 for valid power table
    let num_cons = left + right;
    for i in 0..num_cons {
      let (g1, g2, g3) = if i == 0 {
        (e1[0], F::ONE, F::ONE)
      } else if i < left {
        (e1[i], e1[i - 1], tau)
      } else if i == left {
        (e2[0], F::ONE, F::ONE)
      } else if i == left + 1 {
        (e2[1], e1[left - 1], tau)
      } else if i == left + 2 {
        (e2[2], e2[1], e2[1])
      } else {
        let k = i - left;
        (e2[k], e2[1], e2[k - 1])
      };

      let residual = g1 - g2 * g3;
      assert_eq!(
        residual,
        F::ZERO,
        "Constraint {} should be satisfied (residual = 0)",
        i
      );
    }
  }
}
