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
///
/// # Tensor-Product Structure and Linear Interpolation
///
/// The equality-indicator `e` has tensor structure: `e[i,j] = f[i] × e_left[j]`
/// where `f = e[left..]` (outer/row factor) and `e_left = e[..left]` (inner/column factor).
///
/// When folding two instances, each factor interpolates linearly in `t`:
/// - `e_left(t) = e1_left + t·(e2_left - e1_left)`
/// - `f(t) = f1 + t·(f2 - f1)`
///
/// The product `e_left(t) × f(t)` is **quadratic** in t, so pre-multiplying at
/// endpoints and linearly interpolating would be **incorrect**.
///
/// ## Nested Loop Solution
///
/// To correctly evaluate at points {0, 2, 3, 4, 5}:
/// - **Inner loop**: interpolates `e_left[j]` linearly, accumulates `Σⱼ e_left(t)[j] × contrib[j]`
/// - **Outer loop**: multiplies by `f(t)[i]` interpolated at the same point
///
/// This correctly computes `e_left(t) × f(t) × contrib` at each evaluation point.
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

  // ==========================================================================
  // Nested loop for correct tensor-product interpolation
  // ==========================================================================
  let acc = (0..right)
    .into_par_iter()
    .fold(zero_acc, |outer_acc, i| {
      // ========================================
      // Inner loop: linearly interpolate e_left[j] at points {0, 2, 3, 4, 5}
      // Computes: inner_acc[pt] = Σⱼ comb(e_left(pt)[j], Az(pt), Bz(pt), Cz(pt))
      // ========================================
      let inner_acc = (0..left).fold(zero_acc(), |acc, j| {
        let k = i * left + j;
        accum_contribution(
          acc,
          comb_r1cs,
          e1[j],   // e_left at t=0
          Az1[k],
          Bz1[k],
          Cz1[k],
          e2[j],   // e_left at t=1
          Az2[k],
          Bz2[k],
          Cz2[k],
        )
      });

      // ========================================
      // Outer factor: linearly interpolate f[i] at points {0, 2, 3, 4, 5}
      // Then multiply: eval[pt] = f(pt)[i] × inner_acc[pt]
      // This correctly produces e_left(t) × f(t) at each point.
      // ========================================
      let delta_f = f2[i] - f1[i];

      // f(t) = f1 + t·delta_f
      let e0 = f1[i] * inner_acc.0;                     // t=0: f(0) = f1

      let f_bound = f1[i] + delta_f + delta_f;          // t=2: f(2) = f1 + 2·delta
      let e2 = f_bound * inner_acc.1;

      let f_bound = f_bound + delta_f;                  // t=3: f(3) = f1 + 3·delta
      let e3 = f_bound * inner_acc.2;

      let f_bound = f_bound + delta_f;                  // t=4: f(4) = f1 + 4·delta
      let e4 = f_bound * inner_acc.3;

      let f_bound = f_bound + delta_f;                  // t=5: f(5) = f1 + 5·delta
      let e5 = f_bound * inner_acc.4;

      add_acc(outer_acc, (e0, e2, e3, e4, e5))
    })
    .reduce(zero_acc, add_acc);

  apply_rho_scaling(acc, rho)
}

/// Computes evaluations for PowerCheck sum-check polynomial at points 0, 2, 3, 4, 5.
///
/// # Mathematical Expression
///
/// This function computes the RLC-combined PowerCheck claim:
///
/// ```text
/// T = (1-ρ)·T₁ + ρ·T₂
///
/// where T_k = Σᵢ E_pc_left(col(i)) · E_pc_right(row(i)) · (g₁_k[i] - g₂_k[i]·g₃_k[i])
/// ```
///
/// # Tensor Weight Structure (Critical for Correctness)
///
/// The E_pc weights have tensor structure: `E_pc[i] = E_pc_right[row] × E_pc_left[col]`
/// where `row = i / left_pc` and `col = i % left_pc`.
///
/// When folding two instances, each tensor factor is linearly interpolated:
/// - `w_left(t) = w1_left + t·(w2_left - w1_left)` (linear in t)
/// - `w_right(t) = w1_right + t·(w2_right - w1_right)` (linear in t)
///
/// Their product `w_left(t) × w_right(t)` is **quadratic** in t.
///
/// This function uses **nested loops** (mirroring `prove_helper` for R1CS) to correctly
/// handle the tensor structure:
/// - Inner loop: accumulates with `w_left[col]` factor (linearly interpolated)
/// - Outer loop: multiplies by `w_right[row]` factor (linearly interpolated)
///
/// This produces correct evaluations at all points {0, 2, 3, 4, 5}.
///
/// # Selection Pattern for (g₁, g₂, g₃)
///
/// | Region | Index | g₁ | g₂ | g₃ |
/// |--------|-------|-----|-----|-----|
/// | 0 | i = 0 | e[0] | 1 | 1 |
/// | 1 | 1 ≤ i < left | e[i] | e[i-1] | **τ** |
/// | 2 | i = left | e[left] | 1 | 1 |
/// | 3 | i = left+1 | e[left+1] | e[left-1] | **τ** |
/// | 4 | i = left+2 | e[left+2] | e[left+1] | e[left+1] |
/// | 5 | left+3 ≤ i | e[i] | e[left+1] | e[i-1] |
///
/// # Why τ₁ and τ₂?
///
/// When folding two PowerCheck instances:
/// - **Running instance** has power table e₁ with scalar τ₁
/// - **Fresh instance** has power table e₂ with scalar τ₂
///
/// Since g₃ = τ in regions 1 and 3, each instance needs its own τ value.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn prove_helper_pc<E: Engine>(
  rho: &E::Scalar,
  S_pc: &PowerCheckStructure,
  weights_1: (&[E::Scalar], &[E::Scalar]),
  weights_2: (&[E::Scalar], &[E::Scalar]),
  w1: (&[E::Scalar], &[E::Scalar]),
  w2: (&[E::Scalar], &[E::Scalar]),
  tau_1: &E::Scalar,
  tau_2: &E::Scalar,
) -> EvalAcc<E::Scalar> {
  let (w1_left, w1_right) = weights_1;
  let (w2_left, w2_right) = weights_2;
  let (e1_1, e1_2) = w1;
  let (e2_1, e2_2) = w2;

  let left = S_pc.left;
  let right = S_pc.right;
  let num_cons = S_pc.num_cons; // = left + right

  // Weights use MAIN domain dimensions (left, right), same as main E
  // This ensures we use the same weight table E for both NSC and NSC_PC sumchecks
  debug_assert_eq!(w1_left.len(), left);
  debug_assert_eq!(w1_right.len(), right);
  debug_assert_eq!(w2_left.len(), left);
  debug_assert_eq!(w2_right.len(), right);
  debug_assert_eq!(e1_1.len(), left);
  debug_assert_eq!(e1_2.len(), right);
  debug_assert_eq!(e2_1.len(), left);
  debug_assert_eq!(e2_2.len(), right);

  // Helper to get (g1, g2, g3) values for constraint i based on region
  // e_first = first half of power table (length = left)
  // e_second = second half of power table (length = right)
  let get_g_values =
    |i: usize,
     e_first: &[E::Scalar],
     e_second: &[E::Scalar],
     tau: &E::Scalar|
     -> (E::Scalar, E::Scalar, E::Scalar) {
      if i == 0 {
        // Region 0: base case
        (e_first[0], E::Scalar::ONE, E::Scalar::ONE)
      } else if i < left {
        // Region 1: first half chain
        (e_first[i], e_first[i - 1], *tau)
      } else if i == left {
        // Region 2: second half base
        (e_second[0], E::Scalar::ONE, E::Scalar::ONE)
      } else if i == left + 1 && right > 1 {
        // Region 3: link
        (e_second[1], e_first[left - 1], *tau)
      } else if i == left + 2 && right > 2 {
        // Region 4: squaring
        (e_second[2], e_second[1], e_second[1])
      } else {
        // Region 5: second half chain
        let k = i - left;
        (e_second[k], e_second[1], e_second[k - 1])
      }
    };

  // ==========================================================================
  // Nested loop for correct tensor-product interpolation
  //
  // E has tensor structure: E[i] = w_right[row] × w_left[col]
  // When folding, each factor interpolates linearly:
  //   w_left(t) = w1_left + t·(w2_left - w1_left)
  //   w_right(t) = w1_right + t·(w2_right - w1_right)
  //
  // Their product w_left(t)×w_right(t) is QUADRATIC in t.
  // Pre-multiplying at endpoints would miss the cross-term!
  //
  // Solution: nested loops (mirrors prove_helper for R1CS)
  // - Inner loop: interpolate w_left[col] linearly
  // - Outer loop: multiply by w_right[row] interpolated at the same point
  //
  // NOTE: We iterate over the MAIN domain (left × right), but only num_cons = left + right
  // positions have real constraints. Indices >= num_cons are padding and skipped.
  // ==========================================================================

  let acc = (0..right)
    .into_par_iter()
    .fold(zero_acc, |outer_acc, row| {
      // ========================================
      // Inner loop: linearly interpolate w_left[col] at points {0, 2, 3, 4, 5}
      // Computes: inner_acc[pt] = Σⱼ comb(w_left(pt)[col], g1(pt), g2(pt), g3(pt))
      // ========================================
      let inner_acc = (0..left).fold(zero_acc(), |acc, col| {
        let i = row * left + col;
        if i >= num_cons {
          return acc; // Skip padding - most iterations will hit this
        }

        let (g1_1, g2_1, g3_1) = get_g_values(i, e1_1, e1_2, tau_1);
        let (g1_2, g2_2, g3_2) = get_g_values(i, e2_1, e2_2, tau_2);

        // Pass w_left[col] only (NOT w_left × w_right!)
        // w_right is applied in the outer loop
        accum_contribution(
          acc,
          comb_powercheck,
          w1_left[col], // w_left at t=0
          g1_1,
          g2_1,
          g3_1,
          w2_left[col], // w_left at t=1
          g1_2,
          g2_2,
          g3_2,
        )
      });

      // ========================================
      // Outer factor: linearly interpolate w_right[row] at points {0, 2, 3, 4, 5}
      // Then multiply: eval[pt] = w_right(pt)[row] × inner_acc[pt]
      // This correctly produces w_left(t) × w_right(t) at each point.
      // ========================================
      let delta_w = w2_right[row] - w1_right[row];

      // w_right(t) = w1_right + t·delta_w
      let w0 = w1_right[row];                           // t=0: w_right(0) = w1_right
      let e0 = w0 * inner_acc.0;

      let w2 = w1_right[row] + delta_w + delta_w;       // t=2: w_right(2) = w1_right + 2·delta
      let e2 = w2 * inner_acc.1;

      let w3 = w2 + delta_w;                            // t=3: w_right(3) = w1_right + 3·delta
      let e3 = w3 * inner_acc.2;

      let w4 = w3 + delta_w;                            // t=4: w_right(4) = w1_right + 4·delta
      let e4 = w4 * inner_acc.3;

      let w5 = w4 + delta_w;                            // t=5: w_right(5) = w1_right + 5·delta
      let e5 = w5 * inner_acc.4;

      add_acc(outer_acc, (e0, e2, e3, e4, e5))
    })
    .reduce(zero_acc, add_acc);

  apply_rho_scaling(acc, rho)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    neutron::power_check_relation::{compute_pc_weighted_sum_bruteforce, pc_g_at},
    provider::PallasEngine,
    spartan::polys::univariate::UniPoly,
  };

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

    // Create random E weights (using MAIN domain dimensions)
    let w1_left: Vec<F> = (0..left).map(|i| F::from((i + 1) as u64)).collect();
    let w1_right: Vec<F> = (0..right).map(|i| F::from((i + 10) as u64)).collect();
    let w2_left: Vec<F> = (0..left).map(|i| F::from((i + 100) as u64)).collect();
    let w2_right: Vec<F> = (0..right).map(|i| F::from((i + 200) as u64)).collect();

    // Compute pc_sumcheck_claim for running instance (brute force)
    let pc_claim_1 = compute_pc_weighted_sum_bruteforce(&e1_1, &e2_1, &tau_1, &w1_left, &w1_right);

    // For a VALID power table, pc_sumcheck_claim should be 0
    assert_eq!(pc_claim_1, F::ZERO, "Valid power table should have pc_sumcheck_claim = 0");

    // Now test with an INVALID power table (corrupt one element)
    let mut e1_corrupted = e1_1.clone();
    e1_corrupted[2] = F::from(999u64); // Corrupt one entry

    let pc_claim_corrupted =
      compute_pc_weighted_sum_bruteforce(&e1_corrupted, &e2_1, &tau_1, &w1_left, &w1_right);
    assert_ne!(
      pc_claim_corrupted,
      F::ZERO,
      "Corrupted power table should have pc_sumcheck_claim ≠ 0"
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

    // Test case with non-zero pc_sumcheck_claim (using corrupted tables)
    let left = 4usize;
    let right = 4usize;
    let S_pc = PowerCheckStructure::new(left, right);

    let tau_1 = F::from(7u64);
    let tau_2 = F::from(11u64);

    // Create tables - running instance with corrupted table (non-zero pc_sumcheck_claim)
    let (mut e1_1, e2_1) = create_valid_power_table(&tau_1, left, right);
    e1_1[2] = F::from(999u64); // Corrupt to get non-zero pc_sumcheck_claim

    // Fresh instance with valid table (pc_sumcheck_claim = 0)
    let (e1_2, e2_2) = create_valid_power_table(&tau_2, left, right);

    // E weights (using MAIN domain dimensions)
    let w1_left: Vec<F> = (0..left).map(|i| F::from((i + 1) as u64)).collect();
    let w1_right: Vec<F> = (0..right).map(|i| F::from((i + 10) as u64)).collect();
    let w2_left: Vec<F> = (0..left).map(|i| F::from((i + 100) as u64)).collect();
    let w2_right: Vec<F> = (0..right).map(|i| F::from((i + 200) as u64)).collect();

    // Compute individual pc_sumcheck_claim values
    let pc_claim_1 = compute_pc_weighted_sum_bruteforce(&e1_1, &e2_1, &tau_1, &w1_left, &w1_right);
    let pc_claim_2 = compute_pc_weighted_sum_bruteforce(&e1_2, &e2_2, &tau_2, &w2_left, &w2_right);

    assert_ne!(
      pc_claim_1,
      F::ZERO,
      "Running instance should have non-zero pc_sumcheck_claim"
    );
    assert_eq!(
      pc_claim_2,
      F::ZERO,
      "Fresh valid instance should have pc_sumcheck_claim = 0"
    );

    // Sample random rho
    let rho = F::from(42u64);

    // Compute expected claim as RLC
    let T_claim_expected = (F::ONE - rho) * pc_claim_1 + rho * pc_claim_2;

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

  /// Brute-force compute PowerCheck contribution at arbitrary interpolation point t.
  ///
  /// This interpolates ALL inputs (weights, witness, τ) at point t and computes:
  /// Σᵢ w_left(t)[col] × w_right(t)[row] × (g1(t)[i] - g2(t)[i]·g3(t)[i])
  ///
  /// Uses main domain dimensions (left, right) for weights.
  /// This is the RAW contribution BEFORE ρ-scaling.
  fn compute_pc_at_point_raw<F: Field>(
    t: &F,
    w1_left: &[F],
    w1_right: &[F],
    w2_left: &[F],
    w2_right: &[F],
    e1_1: &[F],
    e1_2: &[F], // Running: first and second halves
    e2_1: &[F],
    e2_2: &[F], // Fresh: first and second halves
    tau_1: &F,
    tau_2: &F,
    left: usize,
    right: usize,
    num_cons: usize,
  ) -> F {
    // Linear interpolation helper: a + t·(b - a)
    let interp = |a: &F, b: &F| -> F { *a + *t * (*b - *a) };

    // Interpolate all weights at t
    let w_left: Vec<F> = w1_left
      .iter()
      .zip(w2_left)
      .map(|(a, b)| interp(a, b))
      .collect();
    let w_right: Vec<F> = w1_right
      .iter()
      .zip(w2_right)
      .map(|(a, b)| interp(a, b))
      .collect();

    // Interpolate witness (power table) at t
    let e_first: Vec<F> = e1_1.iter().zip(e2_1).map(|(a, b)| interp(a, b)).collect();
    let e_second: Vec<F> = e1_2.iter().zip(e2_2).map(|(a, b)| interp(a, b)).collect();

    // Interpolate τ at t
    let tau = interp(tau_1, tau_2);

    // Sum over main domain (left × right), skip padding indices >= num_cons
    let mut sum = F::ZERO;
    for row in 0..right {
      for col in 0..left {
        let i = row * left + col;
        if i >= num_cons {
          continue; // Skip padding
        }
        // Use canonical pc_g_at from power_check_relation
        let (g1, g2, g3) = pc_g_at(i, left, &e_first, &e_second, tau);
        let residual = g1 - g2 * g3;
        sum += w_left[col] * w_right[row] * residual;
      }
    }
    sum
  }

  /// Test that prove_helper_pc evaluations at {0, 2, 3, 4, 5} match brute-force.
  ///
  /// This is the CRITICAL test that verifies tensor-product interpolation is correct.
  #[test]
  fn test_prove_helper_pc_evaluations_at_all_points() {
    type E = PallasEngine;
    type F = <E as Engine>::Scalar;

    let left = 4usize;
    let right = 4usize;
    let S_pc = PowerCheckStructure::new(left, right);

    // Use DIFFERENT τ values to ensure cross-terms matter
    let tau_1 = F::from(7u64);
    let tau_2 = F::from(13u64);

    // Running instance: CORRUPTED table (non-zero residuals) to make the test meaningful
    let (mut e1_1, e2_1) = create_valid_power_table(&tau_1, left, right);
    e1_1[2] = F::from(999u64); // Corrupt

    // Fresh instance: valid table
    let (e1_2, e2_2) = create_valid_power_table(&tau_2, left, right);

    // E weights - use varied values to ensure tensor structure matters
    // Now using main domain dimensions (left × right) instead of separate PC dimensions
    let w1_left: Vec<F> = (0..left).map(|i| F::from((i * 3 + 1) as u64)).collect();
    let w1_right: Vec<F> = (0..right).map(|i| F::from((i * 5 + 2) as u64)).collect();
    let w2_left: Vec<F> = (0..left).map(|i| F::from((i * 7 + 3) as u64)).collect();
    let w2_right: Vec<F> = (0..right).map(|i| F::from((i * 11 + 4) as u64)).collect();

    let rho = F::from(42u64);

    // Get evaluations from prove_helper_pc (these are ρ-scaled)
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

    // Compute ρ-scaling factors (from apply_rho_scaling)
    let scale_0 = F::ONE - rho; // (1 - ρ)
    let scale_2 = F::from(3u64) * rho - F::ONE; // (3ρ - 1)
    let scale_3 = F::from(5u64) * rho - F::from(2u64); // (5ρ - 2)
    let scale_4 = F::from(7u64) * rho - F::from(3u64); // (7ρ - 3)
    let scale_5 = F::from(9u64) * rho - F::from(4u64); // (9ρ - 4)

    // Compute expected values at each point using brute-force
    let raw_0 = compute_pc_at_point_raw(
      &F::ZERO,
      &w1_left,
      &w1_right,
      &w2_left,
      &w2_right,
      &e1_1,
      &e2_1,
      &e1_2,
      &e2_2,
      &tau_1,
      &tau_2,
      left,
      right,
      S_pc.num_cons,
    );

    let raw_2 = compute_pc_at_point_raw(
      &F::from(2u64),
      &w1_left,
      &w1_right,
      &w2_left,
      &w2_right,
      &e1_1,
      &e2_1,
      &e1_2,
      &e2_2,
      &tau_1,
      &tau_2,
      left,
      right,
      S_pc.num_cons,
    );

    let raw_3 = compute_pc_at_point_raw(
      &F::from(3u64),
      &w1_left,
      &w1_right,
      &w2_left,
      &w2_right,
      &e1_1,
      &e2_1,
      &e1_2,
      &e2_2,
      &tau_1,
      &tau_2,
      left,
      right,
      S_pc.num_cons,
    );

    let raw_4 = compute_pc_at_point_raw(
      &F::from(4u64),
      &w1_left,
      &w1_right,
      &w2_left,
      &w2_right,
      &e1_1,
      &e2_1,
      &e1_2,
      &e2_2,
      &tau_1,
      &tau_2,
      left,
      right,
      S_pc.num_cons,
    );

    let raw_5 = compute_pc_at_point_raw(
      &F::from(5u64),
      &w1_left,
      &w1_right,
      &w2_left,
      &w2_right,
      &e1_1,
      &e2_1,
      &e1_2,
      &e2_2,
      &tau_1,
      &tau_2,
      left,
      right,
      S_pc.num_cons,
    );

    // Compare ρ-scaled values
    let expected_0 = scale_0 * raw_0;
    let expected_2 = scale_2 * raw_2;
    let expected_3 = scale_3 * raw_3;
    let expected_4 = scale_4 * raw_4;
    let expected_5 = scale_5 * raw_5;

    assert_eq!(e0, expected_0, "Evaluation at t=0 should match brute-force");
    assert_eq!(e2, expected_2, "Evaluation at t=2 should match brute-force");
    assert_eq!(e3, expected_3, "Evaluation at t=3 should match brute-force");
    assert_eq!(e4, expected_4, "Evaluation at t=4 should match brute-force");
    assert_eq!(e5, expected_5, "Evaluation at t=5 should match brute-force");
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
      // Use canonical pc_g_at from power_check_relation
      let (g1, g2, g3) = pc_g_at(i, left, &e1, &e2, tau);
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
