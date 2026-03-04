//! Translate arkworks R1CS rows to Nova LinearCombinations.
//!
//! Arkworks and Nova have different variable ordering:
//! - Arkworks: instance vars first (index 0 = "one"), witness vars after
//! - Nova: aux/witness first (`Index::Aux`), inputs last (`Index::Input`)
//!
//! This module provides translation between these orderings.

use super::field_bridge::FieldBridge;
use crate::frontend::{LinearCombination, Variable};
use ark_ff::PrimeField as ArkPrimeField;
use ff::PrimeField as NovaPrimeField;

/// Translate one arkworks matrix row to a Nova LinearCombination.
///
/// # Arguments
/// * `row` - The arkworks row as `(coefficient, column_index)` pairs
/// * `num_instance_vars` - Number of instance variables in the arkworks CS (including "one")
/// * `one_var` - The Nova variable representing the constant "one"
/// * `witness_map` - Maps arkworks witness indices to Nova Variables
///
/// # Variable Index Translation
///
/// In arkworks (after `inline_all_lcs()`):
/// - Index 0 to `num_instance_vars - 1`: instance variables (index 0 is "one")
/// - Index `num_instance_vars` onwards: witness variables
///
/// Our wrapper circuit has no public inputs (except the implicit "one"),
/// so only index 0 is expected in instance variables.
pub fn ark_row_to_nova_lc<AF, NF, B>(
  row: &[(AF, usize)],
  num_instance_vars: usize,
  one_var: Variable,
  witness_map: &[Variable],
) -> LinearCombination<NF>
where
  AF: ArkPrimeField,
  NF: NovaPrimeField,
  B: FieldBridge<AF, NF>,
{
  let mut lc = LinearCombination::zero();

  for (coeff_ark, idx) in row {
    let coeff = B::ark_to_nova(*coeff_ark);

    if *idx < num_instance_vars {
      // Instance variable - for our wrapper, only index 0 (one) is valid
      assert_eq!(
        *idx, 0,
        "arkworks wrapper should have no public inputs except 'one', got index {idx}"
      );
      lc = lc + (coeff, one_var);
    } else {
      // Witness variable - look up in our map
      let witness_idx = *idx - num_instance_vars;
      assert!(
        witness_idx < witness_map.len(),
        "witness index {witness_idx} out of bounds (max {})",
        witness_map.len()
      );
      lc = lc + (coeff, witness_map[witness_idx]);
    }
  }

  lc
}

/// Translate an arkworks variable index to a Nova Variable.
///
/// This is a simpler version of `ark_row_to_nova_lc` for single variables.
#[allow(dead_code)]
pub fn ark_var_to_nova<AF, NF, B>(
  idx: usize,
  num_instance_vars: usize,
  one_var: Variable,
  witness_map: &[Variable],
) -> Variable
where
  AF: ArkPrimeField,
  NF: NovaPrimeField,
  B: FieldBridge<AF, NF>,
{
  if idx < num_instance_vars {
    assert_eq!(idx, 0, "only 'one' instance variable supported");
    one_var
  } else {
    let witness_idx = idx - num_instance_vars;
    witness_map[witness_idx]
  }
}

/// Build a mapping from arkworks witness indices to Nova Variables.
///
/// # Arguments
/// * `input_vars` - Nova Variables for the step circuit input (z)
/// * `num_input_bits` - Number of input bits (should match `input_vars.len()`)
/// * `num_ark_witness_vars` - Total number of arkworks witness variables
/// * `alloc_fn` - Function to allocate remaining Nova witness variables
///
/// # Returns
/// A vector mapping arkworks witness index to Nova Variable.
///
/// The mapping is:
/// - Index 0..num_input_bits: mapped to `input_vars`
/// - Index num_input_bits..num_ark_witness_vars: newly allocated via `alloc_fn`
#[allow(dead_code)]
pub fn build_witness_map<NF: NovaPrimeField, F>(
  input_vars: &[Variable],
  num_input_bits: usize,
  num_ark_witness_vars: usize,
  mut alloc_fn: F,
) -> Vec<Variable>
where
  F: FnMut(usize) -> Variable,
{
  assert_eq!(input_vars.len(), num_input_bits);

  let mut witness_map = Vec::with_capacity(num_ark_witness_vars);

  // First `num_input_bits` witness vars map to input z[i]
  for i in 0..num_input_bits {
    witness_map.push(input_vars[i]);
  }

  // Allocate remaining witness vars
  for i in num_input_bits..num_ark_witness_vars {
    witness_map.push(alloc_fn(i));
  }

  witness_map
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::arkworks::Bn254Bridge;
  use crate::frontend::Index;
  use ark_bn254::Fr as ArkFr;
  use halo2curves::bn256::Fr as NovaFr;

  #[test]
  fn test_ark_row_to_nova_lc_one_only() {
    let one_var = Variable::new_unchecked(Index::Input(0));
    let witness_map = vec![
      Variable::new_unchecked(Index::Aux(0)),
      Variable::new_unchecked(Index::Aux(1)),
    ];

    // Row with just "one" coefficient
    let row: Vec<(ArkFr, usize)> = vec![(ArkFr::from(5u64), 0)];
    let lc = ark_row_to_nova_lc::<ArkFr, NovaFr, Bn254Bridge>(&row, 1, one_var, &witness_map);

    // Should have one input term
    assert_eq!(lc.len(), 1);
  }

  #[test]
  fn test_ark_row_to_nova_lc_witness_vars() {
    let one_var = Variable::new_unchecked(Index::Input(0));
    let witness_map = vec![
      Variable::new_unchecked(Index::Aux(0)),
      Variable::new_unchecked(Index::Aux(1)),
      Variable::new_unchecked(Index::Aux(2)),
    ];

    // Row with one + witness vars
    // Arkworks indices: 0 = one, 1 = witness[0], 2 = witness[1]
    let row: Vec<(ArkFr, usize)> = vec![
      (ArkFr::from(1u64), 0), // one
      (ArkFr::from(2u64), 1), // witness[0]
      (ArkFr::from(3u64), 2), // witness[1]
    ];
    let lc = ark_row_to_nova_lc::<ArkFr, NovaFr, Bn254Bridge>(&row, 1, one_var, &witness_map);

    // Should have 3 terms (1 input + 2 aux)
    assert_eq!(lc.len(), 3);
  }

  #[test]
  fn test_build_witness_map() {
    let input_vars = vec![
      Variable::new_unchecked(Index::Aux(100)),
      Variable::new_unchecked(Index::Aux(101)),
      Variable::new_unchecked(Index::Aux(102)),
    ];

    let mut next_idx = 200;
    let map = build_witness_map::<NovaFr, _>(&input_vars, 3, 5, |_| {
      let var = Variable::new_unchecked(Index::Aux(next_idx));
      next_idx += 1;
      var
    });

    assert_eq!(map.len(), 5);
    // First 3 should be input vars
    assert_eq!(map[0], input_vars[0]);
    assert_eq!(map[1], input_vars[1]);
    assert_eq!(map[2], input_vars[2]);
    // Last 2 should be newly allocated
    assert_eq!(map[3], Variable::new_unchecked(Index::Aux(200)));
    assert_eq!(map[4], Variable::new_unchecked(Index::Aux(201)));
  }
}
