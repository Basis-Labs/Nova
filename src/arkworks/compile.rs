//! Compile arkworks circuits and extract R1CS matrices.
//!
//! This module provides utilities to synthesize an arkworks circuit and extract
//! the R1CS matrices and witness assignments for replay into Nova.

use super::sha256_circuit::ArkSha256HashStep;
use ark_ff::PrimeField as ArkPrimeField;
use ark_relations::r1cs::{ConstraintMatrices, ConstraintSystem, ConstraintSynthesizer};

/// A compiled arkworks circuit with extracted matrices and assignments.
pub struct CompiledArkCircuit<F: ArkPrimeField> {
  /// R1CS matrices (A, B, C) in arkworks format: `Vec<Vec<(coefficient, column_index)>>`
  pub matrices: ConstraintMatrices<F>,
  /// Witness assignment (private variables), if available
  pub witness_assignment: Option<Vec<F>>,
  /// Instance assignment (public inputs including the implicit "one")
  pub instance_assignment: Vec<F>,
  /// Number of witness (private) variables
  pub num_witness_vars: usize,
  /// Number of instance (public) variables (includes the implicit "one" at index 0)
  pub num_instance_vars: usize,
  /// Number of constraints
  pub num_constraints: usize,
}

/// Compile an arkworks circuit and extract R1CS matrices.
///
/// This function:
/// 1. Creates an arkworks constraint system
/// 2. Synthesizes the circuit into it
/// 3. Calls `inline_all_lcs()` to expand symbolic linear combinations
/// 4. Extracts the R1CS matrices and witness assignments
///
/// # Arguments
/// * `circuit` - The arkworks circuit to compile
///
/// # Returns
/// A `CompiledArkCircuit` containing the matrices and assignments
pub fn compile_ark_circuit<F, C>(circuit: C) -> Result<CompiledArkCircuit<F>, crate::frontend::SynthesisError>
where
  F: ArkPrimeField,
  C: ConstraintSynthesizer<F>,
{
  let cs = ConstraintSystem::new_ref();

  // Synthesize the circuit
  circuit
    .generate_constraints(cs.clone())
    .map_err(|e| crate::frontend::SynthesisError::Unsatisfiable(format!("arkworks synthesis failed: {e:?}")))?;

  // Inline all symbolic linear combinations before matrix export
  cs.inline_all_lcs();

  // Note: Do NOT call finalize() - it has a bug in transform_lc_map for large circuits.
  // to_matrices() works without finalize() after inline_all_lcs().

  // Extract matrices
  let matrices = cs
    .to_matrices()
    .ok_or_else(|| crate::frontend::SynthesisError::Unsatisfiable("failed to extract matrices".to_string()))?;

  // Get counts before consuming the CS
  let num_witness_vars = cs.num_witness_variables();
  let num_instance_vars = cs.num_instance_variables();
  let num_constraints = cs.num_constraints();

  // Extract the inner constraint system to get assignments
  let cs_inner = cs
    .into_inner()
    .ok_or_else(|| crate::frontend::SynthesisError::Unsatisfiable("failed to get inner CS".to_string()))?;

  let witness_assignment = if cs_inner.witness_assignment.is_empty() {
    None
  } else {
    Some(cs_inner.witness_assignment)
  };

  Ok(CompiledArkCircuit {
    matrices,
    witness_assignment,
    instance_assignment: cs_inner.instance_assignment,
    num_witness_vars,
    num_instance_vars,
    num_constraints,
  })
}

/// Compile an arkworks SHA-256 circuit.
///
/// This is a convenience wrapper around `compile_ark_circuit` for the SHA-256 step circuit.
///
/// # Arguments
/// * `input_bits` - Optional input bits. `None` for setup mode (uses dummy zeros), `Some(bits)` for witness generation.
///
/// # Returns
/// A `CompiledArkCircuit` containing the SHA-256 circuit's matrices and assignments
pub fn compile_ark_sha256<F: ArkPrimeField>(
  input_bits: Option<Vec<bool>>,
) -> Result<CompiledArkCircuit<F>, crate::frontend::SynthesisError> {
  let cs = ConstraintSystem::new_ref();

  // Always use witness mode with actual values to ensure proper variable allocation.
  // For setup, we use dummy zeros; for proving, we use actual input bits.
  let actual_bits = input_bits.unwrap_or_else(|| vec![false; 256]);

  // Create and synthesize the circuit with actual witness values
  let circuit = ArkSha256HashStep::new_witness(actual_bits);

  circuit
    .generate_constraints(cs.clone())
    .map_err(|e| crate::frontend::SynthesisError::Unsatisfiable(format!("SHA256 synthesis failed: {e:?}")))?;

  // Inline all symbolic linear combinations before matrix export
  cs.inline_all_lcs();

  // Note: Do NOT call finalize() - it has a bug in transform_lc_map for large circuits.
  // to_matrices() works without finalize() after inline_all_lcs().

  // Extract matrices
  let matrices = cs
    .to_matrices()
    .ok_or_else(|| crate::frontend::SynthesisError::Unsatisfiable("failed to extract SHA256 matrices".to_string()))?;

  // Get counts before consuming the CS
  let num_witness_vars = cs.num_witness_variables();
  let num_instance_vars = cs.num_instance_variables();
  let num_constraints = cs.num_constraints();

  // Extract the inner constraint system to get assignments
  let cs_inner = cs
    .into_inner()
    .ok_or_else(|| crate::frontend::SynthesisError::Unsatisfiable("failed to get inner SHA256 CS".to_string()))?;

  let witness_assignment = if cs_inner.witness_assignment.is_empty() {
    None
  } else {
    Some(cs_inner.witness_assignment)
  };

  Ok(CompiledArkCircuit {
    matrices,
    witness_assignment,
    instance_assignment: cs_inner.instance_assignment,
    num_witness_vars,
    num_instance_vars,
    num_constraints,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr;

  #[test]
  fn test_compile_sha256_setup() {
    let compiled = compile_ark_sha256::<Fr>(None).unwrap();

    // Should have exactly 1 instance var (the implicit "one")
    assert_eq!(compiled.num_instance_vars, 1);

    // Should have at least 512 witness vars (256 input + 256 output)
    assert!(
      compiled.num_witness_vars >= 512,
      "expected >= 512 witness vars, got {}",
      compiled.num_witness_vars
    );

    // Should have constraints
    assert!(compiled.num_constraints > 0);

    // In setup mode, witness assignment should be None or empty values
    // (arkworks may still populate with zeros)
  }

  #[test]
  fn test_compile_sha256_witness() {
    let input_bits = vec![false; 256];
    let compiled = compile_ark_sha256::<Fr>(Some(input_bits)).unwrap();

    // Should have witness assignment
    assert!(compiled.witness_assignment.is_some());
    let witness = compiled.witness_assignment.unwrap();
    assert_eq!(witness.len(), compiled.num_witness_vars);
  }
}
