//! Nova StepCircuit adapter that replays arkworks constraints.
//!
//! This module provides `NovaArkSha256Step`, a StepCircuit implementation that:
//! 1. Compiles an arkworks SHA-256 circuit
//! 2. Extracts the R1CS matrices and witness
//! 3. Replays all constraints into Nova's constraint system

use super::compile::compile_ark_sha256;
use super::field_bridge::FieldBridge;
use super::translate::ark_row_to_nova_lc;
use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use crate::traits::circuit::StepCircuit;
use ark_ff::PrimeField as ArkPrimeField;
use core::marker::PhantomData;
use ff::PrimeField as NovaPrimeField;

/// Nova StepCircuit that wraps an arkworks SHA-256 circuit.
///
/// This circuit:
/// - Takes 256 input bits (as field elements, each 0 or 1)
/// - Computes SHA-256 of the input
/// - Returns 256 output bits
///
/// The circuit works by:
/// 1. Compiling the arkworks SHA-256 circuit
/// 2. Mapping arkworks variables to Nova variables
/// 3. Replaying all arkworks constraints into Nova
#[derive(Clone, Default)]
pub struct NovaArkSha256Step<AF, NF, B>
where
  AF: ArkPrimeField,
  NF: NovaPrimeField,
  B: FieldBridge<AF, NF>,
{
  _phantom: PhantomData<(AF, NF, B)>,
}

impl<AF, NF, B> NovaArkSha256Step<AF, NF, B>
where
  AF: ArkPrimeField,
  NF: NovaPrimeField,
  B: FieldBridge<AF, NF>,
{
  /// Create a new instance of the SHA-256 step circuit.
  pub fn new() -> Self {
    Self {
      _phantom: PhantomData,
    }
  }
}

impl<AF, NF, B> StepCircuit<NF> for NovaArkSha256Step<AF, NF, B>
where
  AF: ArkPrimeField,
  NF: NovaPrimeField,
  B: FieldBridge<AF, NF> + Clone + Send + Sync,
{
  fn arity(&self) -> usize {
    256 // 256 bits for input and output
  }

  fn synthesize<CS: ConstraintSystem<NF>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<NF>],
  ) -> Result<Vec<AllocatedNum<NF>>, SynthesisError> {
    assert_eq!(z.len(), 256, "input must be exactly 256 bits");

    // 1) Constrain each z[i] to be a bit: z[i] * (1 - z[i]) = 0
    for (i, zi) in z.iter().enumerate() {
      cs.enforce(
        || format!("z[{i}] is bit"),
        |lc| lc + zi.get_variable(),
        |lc| lc + CS::one() - zi.get_variable(),
        |lc| lc,
      );
    }

    // 2) Extract bit values if in witness mode
    let ark_input_bits: Option<Vec<bool>> = if z.iter().all(|zi| zi.get_value().is_some()) {
      let bits: Result<Vec<bool>, _> = z
        .iter()
        .map(|zi| B::nova_bit(zi.get_value().unwrap()))
        .collect();
      Some(bits?)
    } else {
      None
    };

    // 3) Compile arkworks circuit
    let compiled = compile_ark_sha256::<AF>(ark_input_bits)?;

    // Sanity checks
    assert_eq!(
      compiled.num_instance_vars, 1,
      "arkworks circuit should have only the implicit 'one' instance variable"
    );
    assert!(
      compiled.num_witness_vars >= 512,
      "arkworks circuit should have at least 512 witness vars (256 in + 256 out), got {}",
      compiled.num_witness_vars
    );

    // 4) Build witness map: arkworks witness index -> Nova variable
    // First 256 witness vars map to input z[i]
    let mut witness_map = Vec::with_capacity(compiled.num_witness_vars);
    for i in 0..256 {
      witness_map.push(z[i].get_variable());
    }

    // Allocate remaining witness vars in Nova
    for i in 256..compiled.num_witness_vars {
      let val = compiled
        .witness_assignment
        .as_ref()
        .map(|ws| B::ark_to_nova(ws[i]));

      let var = cs.alloc(
        || format!("ark_witness_{i}"),
        || val.ok_or(SynthesisError::AssignmentMissing),
      )?;
      witness_map.push(var);
    }

    // 5) Replay every arkworks constraint row
    for r in 0..compiled.num_constraints {
      let a_row = &compiled.matrices.a[r];
      let b_row = &compiled.matrices.b[r];
      let c_row = &compiled.matrices.c[r];

      // Build linear combinations by translating arkworks indices to Nova variables
      let a_lc = ark_row_to_nova_lc::<AF, NF, B>(
        a_row,
        compiled.num_instance_vars,
        CS::one(),
        &witness_map,
      );
      let b_lc = ark_row_to_nova_lc::<AF, NF, B>(
        b_row,
        compiled.num_instance_vars,
        CS::one(),
        &witness_map,
      );
      let c_lc = ark_row_to_nova_lc::<AF, NF, B>(
        c_row,
        compiled.num_instance_vars,
        CS::one(),
        &witness_map,
      );

      cs.enforce(
        || format!("ark_constraint_{r}"),
        |_| a_lc.clone(),
        |_| b_lc.clone(),
        |_| c_lc.clone(),
      );
    }

    // 6) Return last 256 witness vars as z_{i+1}
    // The arkworks circuit allocates output bits as the last 256 witness variables
    let out_start = compiled.num_witness_vars - 256;
    let mut z_next = Vec::with_capacity(256);

    for j in 0..256 {
      let w_idx = out_start + j;

      let val = compiled
        .witness_assignment
        .as_ref()
        .map(|ws| B::ark_to_nova(ws[w_idx]));

      let num = AllocatedNum::alloc(cs.namespace(|| format!("z_next_{j}")), || {
        val.ok_or(SynthesisError::AssignmentMissing)
      })?;

      // Constrain: witness_map[w_idx] == num
      // This binds the output AllocatedNum to the corresponding arkworks witness variable
      cs.enforce(
        || format!("bind_z_next_{j}"),
        |lc| lc + witness_map[w_idx] - num.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
      );

      z_next.push(num);
    }

    Ok(z_next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::arkworks::Bn254Bridge;
  use ark_bn254::Fr as ArkFr;
  use halo2curves::bn256::Fr as NovaFr;

  // Note: TestConstraintSystem would be at crate::frontend::util_cs::test_cs::TestConstraintSystem
  // but it's not needed for the basic arity test

  #[test]
  fn test_step_circuit_arity() {
    let circuit = NovaArkSha256Step::<ArkFr, NovaFr, Bn254Bridge>::new();
    assert_eq!(circuit.arity(), 256);
  }

  // Note: Full synthesis tests are expensive and should be run in release mode
  // See examples/sha256_arkworks.rs for a complete end-to-end test
}
