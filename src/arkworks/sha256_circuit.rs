//! Arkworks SHA-256 circuit wrapper for Nova IVC.
//!
//! This module provides an arkworks-based SHA-256 step circuit that computes
//! SHA-256(input) where input and output are represented as 256 bits.

use ark_crypto_primitives::crh::sha256::constraints::Sha256Gadget;
use ark_ff::PrimeField as ArkPrimeField;
use ark_r1cs_std::{alloc::AllocVar, boolean::Boolean, convert::ToBitsGadget, eq::EqGadget, uint8::UInt8, R1CSVar};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};

/// Arkworks circuit for one SHA-256 hash step.
///
/// The circuit takes 256 input bits and produces 256 output bits (SHA-256 hash).
///
/// **Allocation order is critical:**
/// - First 256 witness variables = input bits
/// - Last 256 witness variables = output bits
///
/// This ordering is used by the replay adapter to map arkworks variables
/// to Nova variables.
#[derive(Clone, Default)]
pub struct ArkSha256HashStep<F: ArkPrimeField> {
  /// Input bits for the SHA-256 hash. `None` during setup, `Some(...)` during witness generation.
  pub in_bits: Option<Vec<bool>>,
  _phantom: core::marker::PhantomData<F>,
}

impl<F: ArkPrimeField> ArkSha256HashStep<F> {
  /// Create a new circuit for setup (no witness values).
  pub fn new_setup() -> Self {
    Self {
      in_bits: None,
      _phantom: core::marker::PhantomData,
    }
  }

  /// Create a new circuit with witness values.
  pub fn new_witness(in_bits: Vec<bool>) -> Self {
    assert_eq!(in_bits.len(), 256, "input must be exactly 256 bits");
    Self {
      in_bits: Some(in_bits),
      _phantom: core::marker::PhantomData,
    }
  }
}

impl<F: ArkPrimeField> ConstraintSynthesizer<F> for ArkSha256HashStep<F> {
  fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
    // 1) Allocate 256 input bits FIRST (this ensures they get witness indices 0..255)
    let input_bits: Vec<Boolean<F>> = (0..256)
      .map(|i| {
        Boolean::new_witness(cs.clone(), || {
          self
            .in_bits
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .ok_or(SynthesisError::AssignmentMissing)
        })
      })
      .collect::<Result<_, _>>()?;

    // 2) Pack bits into bytes (little-endian within each byte)
    let input_bytes: Vec<UInt8<F>> = input_bits
      .chunks(8)
      .map(|chunk| UInt8::from_bits_le(chunk))
      .collect();

    // 3) Compute SHA-256 digest using arkworks gadget
    let digest = Sha256Gadget::digest(&input_bytes)?;

    // 4) Flatten digest to 256 bits
    let mut digest_bits: Vec<Boolean<F>> = Vec::with_capacity(256);
    for byte in &digest.0 {
      digest_bits.extend(byte.to_bits_le()?);
    }
    assert_eq!(digest_bits.len(), 256);

    // 5) Allocate 256 output bits LAST, constrain equality to digest
    // This ensures output bits get the last 256 witness indices
    for bit in digest_bits {
      let out = Boolean::new_witness(cs.clone(), || bit.value())?;
      out.enforce_equal(&bit)?;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr;
  use ark_relations::r1cs::ConstraintSystem;

  #[test]
  fn test_sha256_circuit_setup() {
    let cs = ConstraintSystem::<Fr>::new_ref();
    let circuit = ArkSha256HashStep::<Fr>::new_setup();
    circuit.generate_constraints(cs.clone()).unwrap();

    // Should have constraints
    assert!(cs.num_constraints() > 0);
    // Should have witness variables (at least 512 for input + output)
    assert!(cs.num_witness_variables() >= 512);
  }

  #[test]
  fn test_sha256_circuit_witness() {
    let cs = ConstraintSystem::<Fr>::new_ref();

    // Create input bits (all zeros for simplicity)
    let in_bits = vec![false; 256];
    let circuit = ArkSha256HashStep::<Fr>::new_witness(in_bits);
    circuit.generate_constraints(cs.clone()).unwrap();

    // Should be satisfied
    assert!(cs.is_satisfied().unwrap());
  }
}
