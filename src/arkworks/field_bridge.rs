//! Field bridge between arkworks and Nova field types.

use crate::frontend::SynthesisError;
use ark_ff::PrimeField as ArkPrimeField;
use ark_serialize::CanonicalSerialize;
use ff::PrimeField as NovaPrimeField;

/// Bridge trait for converting between arkworks and Nova field types.
pub trait FieldBridge<AF: ArkPrimeField, NF: NovaPrimeField>: Clone + Send + Sync {
  /// Convert an arkworks field element to a Nova field element.
  fn ark_to_nova(af: AF) -> NF;

  /// Convert a Nova field element to an arkworks field element.
  fn nova_to_ark(nf: NF) -> AF;

  /// Convert a Nova field element (expected to be 0 or 1) to a boolean.
  fn nova_bit(nf: NF) -> Result<bool, SynthesisError>;
}

/// Field bridge for BN254 scalar field.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bn254Bridge;

impl FieldBridge<ark_bn254::Fr, halo2curves::bn256::Fr> for Bn254Bridge {
  fn ark_to_nova(af: ark_bn254::Fr) -> halo2curves::bn256::Fr {
    let mut bytes = [0u8; 32];
    af.serialize_compressed(&mut bytes[..])
      .expect("serialization should not fail");
    halo2curves::bn256::Fr::from_repr(bytes.into()).expect("bytes should be valid")
  }

  fn nova_to_ark(nf: halo2curves::bn256::Fr) -> ark_bn254::Fr {
    use ff::PrimeField;
    let bytes: [u8; 32] = nf.to_repr().into();
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
  }

  fn nova_bit(nf: halo2curves::bn256::Fr) -> Result<bool, SynthesisError> {
    use ff::Field;
    if nf == halo2curves::bn256::Fr::ZERO {
      Ok(false)
    } else if nf == halo2curves::bn256::Fr::ONE {
      Ok(true)
    } else {
      Err(SynthesisError::Unsatisfiable("expected 0 or 1".to_string()))
    }
  }
}
