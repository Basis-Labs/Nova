//! Example: SHA-256 hash chain using arkworks crypto-primitives with Nova IVC.
//!
//! This example demonstrates how to use arkworks SHA-256 circuit gadgets within
//! Nova's IVC framework. The circuit computes a hash chain:
//!
//! z_0 -> SHA256(z_0) -> SHA256(SHA256(z_0)) -> ...
//!
//! Each step takes 256 bits (as field elements) and produces 256 bits.
//!
//! Run with: cargo run --release --features arkworks --example sha256_arkworks

use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  arkworks::{Bn254Bridge, NovaArkSha256Step},
  nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::snark::RelaxedR1CSSNARKTrait,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;

type ArkFr = ark_bn254::Fr;
type NovaFr = halo2curves::bn256::Fr;

/// Convert a byte array to 256 field elements (one per bit).
fn bytes_to_bits<F: Field>(bytes: &[u8]) -> Vec<F> {
  assert_eq!(bytes.len(), 32, "expected 32 bytes for SHA-256 hash");
  let mut bits = Vec::with_capacity(256);
  for byte in bytes {
    for i in 0..8 {
      bits.push(if (byte >> i) & 1 == 1 {
        F::ONE
      } else {
        F::ZERO
      });
    }
  }
  bits
}

/// Convert 256 field elements (bits) to a byte array.
#[allow(dead_code)]
fn bits_to_bytes<F: Field + std::cmp::PartialEq>(bits: &[F]) -> [u8; 32] {
  assert_eq!(bits.len(), 256, "expected 256 bits");
  let mut bytes = [0u8; 32];
  for (i, chunk) in bits.chunks(8).enumerate() {
    let mut byte = 0u8;
    for (j, bit) in chunk.iter().enumerate() {
      if *bit == F::ONE {
        byte |= 1 << j;
      }
    }
    bytes[i] = byte;
  }
  bytes
}

/// Compute SHA-256 of the input bits and return output bits.
#[allow(dead_code)]
fn compute_expected_hash<F: Field>(input_bits: &[F]) -> Vec<F>
where
  F: std::cmp::PartialEq,
{
  let input_bytes = bits_to_bytes(input_bits);
  let hash = Sha256::digest(input_bytes);
  bytes_to_bits::<F>(&hash)
}

fn main() {
  println!("=========================================================");
  println!("Nova-based arkworks SHA-256 hashchain example");
  println!("=========================================================");

  let num_steps = 3;

  // Create the arkworks SHA-256 step circuit
  let circuit = NovaArkSha256Step::<ArkFr, NovaFr, Bn254Bridge>::new();

  // Produce public parameters
  println!("Producing public parameters...");
  let start = Instant::now();
  let pp = PublicParams::<E1, E2, NovaArkSha256Step<ArkFr, NovaFr, Bn254Bridge>>::setup(
    &circuit,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
  )
  .expect("Failed to setup public parameters");
  println!("PublicParams::setup took {:?}", start.elapsed());

  println!(
    "Number of constraints per step (primary circuit): {}",
    pp.num_constraints().0
  );
  println!(
    "Number of constraints per step (secondary circuit): {}",
    pp.num_constraints().1
  );
  println!(
    "Number of variables per step (primary circuit): {}",
    pp.num_variables().0
  );
  println!(
    "Number of variables per step (secondary circuit): {}",
    pp.num_variables().1
  );

  // Initial state: SHA256 of "hello" as 256 bits
  let initial_hash = Sha256::digest(b"hello");
  let z0: Vec<NovaFr> = bytes_to_bits(&initial_hash);
  assert_eq!(z0.len(), 256);

  println!("\nInitial hash (SHA256(\"hello\")): {:x}", initial_hash);

  // Create recursive SNARK
  println!("\nCreating RecursiveSNARK...");
  let start = Instant::now();
  let mut recursive_snark =
    RecursiveSNARK::<E1, E2, NovaArkSha256Step<ArkFr, NovaFr, Bn254Bridge>>::new(
      &pp, &circuit, &z0,
    )
    .expect("Failed to create RecursiveSNARK");
  println!("RecursiveSNARK::new took {:?}", start.elapsed());

  // Prove multiple steps
  println!("\nProving {num_steps} steps of SHA-256 hash chain...");

  // Track expected values for verification
  let mut expected_hash = initial_hash.to_vec();

  for step in 0..num_steps {
    let start = Instant::now();
    recursive_snark
      .prove_step(&pp, &circuit)
      .expect("Failed to prove step");
    println!(
      "RecursiveSNARK::prove_step {} took {:?}",
      step,
      start.elapsed()
    );

    // Compute expected hash for this step
    expected_hash = Sha256::digest(&expected_hash).to_vec();
  }

  println!("\nExpected final hash: {:02x?}", &expected_hash[..8]);

  // Verify the recursive SNARK
  println!("\nVerifying RecursiveSNARK...");
  let start = Instant::now();
  let zi = recursive_snark
    .verify(&pp, num_steps, &z0)
    .expect("Failed to verify RecursiveSNARK");
  println!("RecursiveSNARK::verify took {:?}", start.elapsed());

  // Verify the output matches expected
  let final_hash_bits: Vec<NovaFr> = bytes_to_bits(&expected_hash);
  assert_eq!(zi.len(), 256, "Output should have 256 elements");

  // Check that output bits match
  let mut matching = true;
  for i in 0..256 {
    if zi[i] != final_hash_bits[i] {
      matching = false;
      println!(
        "Mismatch at bit {i}: got {:?}, expected {:?}",
        zi[i], final_hash_bits[i]
      );
    }
  }

  if matching {
    println!("\n✓ Output matches expected hash!");
  } else {
    println!("\n✗ Output does not match expected hash");
  }

  // Generate CompressedSNARK
  println!("\n---------------------------------------------------------");
  println!("Generating CompressedSNARK using Spartan with HyperKZG...");
  println!("---------------------------------------------------------");

  let start = Instant::now();
  let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp).unwrap();
  println!("CompressedSNARK::setup took {:?}", start.elapsed());

  let start = Instant::now();
  let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark)
    .expect("Failed to create CompressedSNARK");
  println!("CompressedSNARK::prove took {:?}", start.elapsed());

  // Measure compressed size
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serde::encode_into_std_write(&compressed_snark, &mut encoder, bincode::config::legacy())
    .expect("Failed to serialize compressed SNARK");
  let compressed_snark_encoded = encoder.finish().unwrap();
  println!(
    "CompressedSNARK size: {} bytes (compressed)",
    compressed_snark_encoded.len()
  );

  // Verify the compressed SNARK
  println!("\nVerifying CompressedSNARK...");
  let start = Instant::now();
  let zi_compressed = compressed_snark
    .verify(&vk, num_steps, &z0)
    .expect("Failed to verify CompressedSNARK");
  println!("CompressedSNARK::verify took {:?}", start.elapsed());

  // Verify output matches
  assert_eq!(
    zi_compressed, zi,
    "CompressedSNARK output should match RecursiveSNARK output"
  );
  println!("✓ CompressedSNARK output matches RecursiveSNARK output!");

  println!("\n=========================================================");
  println!("SHA-256 arkworks integration test completed successfully!");
  println!("=========================================================");
}
