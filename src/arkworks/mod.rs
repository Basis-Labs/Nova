//! Integration layer for using arkworks circuits with Nova IVC.
//!
//! This module provides a bridge that allows arkworks-based circuits (like SHA-256 from
//! `ark-crypto-primitives`) to be used as Nova step circuits.
//!
//! # Architecture
//!
//! The bridge works by:
//! 1. Compiling the arkworks circuit into arkworks' constraint system
//! 2. Extracting the R1CS matrices and witness assignments
//! 3. Replaying the constraints into Nova's constraint system
//!
//! This approach avoids modifying Nova's core while enabling arkworks circuit reuse.
//!
//! # Example
//!
//! ```ignore
//! use nova_snark::arkworks::{Bn254Bridge, NovaArkSha256Step};
//! use nova_snark::provider::{Bn256EngineKZG, GrumpkinEngine};
//!
//! type E1 = Bn256EngineKZG;
//! type E2 = GrumpkinEngine;
//!
//! let circuit = NovaArkSha256Step::<ark_bn254::Fr, halo2curves::bn256::Fr, Bn254Bridge>::default();
//! let pp = nova_snark::PublicParams::<E1, E2, _>::setup(&circuit, ...)?;
//! ```

mod compile;
mod field_bridge;
mod sha256_circuit;
mod step_circuit;
mod translate;

pub use compile::{compile_ark_circuit, CompiledArkCircuit};
pub use field_bridge::{Bn254Bridge, FieldBridge};
pub use sha256_circuit::ArkSha256HashStep;
pub use step_circuit::NovaArkSha256Step;
pub use translate::ark_row_to_nova_lc;
