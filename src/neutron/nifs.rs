//! This module implements the basic NIFS (Non-Interactive Folding Scheme) for NeutronNova
//!
//! For the complete ZeroFold prover (NSC + NSC_PC), see `zerofold_nifs.rs`.
#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  neutron::{
    relation::{FoldedInstance, FoldedWitness, Structure},
    sumcheck::{prove_helper, EvalAcc},
    weight_table::WeightTable,
  },
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::polys::univariate::UniPoly,
  traits::{AbsorbInRO2Trait, Engine, RO2Constants, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use serde::{Deserialize, Serialize};

/// An NIFS message from NeutronNova's folding scheme (single NSC relation)
///
/// For the combined NSC + NSC_PC proof, see `ZeroFoldNIFS` in `zerofold_nifs.rs`.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_E: Commitment<E>,
  pub(crate) poly: UniPoly<E::Scalar>,
}

impl<E: Engine> NIFS<E> {
  /// Takes as input a folded instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with a compatible structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  ///
  /// Note that this code is tailored for use with NeutronNova's IVC scheme, which enforces
  /// certain requirements between the two instances that are folded.
  /// In particular, it requires that `U1` and `U2` are such that the hash of `U1` is stored in the public IO of `U2`.
  /// In this particular setting, this means that if `U2` is absorbed in the RO, it implicitly absorbs `U1` as well.
  /// So the code below avoids absorbing `U1` in the RO.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(NIFS<E>, (FoldedInstance<E>, FoldedWitness<E>)), NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute a commitment to the eq polynomial
    let E = WeightTable::from_tau(&tau, S.left, S.right);
    let comm_E: Commitment<E> = E.commit(ck);

    comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // We now run a single round of the sum-check protocol to establish
    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    let (res1, res2) = rayon::join(
      || {
        let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
        S.S.multiply_vec(&z1)
      },
      || {
        let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
        S.S.multiply_vec(&z2)
      },
    );

    let (Az1, Bz1, Cz1) = res1?;
    let (Az2, Bz2, Cz2) = res2?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4, eval_point_5): EvalAcc<E::Scalar> =
      prove_helper::<E>(
        &rho,
        (S.left, S.right),
        W1.E.as_slice(),
        &Az1,
        &Bz1,
        &Cz1,
        E.as_slice(),
        &Az2,
        &Bz2,
        &Cz2,
      );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // === ASSERT: Sumcheck Identity ===
    // poly(0) + poly(1) = (1-ρ)·T_running + ρ·T_fresh where T_fresh = 0
    assert_eq!(
      poly.eval_at_zero() + poly.eval_at_one(),
      T,
      "Sumcheck identity violated: poly(0) + poly(1) ≠ (1-ρ)·T_running"
    );

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_inv: E::Scalar =
      Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b_inv;

    // === ASSERT: Output Claim Consistency ===
    // poly(r_b) = T_out · eq(ρ, r_b)
    assert_eq!(
      poly.evaluate(&r_b),
      T_out * eq_rho_r_b,
      "Output claim inconsistent: poly(r_b) ≠ T_out · eq(ρ, r_b)"
    );

    let U = U1.fold(U2, &comm_E, &r_b, &T_out)?;
    let W = W1.fold(W2, &E, &r_b)?;

    // return the folded instance and witness
    Ok((Self { comm_E, poly }, (U, W)))
  }

  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  #[cfg(test)]
  pub fn verify(
    &self,
    ro_consts: &RO2Constants<E>,
    pp_digest: &E::Scalar,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
    // TODO: U_PC: FoldedPowerCheckInstance
    // TODO: Generated PowerCheckInstance (Maybe)
    // U2
  ) -> Result<FoldedInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO2::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript
    U2.absorb_in_ro2(&mut ro);

    // TODO: absorb U_pc

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(NUM_CHALLENGE_BITS, false);

    self.comm_E.absorb_in_ro2(&mut ro); // absorb the commitment in the NIFS

    // Check that tau and comm_E is equal to the generated nested power check instance
    //

    // TODO: We should also try to convert the R1CSInstance and PowerCheckInstance into their folding counterparts by this time

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // TODO: Using the sumcheck for all the NSC relations. Add them up together and the check with the polynomial

    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;
    // check if poly(0) + poly(1) = T
    if self.poly.eval_at_zero() + self.poly.eval_at_one() != T {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // TODO: Now note that the sumchecks of the folded NSC and NSC_PC are computed already.

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInRO2Trait<E>>::absorb_in_ro2(&self.poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS, false);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_inv: E::Scalar =
      Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;
    let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b_inv;

    /////////////NOTE - We should do this for the verifier
    // let r_b = transcript.squeeze(NUM_CHALLENGE_BITS, false);

    // // Step 7: Compute output claims
    // let eq_rho_r_b = one_minus_rho * (E::Scalar::ONE - r_b) + *rho * r_b;
    // let eq_rho_r_b_inv: E::Scalar =
    //   Option::from(eq_rho_r_b.invert()).ok_or(NovaError::DivideByZero)?;

    // let sumcheck_claim_out_nsc = poly_nsc.evaluate(&r_b) * eq_rho_r_b_inv;
    // let sumcheck_claim_out_pc = poly_pc.evaluate(&r_b) * eq_rho_r_b_inv;

    let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;
    // TODO: update the ZeroCheck Instance here as well with a fold.

    // return the folded instance and witness
    Ok(U)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      Circuit, ConstraintSystem,
    },
    provider::{
      hyperkzg::EvaluationEngine as HyperKZGEE, ipa_pc::EvaluationEngine, Bn256EngineKZG,
      PallasEngine, Secp256k1Engine,
    },
    r1cs::R1CSShape,
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait, Engine, RO2Constants},
  };

  fn execute_sequence<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &RO2Constants<E>,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) {
    // produce a default running instance
    let str = Structure::new(shape);
    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U1, W1,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce an NIFS with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U2, W2,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify an NIFS with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    // check if the running instance is satisfiable
    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());
  }

  fn test_tiny_r1cs_bellpepper_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let ro_consts = RO2Constants::<E>::default();

    // generate a non-trivial circuit
    let num_cons: usize = 32;

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let shape = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()]).unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(2)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U1, W1) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> = DirectCircuit::new(
      Some(vec![E::Scalar::from(3)]),
      NonTrivialCircuit::<E::Scalar>::new(num_cons),
    );
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U2, W2) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })
      .unwrap();

    // pad the shape and witnesses
    let shape = shape.pad();
    let W1 = W1.pad(&shape);
    let W2 = W2.pad(&shape);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
    test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG, RelaxedR1CSSNARK<_, HyperKZGEE<_>>>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine, RelaxedR1CSSNARK<_, EvaluationEngine<_>>>();
  }
}

#[cfg(test)]
mod benchmarks {
  use super::*;
  use crate::{
    frontend::{
      gadgets::{
        boolean::{AllocatedBit, Boolean},
        num::AllocatedNum,
        sha256::sha256,
      },
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      ConstraintSystem, SynthesisError,
    },
    nova::nifs::NIFS as NovaNIFS,
    provider::Bn256EngineKZG,
    r1cs::{R1CSShape, SparseMatrix},
    traits::{commitment::CommitmentEngineTrait, snark::default_ck_hint, ROConstants},
  };
  use core::marker::PhantomData;
  use criterion::Criterion;
  use ff::PrimeField;
  use num_integer::Integer;
  use num_traits::ToPrimitive;
  use rand::Rng;
  use rayon::prelude::*;

  /// generates a satisfying R1CS with small witness values
  fn generate_sample_r1cs<E: Engine>(
    num_cons: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let num_vars = num_cons;
    let num_io = 1;

    // we will just generate constraints of the form x * x = x, checking Booleanity
    // generate the constraints by creating sparse matrices
    let A = SparseMatrix::new(
      &(0..num_cons)
        .map(|i| (i, i, E::Scalar::ONE))
        .collect::<Vec<_>>(),
      num_cons,
      num_vars + 1 + num_io,
    );
    let B = A.clone();
    let C = A.clone();

    let S: R1CSShape<E> = R1CSShape::new(num_cons, num_vars, num_io, A, B, C).unwrap();

    let S = S.pad();

    // sample a ck
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    // let witness be randomly generated booleans
    let w = (0..S.num_cons)
      .into_par_iter()
      .map(|_| {
        let mut rng = rand::thread_rng();
        let result: u8 = rng.gen();
        result % 2
      })
      .collect::<Vec<_>>();

    let W = {
      // convert W to field elements
      let W = (0..S.num_cons)
        .into_par_iter()
        .map(|i| <E as Engine>::Scalar::from(w[i] as u64))
        .collect::<Vec<_>>();
      R1CSWitness::new(&S, &W).unwrap()
    };

    let x = vec![E::Scalar::from(0)];
    (S, ck, W, w, x)
  }

  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> Sha256Circuit<E> {
    pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  fn generarate_sha_r1cs<E: Engine>(
    len: usize,
  ) -> (
    R1CSShape<E>,
    CommitmentKey<E>,
    R1CSWitness<E>,
    Vec<u8>,
    Vec<E::Scalar>,
  ) {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let S = cs.r1cs_shape().unwrap();
    let ck = R1CSShape::commitment_key(&[&S], &[&*default_ck_hint()]).unwrap();

    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (U, W) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

    let S = S.pad();
    let W = W.pad(&S);

    let w = W
      .W
      .iter()
      .map(|e| {
        // map field element to u8
        // this assumes little-endian representation
        e.to_repr().as_ref()[0]
      })
      .collect::<Vec<_>>();

    // sanity check by recommiting to w
    let comm_W = <E as Engine>::CE::commit_small(&ck, &w, &W.r_W);
    assert_eq!(comm_W, U.comm_W);

    let X = U.X.clone();
    (S, ck, W, w, X)
  }

  fn bench_nifs_inner<E: Engine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    c: &mut Criterion,
    name: &str,
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    W: &R1CSWitness<E>,
    w: &[T],
    x: &[E::Scalar],
  ) {
    let num_cons = S.num_cons;

    // generate a default running instance
    let str = Structure::new(S);
    let f_W = FoldedWitness::default(&str);
    let f_U = FoldedInstance::default(&str);
    let res = str.is_sat(ck, &f_U, &f_W);
    assert!(res.is_ok());

    // generate default values
    let pp_digest = E::Scalar::ZERO;
    let ro_consts = RO2Constants::<E>::default();

    // produce an NIFS with (W, U) as the first incoming witness-instance pair
    c.bench_function(&format!("neutron_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit with the specialized method
        let comm_W = E::CE::commit_small(ck, w, &W.r_W);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NIFS::prove(ck, &ro_consts, &pp_digest, &str, &f_U, &f_W, &U, W);
        assert!(res.is_ok());
      })
    });

    // generate a random relaxed R1CS instance-witness pair
    let (r_U, r_W) = R1CSShape::<E>::sample_random_instance_witness(S, ck).unwrap();
    let ro_consts = ROConstants::<E>::default();

    // produce an NIFS with (r_W, r_U) as the second incoming witness-instance pair
    c.bench_function(&format!("nova_nifs_{name}_{num_cons}"), |b| {
      b.iter(|| {
        // commit to R1CS witness
        let comm_W = W.commit(ck);

        // make an R1CS instance
        let U = R1CSInstance::new(S, &comm_W, x).unwrap();

        let res = NovaNIFS::prove(ck, &ro_consts, &pp_digest, S, &r_U, &r_W, &U, W);
        assert!(res.is_ok());
      })
    });
  }

  #[test]
  fn bench_nifs_simple() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    let num_cons = 1024;
    let (S, ck, W, w, x) = generate_sample_r1cs::<E>(num_cons); // W is R1CSWitness, w is a vector of u8, x is a vector of field elements
    bench_nifs_inner(&mut criterion, "simple", &S, &ck, &W, &w, &x);
  }

  #[test]
  fn bench_nifs_sha256() {
    type E = Bn256EngineKZG;

    let mut criterion = Criterion::default();
    for len in [32, 64].iter() {
      let (S, ck, W, w, x) = generarate_sha_r1cs::<E>(*len);
      bench_nifs_inner(&mut criterion, "sha256", &S, &ck, &W, &w, &x);
    }
  }
}
