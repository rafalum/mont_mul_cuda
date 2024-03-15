use mont_mul_cuda::*;
use mont_mul_cuda::utils::*;


#[test]
fn test_montmul_raw() {
    let (storage_a, storage_b) = generate_points();

    let storage_r = montmul_raw_wrapper(&storage_a, &storage_b);

    println!("Result: {}", storage_to_biguint(storage_r));
}
