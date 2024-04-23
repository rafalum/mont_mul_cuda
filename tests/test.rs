use mont_mul_cuda::*;
use mont_mul_cuda::utils::*;

#[test]
fn test_montmul_raw() {
    // TODO: Write actual test suite
    let num_points = 2;
    let points = generate_points(num_points);

    print_point(points[0]);

    let storage_r: Vec<Storage> = montmul_era_wrapper(&points.as_slice(), num_points);

    print_point(storage_r[0]);
}