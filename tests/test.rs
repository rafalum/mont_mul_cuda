use mont_mul_cuda::*;
use mont_mul_cuda::utils::*;


#[test]
fn test_montmul_raw() {
    let points = generate_points(2);

    for point in &points {
        print_point(*point);
    }

    let storage_r: Vec<Storage> = montmul_raw_wrapper(&points.as_slice(), 2);

    for point in &storage_r {
        print_point(*point);
    }
}