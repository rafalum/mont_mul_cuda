use criterion::{criterion_group, criterion_main, Criterion};

use mont_mul_cuda::*;
use mont_mul_cuda::utils::*;


fn criterion_benchmark(c: &mut Criterion) {

    // Sample two random G1 points in projective form and convert to affine
    let num_points = 16777216;
    let points = generate_points(num_points);

    let mut group = c.benchmark_group("CUDA");
    group.sample_size(10);


    // Ensure batches is defined, assuming it is for demonstration purposes
    let batches = 10; // Example value
    let name = format!("montmul_raw_{}", batches);
    group.bench_function(&name, |b| {
        b.iter(|| {
            // Call the CUDA function with the prepared Storage inputs
            let _result = montmul_raw_wrapper(&points.as_slice(), num_points);
        })
    });

    group.finish();
}

// Define criterion group and main function
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
