pub mod utils;


// The Rust equivalent of the CUDA `storage` structure, with total limb count (TLC) value of 8
#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub struct Storage {
    limbs: [u32; 12], // 12 limbs for 384 bits
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes), allow(dead_code))]
extern "C" {
    fn montmul_era(ret: *mut Storage, points: *const Storage, num_points: u32);

    fn montmul_supra(ret: *mut Storage, points: *const Storage, num_points: u32);
}

pub fn montmul_era_wrapper(points: &[Storage], num_points: u32) -> Vec<Storage> {

    // Init vector of size num_points
    let mut results: Vec<Storage> = vec![Storage { limbs: [0; 12] }; (num_points / 2) as usize];

    // Assume proper CUDA initialization and memory management done here...
    unsafe {
        // Launch the kernel with appropriate configuration
        montmul_era(results.as_mut_ptr() as *mut _, points.as_ptr() as *const _, num_points);
    }
    results
}

pub fn montmul_supra_wrapper(points: &[Storage], num_points: u32) -> Vec<Storage> {

    // Init vector of size num_points
    let mut results: Vec<Storage> = vec![Storage { limbs: [0; 12] }; (num_points / 2) as usize];

    // Assume proper CUDA initialization and memory management done here...
    unsafe {
        // Launch the kernel with appropriate configuration
        montmul_supra(results.as_mut_ptr() as *mut _, points.as_ptr() as *const _, num_points);
    }
    results
}