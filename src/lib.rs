pub mod utils;


// The Rust equivalent of the CUDA `storage` structure, with total limb count (TLC) value of 8
#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub struct Storage {
    limbs: [u32; 12], // 12 limbs for 384 bits
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes), allow(dead_code))]
extern "C" {
    fn montmul_raw(points: *const Storage, ret: *mut Storage, num_points: u32);
}

pub fn montmul_raw_wrapper(points: &[Storage], num_points: u32) -> Vec<Storage> {

    // Init vector of size num_points
    let mut results: Vec<Storage> = vec![Storage { limbs: [0; 12] }; (num_points / 2) as usize];

    // Assume proper CUDA initialization and memory management done here...
    unsafe {
        // Launch the kernel with appropriate configuration
        montmul_raw(points.as_ptr() as *const _, results.as_mut_ptr() as *mut _, num_points);
    }
    results
}