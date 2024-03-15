use rand::{rngs::ThreadRng, Rng};
use num_bigint::BigUint;
use std::iter;

use super::Storage;

fn generate_random_biguint(rng: &mut ThreadRng, bits: usize) -> BigUint {
    let bytes = (bits + 7) / 8; // Calculate the number of bytes needed
    let mut random_bytes: Vec<u8> = iter::repeat_with(|| rng.gen::<u8>())
                                    .take(bytes)
                                    .collect();

    // Ensure the number fits exactly `bits` bits by masking excess bits in the last byte
    let excess_bits = bytes * 8 - bits;
    if excess_bits > 0 {
        let mask = 0xFF >> excess_bits;
        if let Some(last) = random_bytes.last_mut() {
            *last &= mask;
        }
    }

    BigUint::from_bytes_le(&random_bytes)
}

pub fn biguint_to_storage(num: BigUint) -> Storage {
    let mut limbs = [0u32; 12];
    let bytes = num.to_bytes_le();
    
    // Each limb is 4 bytes (u32), so we iterate over the bytes in chunks of 4,
    // converting each chunk into a u32 and storing it in the limbs array.
    for (i, chunk) in bytes.chunks(4).enumerate() {
        let mut limb = [0u8; 4];
        for (j, &byte) in chunk.iter().enumerate() {
            limb[j] = byte;
        }
        limbs[i] = u32::from_le_bytes(limb);
    }

    Storage { limbs }
}

pub fn storage_to_biguint(storage: Storage) -> BigUint {
    let mut bytes = Vec::new();
    for &limb in storage.limbs.iter() {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

pub fn generate_points() -> (Storage, Storage) {
    let mut rng = rand::thread_rng();
    let number1 = generate_random_biguint(&mut rng, 381);
    let number2 = generate_random_biguint(&mut rng, 381);

    println!("Number 1: {}", number1);
    println!("Number 2: {}", number2);

    let storage1 = biguint_to_storage(number1);
    let storage2 = biguint_to_storage(number2);

    (storage1, storage2)
}