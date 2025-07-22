use rand::SeedableRng;
use rand::rngs::StdRng;

#[allow(unused_imports)]
#[allow(dead_code)]
fn main() {
    // Test from_entropy
    let _rng1 = StdRng::from_entropy();

    // Test from_seed
    let _rng2 = StdRng::from_seed([0u8; 32]);

    // Test rng
    let mut _rng3 = rand::rng();

    println!("Tests passed!");
}
