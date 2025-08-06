fn main() {
    let n = 100;
    let chunk = 512_usize.min(n / num_cpus::get());
    let overlap = 5 / 2; // half_window for window_size=5
    
    println\!("n = {}", n);
    println\!("original chunk = {}", chunk);
    println\!("overlap = {}", overlap);
    
    // Original calculation
    if chunk > overlap {
        let old_calc = (n + chunk - overlap - 1) / (chunk - overlap);
        println\!("old n_chunks would be = {}", old_calc);
    } else {
        println\!("old calculation would underflow\!");
    }
    
    // New calculation
    let effective_chunk = chunk.saturating_sub(overlap).max(1);
    let new_calc = (n + effective_chunk - 1) / effective_chunk;
    println\!("effective_chunk = {}", effective_chunk);
    println\!("new n_chunks = {}", new_calc);
}
EOF < /dev/null
