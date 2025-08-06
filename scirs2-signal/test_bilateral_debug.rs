fn main() {
    let n = 100;
    let window_size = 5;
    let half_window = window_size / 2;
    let num_cpus = 8; // typical value
    let chunk = 512_usize.min(n / num_cpus);
    let overlap = half_window;
    
    println\!("n = {}", n);
    println\!("window_size = {}", window_size);
    println\!("half_window = {}", half_window);
    println\!("chunk = {}", chunk);
    println\!("overlap = {}", overlap);
    
    let effective_chunk_size = if chunk > overlap { chunk - overlap } else { n };
    let n_chunks = if effective_chunk_size >= n { 1 } else { (n + effective_chunk_size - 1) / effective_chunk_size };
    
    println\!("effective_chunk_size = {}", effective_chunk_size);
    println\!("n_chunks = {}", n_chunks);
    
    // Simulate chunk processing
    let mut total_output = 0;
    for i in 0..n_chunks {
        let start = i * effective_chunk_size;
        let end = (start + effective_chunk_size).min(n);
        let chunk_len = end - start;
        println\!("Chunk {}: start={}, end={}, len={}", i, start, end, chunk_len);
        total_output += chunk_len;
    }
    println\!("Total output length: {}", total_output);
}
