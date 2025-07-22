use std::fs;
use std::io::{self, Write};
use std::path::Path;
use walkdir::WalkDir;

fn main() -> io::Result<()> {
    println!("Fixing all double underscores in scirs2-special module...");
    
    let mut fixed_count = 0;
    let mut total_files = 0;
    
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
        .filter(|e| !e.path().to_str().unwrap_or("").contains("target"))
    {
        let path = entry.path();
        total_files += 1;
        
        // Read file content
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to read {}: {}", path.display(), e);
                continue;
            }
        };
        
        // Replace all double underscore patterns
        let mut new_content = content.clone();
        new_content = new_content.replace("scirs2__special", "scirs2_special");
        new_content = new_content.replace("scirs2__spatial", "scirs2_spatial");
        new_content = new_content.replace("scirs2__datasets", "scirs2_datasets");
        new_content = new_content.replace("scirs2__core", "scirs2_core");
        new_content = new_content.replace("num__complex", "num_complex");
        new_content = new_content.replace("serde__json", "serde_json");
        
        // Check if any changes were made
        if new_content != content {
            // Write the fixed content back
            match fs::write(&path, new_content) {
                Ok(_) => {
                    println!("✓ Fixed: {}", path.display());
                    fixed_count += 1;
                }
                Err(e) => {
                    eprintln!("Failed to write {}: {}", path.display(), e);
                }
            }
        }
    }
    
    println!("\nSummary:");
    println!("Total files scanned: {}", total_files);
    println!("Files fixed: {}", fixed_count);
    println!("✅ Done! All double underscores have been fixed.");
    
    Ok(())
}