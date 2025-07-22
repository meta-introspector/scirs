use std::fs;
use std::path::Path;
use std::io::{self, Write};

fn fix_file(path: &Path) -> io::Result<bool> {
    let content = fs::read_to_string(path)?;
    
    if content.contains("scirs2__") {
        let fixed_content = content
            .replace("scirs2__special", "scirs2_special")
            .replace("scirs2__spatial", "scirs2_spatial")
            .replace("scirs2__core", "scirs2_core");
        
        if fixed_content != content {
            fs::write(path, fixed_content)?;
            return Ok(true);
        }
    }
    
    Ok(false)
}

fn process_directory(dir: &Path) -> io::Result<(usize, usize)> {
    let mut files_checked = 0;
    let mut files_fixed = 0;
    
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() && !path.to_str().unwrap_or("").contains("target") {
            let (sub_checked, sub_fixed) = process_directory(&path)?;
            files_checked += sub_checked;
            files_fixed += sub_fixed;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            files_checked += 1;
            match fix_file(&path) {
                Ok(true) => {
                    files_fixed += 1;
                    println!("Fixed: {}", path.display());
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", path.display(), e);
                }
                _ => {}
            }
        }
    }
    
    Ok((files_checked, files_fixed))
}

fn main() -> io::Result<()> {
    let target_dir = Path::new(".");
    
    println!("Fixing double underscores in {}", target_dir.display());
    println!("=========================================");
    
    let (files_checked, files_fixed) = process_directory(target_dir)?;
    
    println!("\nSummary:");
    println!("Files checked: {}", files_checked);
    println!("Files fixed: {}", files_fixed);
    
    if files_fixed > 0 {
        println!("\n✅ Successfully fixed all double underscores!");
    } else {
        println!("\n✅ No double underscores found.");
    }
    
    Ok(())
}