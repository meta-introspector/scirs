#!/usr/bin/env python3
"""
Version Consistency Checker

Ensures all modules in the workspace have consistent versions and dependencies.
"""

import os
import sys
import toml
from pathlib import Path
from typing import Dict, List, Set

def find_workspace_members(workspace_root: Path) -> List[str]:
    """Find all workspace members"""
    workspace_toml = workspace_root / "Cargo.toml"
    
    with open(workspace_toml) as f:
        config = toml.load(f)
    
    return config.get("workspace", {}).get("members", [])

def get_package_info(package_path: Path) -> Dict:
    """Get package information from Cargo.toml"""
    cargo_toml = package_path / "Cargo.toml"
    
    if not cargo_toml.exists():
        return {}
    
    with open(cargo_toml) as f:
        config = toml.load(f)
    
    return config.get("package", {})

def check_version_consistency(workspace_root: Path) -> bool:
    """Check version consistency across workspace"""
    print("🔍 Checking version consistency across workspace...")
    
    # Get workspace version
    workspace_toml = workspace_root / "Cargo.toml"
    with open(workspace_toml) as f:
        workspace_config = toml.load(f)
    
    workspace_version = workspace_config.get("workspace", {}).get("package", {}).get("version")
    if not workspace_version:
        print("❌ No workspace version found")
        return False
    
    print(f"📌 Workspace version: {workspace_version}")
    
    # Check all members
    members = find_workspace_members(workspace_root)
    issues = []
    
    for member in members:
        member_path = workspace_root / member
        package_info = get_package_info(member_path)
        
        if not package_info:
            issues.append(f"No package info found for {member}")
            continue
        
        package_version = package_info.get("version")
        
        # Check if using workspace version
        if isinstance(package_version, dict) and package_version.get("workspace"):
            print(f"   ✓ {member}: using workspace version")
        elif package_version == workspace_version:
            print(f"   ✓ {member}: version {package_version} matches workspace")
        else:
            issues.append(f"{member}: version {package_version} != workspace {workspace_version}")
    
    if issues:
        print("\n❌ Version inconsistencies found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ All versions consistent")
    return True

def check_dependency_consistency(workspace_root: Path) -> bool:
    """Check dependency version consistency"""
    print("\n🔗 Checking dependency consistency...")
    
    members = find_workspace_members(workspace_root)
    all_deps = {}  # dep_name -> {version -> [packages]}
    
    for member in members:
        member_path = workspace_root / member
        cargo_toml = member_path / "Cargo.toml"
        
        if not cargo_toml.exists():
            continue
        
        with open(cargo_toml) as f:
            config = toml.load(f)
        
        # Check dependencies
        for dep_section in ["dependencies", "dev-dependencies", "build-dependencies"]:
            deps = config.get(dep_section, {})
            
            for dep_name, dep_info in deps.items():
                if dep_name.startswith("scirs2-"):
                    continue  # Skip workspace dependencies
                
                if isinstance(dep_info, str):
                    version = dep_info
                elif isinstance(dep_info, dict):
                    version = dep_info.get("version", "no-version")
                    if dep_info.get("workspace"):
                        version = "workspace"
                else:
                    version = "unknown"
                
                if dep_name not in all_deps:
                    all_deps[dep_name] = {}
                if version not in all_deps[dep_name]:
                    all_deps[dep_name][version] = []
                
                all_deps[dep_name][version].append(member)
    
    # Find inconsistencies
    inconsistencies = []
    for dep_name, versions in all_deps.items():
        if len(versions) > 1:
            inconsistencies.append((dep_name, versions))
    
    if inconsistencies:
        print("⚠️  Dependency version inconsistencies found:")
        for dep_name, versions in inconsistencies:
            print(f"   📦 {dep_name}:")
            for version, packages in versions.items():
                print(f"      {version}: {', '.join(packages)}")
        return False
    
    print("✅ All dependencies consistent")
    return True

def check_workspace_dependency_usage(workspace_root: Path) -> bool:
    """Check that workspace dependencies are used properly"""
    print("\n🏗️  Checking workspace dependency usage...")
    
    # Get workspace dependencies
    workspace_toml = workspace_root / "Cargo.toml"
    with open(workspace_toml) as f:
        workspace_config = toml.load(f)
    
    workspace_deps = workspace_config.get("workspace", {}).get("dependencies", {})
    members = find_workspace_members(workspace_root)
    
    issues = []
    
    for member in members:
        member_path = workspace_root / member
        cargo_toml = member_path / "Cargo.toml"
        
        if not cargo_toml.exists():
            continue
        
        with open(cargo_toml) as f:
            config = toml.load(f)
        
        # Check if member uses workspace dependencies properly
        for dep_section in ["dependencies", "dev-dependencies", "build-dependencies"]:
            deps = config.get(dep_section, {})
            
            for dep_name, dep_info in deps.items():
                if dep_name in workspace_deps:
                    # Should use workspace = true
                    if isinstance(dep_info, dict):
                        if not dep_info.get("workspace"):
                            issues.append(f"{member}: {dep_name} should use workspace dependency")
                    else:
                        issues.append(f"{member}: {dep_name} should use workspace dependency")
    
    if issues:
        print("⚠️  Workspace dependency usage issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Workspace dependencies used correctly")
    return True

def main():
    """Main entry point"""
    workspace_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    if not workspace_root.exists():
        print(f"❌ Workspace root not found: {workspace_root}")
        sys.exit(1)
    
    print(f"🏗️  Checking workspace at: {workspace_root}")
    print("="*60)
    
    success = True
    success &= check_version_consistency(workspace_root)
    success &= check_dependency_consistency(workspace_root)
    success &= check_workspace_dependency_usage(workspace_root)
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL VERSION CHECKS PASSED")
    else:
        print("❌ VERSION CHECK FAILURES DETECTED")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()