#!/usr/bin/env python3
"""
Workspace Release Validation Script

Validates that all modules in the workspace are ready for release by checking:
- Version consistency across workspace
- Documentation completeness  
- Test coverage
- No broken dependencies
- Changelog entries
"""

import os
import sys
import json
import toml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class WorkspaceValidator:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(self) -> bool:
        """Run all validation checks"""
        print("🔍 Validating SciRS2 workspace for release...")
        
        self.check_workspace_structure()
        self.check_version_consistency()
        self.check_documentation()
        self.check_dependencies()
        self.check_changelogs()
        self.check_tests()
        
        self.print_results()
        return len(self.errors) == 0
    
    def check_workspace_structure(self):
        """Validate workspace structure"""
        print("📁 Checking workspace structure...")
        
        workspace_toml = self.workspace_root / "Cargo.toml"
        if not workspace_toml.exists():
            self.errors.append("Workspace Cargo.toml not found")
            return
            
        try:
            with open(workspace_toml) as f:
                workspace_config = toml.load(f)
                
            if "workspace" not in workspace_config:
                self.errors.append("Not a valid workspace Cargo.toml")
                return
                
            members = workspace_config["workspace"].get("members", [])
            if not members:
                self.errors.append("No workspace members found")
                return
                
            print(f"   ✓ Found {len(members)} workspace members")
            
            # Check that all members exist
            for member in members:
                member_path = self.workspace_root / member
                if not member_path.exists():
                    self.errors.append(f"Workspace member not found: {member}")
                elif not (member_path / "Cargo.toml").exists():
                    self.errors.append(f"Member Cargo.toml not found: {member}")
                    
        except Exception as e:
            self.errors.append(f"Error reading workspace Cargo.toml: {e}")
    
    def check_version_consistency(self):
        """Check version consistency across workspace"""
        print("🔢 Checking version consistency...")
        
        workspace_toml = self.workspace_root / "Cargo.toml"
        try:
            with open(workspace_toml) as f:
                workspace_config = toml.load(f)
                
            workspace_version = workspace_config["workspace"]["package"].get("version")
            if not workspace_version:
                self.errors.append("Workspace version not found")
                return
                
            print(f"   📌 Workspace version: {workspace_version}")
            
            members = workspace_config["workspace"].get("members", [])
            inconsistent_versions = []
            
            for member in members:
                member_toml = self.workspace_root / member / "Cargo.toml"
                if member_toml.exists():
                    with open(member_toml) as f:
                        member_config = toml.load(f)
                    
                    member_version = member_config["package"].get("version")
                    if member_version and member_version.get("workspace") != True:
                        # Member has its own version instead of using workspace
                        if member_version != workspace_version:
                            inconsistent_versions.append((member, member_version))
            
            if inconsistent_versions:
                for member, version in inconsistent_versions:
                    self.errors.append(f"Version mismatch in {member}: {version} != {workspace_version}")
            else:
                print("   ✓ All versions consistent")
                
        except Exception as e:
            self.errors.append(f"Error checking versions: {e}")
    
    def check_documentation(self):
        """Check documentation completeness"""
        print("📚 Checking documentation...")
        
        # Check README files
        readme_files = list(self.workspace_root.glob("**/README.md"))
        print(f"   📄 Found {len(readme_files)} README files")
        
        # Check for main README
        main_readme = self.workspace_root / "README.md"
        if not main_readme.exists():
            self.errors.append("Main README.md not found")
        
        # Check API documentation can be generated
        try:
            result = subprocess.run(
                ["cargo", "doc", "--workspace", "--all-features", "--no-deps"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                self.errors.append(f"Documentation generation failed: {result.stderr}")
            else:
                print("   ✓ API documentation generates successfully")
        except subprocess.TimeoutExpired:
            self.errors.append("Documentation generation timed out")
        except Exception as e:
            self.errors.append(f"Error generating documentation: {e}")
    
    def check_dependencies(self):
        """Check dependency consistency and security"""
        print("🔗 Checking dependencies...")
        
        try:
            # Check for dependency issues
            result = subprocess.run(
                ["cargo", "tree", "--workspace", "--duplicates"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                self.warnings.append(f"Duplicate dependencies found:\n{result.stdout}")
            
            # Security audit
            try:
                result = subprocess.run(
                    ["cargo", "audit"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    self.errors.append(f"Security audit failed: {result.stdout}")
                else:
                    print("   ✓ Security audit passed")
            except FileNotFoundError:
                self.warnings.append("cargo-audit not installed, skipping security check")
                
        except Exception as e:
            self.errors.append(f"Error checking dependencies: {e}")
    
    def check_changelogs(self):
        """Check changelog entries"""
        print("📝 Checking changelogs...")
        
        changelog_files = []
        for name in ["CHANGELOG.md", "CHANGES.md", "HISTORY.md"]:
            changelog = self.workspace_root / name
            if changelog.exists():
                changelog_files.append(changelog)
        
        if not changelog_files:
            self.warnings.append("No changelog file found")
        else:
            print(f"   📋 Found {len(changelog_files)} changelog files")
            
            # Check if latest version is documented
            workspace_toml = self.workspace_root / "Cargo.toml"
            try:
                with open(workspace_toml) as f:
                    workspace_config = toml.load(f)
                    
                workspace_version = workspace_config["workspace"]["package"].get("version")
                if workspace_version:
                    for changelog in changelog_files:
                        with open(changelog) as f:
                            content = f.read()
                        if workspace_version not in content:
                            self.warnings.append(f"Version {workspace_version} not found in {changelog.name}")
                        else:
                            print(f"   ✓ Version documented in {changelog.name}")
            except Exception as e:
                self.warnings.append(f"Error checking changelog: {e}")
    
    def check_tests(self):
        """Check test coverage and status"""
        print("🧪 Checking tests...")
        
        try:
            # Run tests to ensure they pass
            result = subprocess.run(
                ["cargo", "test", "--workspace", "--all-features"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                self.errors.append(f"Tests failed: {result.stderr}")
            else:
                # Parse test output for statistics
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "test result:" in line:
                        print(f"   📊 {line.strip()}")
                        
                print("   ✓ All tests pass")
                
        except subprocess.TimeoutExpired:
            self.errors.append("Tests timed out")
        except Exception as e:
            self.errors.append(f"Error running tests: {e}")
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*60)
        print("🎯 VALIDATION RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED - READY FOR RELEASE!")
        elif not self.errors:
            print("\n✅ VALIDATION PASSED WITH WARNINGS")
        else:
            print("\n❌ VALIDATION FAILED - CANNOT RELEASE")
        
        print("="*60)

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        workspace_root = Path(sys.argv[1])
    else:
        workspace_root = Path.cwd()
    
    if not workspace_root.exists():
        print(f"❌ Workspace root not found: {workspace_root}")
        sys.exit(1)
    
    validator = WorkspaceValidator(workspace_root)
    success = validator.validate()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()