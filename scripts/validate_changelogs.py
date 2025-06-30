#!/usr/bin/env python3
"""
Changelog Validation Script

Validates that changelog entries exist for the current version and follow proper format.
"""

import os
import sys
import re
import toml
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

class ChangelogValidator:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """Validate all changelogs in the workspace"""
        print("📝 Validating changelogs...")
        
        # Get workspace version
        workspace_version = self.get_workspace_version()
        if not workspace_version:
            return False
        
        print(f"   📌 Checking for version: {workspace_version}")
        
        # Find and validate changelogs
        changelog_files = self.find_changelog_files()
        
        if not changelog_files:
            self.errors.append("No changelog files found")
            return False
        
        for changelog_path in changelog_files:
            self.validate_changelog(changelog_path, workspace_version)
        
        self.print_results()
        return len(self.errors) == 0
    
    def get_workspace_version(self) -> Optional[str]:
        """Get the workspace version from Cargo.toml"""
        workspace_toml = self.workspace_root / "Cargo.toml"
        
        if not workspace_toml.exists():
            self.errors.append("Workspace Cargo.toml not found")
            return None
        
        try:
            with open(workspace_toml) as f:
                config = toml.load(f)
            
            version = config.get("workspace", {}).get("package", {}).get("version")
            if not version:
                self.errors.append("Workspace version not found in Cargo.toml")
                return None
            
            return version
        except Exception as e:
            self.errors.append(f"Error reading workspace version: {e}")
            return None
    
    def find_changelog_files(self) -> List[Path]:
        """Find all changelog files in the workspace"""
        changelog_names = [
            "CHANGELOG.md", "CHANGES.md", "HISTORY.md", "NEWS.md",
            "changelog.md", "changes.md", "history.md", "news.md"
        ]
        
        changelog_files = []
        
        # Check workspace root
        for name in changelog_names:
            changelog = self.workspace_root / name
            if changelog.exists():
                changelog_files.append(changelog)
        
        # Check each module
        if (self.workspace_root / "Cargo.toml").exists():
            try:
                with open(self.workspace_root / "Cargo.toml") as f:
                    config = toml.load(f)
                
                members = config.get("workspace", {}).get("members", [])
                for member in members:
                    member_path = self.workspace_root / member
                    for name in changelog_names:
                        changelog = member_path / name
                        if changelog.exists():
                            changelog_files.append(changelog)
            except:
                pass
        
        return list(set(changelog_files))  # Remove duplicates
    
    def validate_changelog(self, changelog_path: Path, version: str):
        """Validate a specific changelog file"""
        print(f"   📋 Validating {changelog_path.relative_to(self.workspace_root)}")
        
        try:
            with open(changelog_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Error reading {changelog_path}: {e}")
            return
        
        # Check if version is mentioned
        if version not in content:
            self.errors.append(f"Version {version} not found in {changelog_path.name}")
            return
        
        # Validate format
        self.validate_format(changelog_path, content, version)
        self.validate_version_entry(changelog_path, content, version)
    
    def validate_format(self, changelog_path: Path, content: str, version: str):
        """Validate changelog format"""
        lines = content.split('\n')
        
        # Check if it starts with a proper header
        if not lines[0].startswith('#'):
            self.warnings.append(f"{changelog_path.name}: Should start with a markdown header")
        
        # Check for proper version headers
        version_headers = []
        for i, line in enumerate(lines):
            if re.match(r'^#+\s*(?:v?)?' + re.escape(version), line, re.IGNORECASE):
                version_headers.append((i, line))
        
        if not version_headers:
            # Try alternative patterns
            for i, line in enumerate(lines):
                if version in line and any(marker in line.lower() for marker in ['##', '###', 'version', 'release']):
                    version_headers.append((i, line))
        
        if not version_headers:
            self.warnings.append(f"{changelog_path.name}: Version {version} header format could be improved")
        else:
            print(f"      ✓ Found version header: {version_headers[0][1].strip()}")
    
    def validate_version_entry(self, changelog_path: Path, content: str, version: str):
        """Validate the version entry content"""
        lines = content.split('\n')
        
        # Find the version section
        version_start = None
        version_end = None
        
        for i, line in enumerate(lines):
            if version in line and any(marker in line for marker in ['#', '*', '-']):
                version_start = i
                break
        
        if version_start is None:
            return
        
        # Find the end of the version section
        for i in range(version_start + 1, len(lines)):
            line = lines[i]
            if re.match(r'^#+\s*(?:v?)?\d+\.\d+', line):  # Next version
                version_end = i
                break
        
        if version_end is None:
            version_end = len(lines)
        
        # Extract version section
        version_section = '\n'.join(lines[version_start:version_end])
        
        # Check for common sections
        common_sections = ['added', 'changed', 'deprecated', 'removed', 'fixed', 'security']
        has_sections = any(section in version_section.lower() for section in common_sections)
        
        if not has_sections:
            self.warnings.append(f"{changelog_path.name}: Version {version} entry could use standard sections (Added, Changed, Fixed, etc.)")
        
        # Check for date
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ]
        
        has_date = any(re.search(pattern, version_section) for pattern in date_patterns)
        if not has_date:
            self.warnings.append(f"{changelog_path.name}: Version {version} entry missing release date")
        
        # Check for content
        content_lines = [line.strip() for line in version_section.split('\n') if line.strip() and not line.startswith('#')]
        if len(content_lines) < 3:  # Header + minimal content
            self.warnings.append(f"{changelog_path.name}: Version {version} entry seems sparse")
        else:
            print(f"      ✓ Version entry has content ({len(content_lines)} lines)")
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*50)
        print("📝 CHANGELOG VALIDATION RESULTS")
        print("="*50)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHANGELOG CHECKS PASSED!")
        elif not self.errors:
            print("\n✅ CHANGELOG VALIDATION PASSED WITH WARNINGS")
        else:
            print("\n❌ CHANGELOG VALIDATION FAILED")

def main():
    """Main entry point"""
    workspace_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    if not workspace_root.exists():
        print(f"❌ Workspace root not found: {workspace_root}")
        sys.exit(1)
    
    validator = ChangelogValidator(workspace_root)
    success = validator.validate()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()