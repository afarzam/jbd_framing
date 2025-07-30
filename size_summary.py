#!/usr/bin/env python3
"""
Simple script to analyze repository size and categorize components.
"""

import os
import sys
from pathlib import Path

def humanize_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def get_dir_size(path):
    """Calculate total size of a directory recursively."""
    total_size = 0
    file_count = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, FileNotFoundError):
        pass
    
    return total_size, file_count

def analyze_repository():
    """Analyze the repository and categorize components."""
    repo_path = Path('.')
    
    # Categories to track
    categories = {
        'code': {
            'extensions': ['.py', '.ipynb', '.sh', '.md', '.txt', '.yml', '.yaml', '.json', '.cfg', '.conf'],
            'size': 0,
            'files': 0
        },
        'data': {
            'dirs': ['data', 'cache', 'train_test', 'benchmarks'],
            'size': 0,
            'files': 0
        },
        'models': {
            'dirs': ['models', 'checkpoints'],
            'size': 0,
            'files': 0
        },
        'output': {
            'dirs': ['output', 'results', 'logs', 'experiments'],
            'size': 0,
            'files': 0
        },
        'configs': {
            'dirs': ['configs', 'slurm'],
            'size': 0,
            'files': 0
        },
        'git': {
            'dirs': ['.git'],
            'size': 0,
            'files': 0
        },
        'other': {
            'size': 0,
            'files': 0
        }
    }
    
    total_size = 0
    total_files = 0
    
    print("🔍 Analyzing repository size...")
    print("=" * 50)
    
    # Analyze each file and directory
    for item in repo_path.iterdir():
        if item.is_file():
            try:
                size = item.stat().st_size
                total_size += size
                total_files += 1
                
                # Categorize file
                categorized = False
                
                # Check if it's code
                if item.suffix.lower() in categories['code']['extensions']:
                    categories['code']['size'] += size
                    categories['code']['files'] += 1
                    categorized = True
                
                if not categorized:
                    categories['other']['size'] += size
                    categories['other']['files'] += 1
                    
            except (OSError, FileNotFoundError):
                continue
                
        elif item.is_dir():
            try:
                size, file_count = get_dir_size(item)
                total_size += size
                total_files += file_count
                
                # Categorize directory
                categorized = False
                for cat_name, cat_info in categories.items():
                    if cat_name != 'code' and cat_name != 'other':
                        if item.name in cat_info['dirs']:
                            cat_info['size'] += size
                            cat_info['files'] += file_count
                            categorized = True
                            break
                
                if not categorized:
                    categories['other']['size'] += size
                    categories['other']['files'] += file_count
                    
            except (OSError, FileNotFoundError):
                continue
    
    # Print results
    print(f"📊 REPOSITORY SIZE SUMMARY")
    print(f"Total Size: {humanize_size(total_size)}")
    print(f"Total Files: {total_files:,}")
    print()
    
    print("📁 BREAKDOWN BY CATEGORY:")
    print("-" * 40)
    
    for category, info in categories.items():
        if info['size'] > 0:
            percentage = (info['size'] / total_size) * 100
            print(f"{category.upper():12} | {humanize_size(info['size']):>10} | {info['files']:>6} files | {percentage:>5.1f}%")
    
    print()
    print("📋 DETAILED BREAKDOWN:")
    print("-" * 40)
    
    # Code breakdown
    if categories['code']['size'] > 0:
        print(f"\n💻 CODE FILES:")
        print(f"   Size: {humanize_size(categories['code']['size'])}")
        print(f"   Files: {categories['code']['files']:,}")
    
    # Data breakdown
    if categories['data']['size'] > 0:
        print(f"\n📊 DATA:")
        print(f"   Size: {humanize_size(categories['data']['size'])}")
        print(f"   Files: {categories['data']['files']:,}")
    
    # Models breakdown
    if categories['models']['size'] > 0:
        print(f"\n🤖 MODELS & CHECKPOINTS:")
        print(f"   Size: {humanize_size(categories['models']['size'])}")
        print(f"   Files: {categories['models']['files']:,}")
    
    # Output breakdown
    if categories['output']['size'] > 0:
        print(f"\n📤 OUTPUT & RESULTS:")
        print(f"   Size: {humanize_size(categories['output']['size'])}")
        print(f"   Files: {categories['output']['files']:,}")
    
    # Configs breakdown
    if categories['configs']['size'] > 0:
        print(f"\n⚙️  CONFIGS:")
        print(f"   Size: {humanize_size(categories['configs']['size'])}")
        print(f"   Files: {categories['configs']['files']:,}")
    
    # Git breakdown
    if categories['git']['size'] > 0:
        print(f"\n🔧 GIT:")
        print(f"   Size: {humanize_size(categories['git']['size'])}")
        print(f"   Files: {categories['git']['files']:,}")
    
    # Other breakdown
    if categories['other']['size'] > 0:
        print(f"\n📁 OTHER:")
        print(f"   Size: {humanize_size(categories['other']['size'])}")
        print(f"   Files: {categories['other']['files']:,}")

if __name__ == "__main__":
    analyze_repository() 