#!/usr/bin/env python
"""Rename existing LIBERO video directories to include task descriptions.

This script renames directories like:
  libero_spatial_0 -> libero_spatial_0_Pick_up_the_red_block_and_place_it_in_the_front_of_the_blue_block
"""

import re
from pathlib import Path

from libero.libero import benchmark


def sanitize_task_description(task_desc: str, max_length: int = 200) -> str:
    """Sanitize task description for use in filesystem paths."""
    # Remove newlines and extra whitespace
    sanitized = ' '.join(task_desc.split())
    # Replace spaces and special characters with underscores, keep alphanumeric and hyphens
    sanitized = re.sub(r'[^\w\s-]', '', sanitized)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    # Remove leading/trailing underscores and limit length (200 chars should be enough for most tasks)
    sanitized = sanitized.strip('_')[:max_length]
    # Ensure it's not empty
    if not sanitized:
        sanitized = "task"
    return sanitized


def get_task_description(suite_name: str, task_id: int) -> str:
    """Get the language instruction for a specific task."""
    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise ValueError(f"Unknown suite: {suite_name}")
    suite = bench[suite_name]()
    if task_id >= len(suite.tasks):
        raise ValueError(f"Task ID {task_id} out of range for {suite_name} (max: {len(suite.tasks)-1})")
    task = suite.tasks[task_id]
    return task.language


def rename_video_directories(videos_dir: Path, suite_name: str = "libero_spatial"):
    """Rename video directories to include task descriptions."""
    if not videos_dir.exists():
        print(f"Directory {videos_dir} does not exist!")
        return
    
    # Pattern to match: suite_name_taskid (e.g., libero_spatial_0)
    # Also matches already-renamed: suite_name_taskid_description
    pattern = re.compile(rf"^{re.escape(suite_name)}_(\d+)(?:_.*)?$")
    
    directories_to_rename = []
    for item in videos_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                task_id = int(match.group(1))
                directories_to_rename.append((item, task_id))
    
    if not directories_to_rename:
        print(f"No directories matching pattern '{suite_name}_*' found in {videos_dir}")
        return
    
    print(f"Found {len(directories_to_rename)} directories to rename:")
    for old_dir, task_id in sorted(directories_to_rename, key=lambda x: x[1]):
        try:
            # Get task description
            task_desc = get_task_description(suite_name, task_id)
            sanitized_desc = sanitize_task_description(task_desc)
            
            # Create new directory name
            new_name = f"{suite_name}_{task_id}_{sanitized_desc}"
            new_path = old_dir.parent / new_name
            
            # Check if directory already has the correct name
            if old_dir.name == new_name:
                print(f"  SKIP: {old_dir.name} (already has correct name)")
                continue
            
            # Check if new name already exists (different directory)
            if new_path.exists() and new_path != old_dir:
                print(f"  SKIP: {old_dir.name} -> {new_name} (target already exists)")
                continue
            
            # Rename
            old_dir.rename(new_path)
            print(f"  ✓ {old_dir.name} -> {new_name}")
            print(f"    Task: {task_desc}")
        except Exception as e:
            print(f"  ✗ ERROR renaming {old_dir.name}: {e}")
    
    print(f"\nDone! Renamed {len([d for d, _ in directories_to_rename])} directories.")


if __name__ == "__main__":
    import sys
    
    # Default path
    videos_dir = Path("eval_logs/videos")
    
    # Allow custom path via command line
    if len(sys.argv) > 1:
        videos_dir = Path(sys.argv[1])
    
    # Allow custom suite name
    suite_name = "libero_spatial"
    if len(sys.argv) > 2:
        suite_name = sys.argv[2]
    
    print(f"Renaming video directories in: {videos_dir.absolute()}")
    print(f"Suite: {suite_name}\n")
    
    rename_video_directories(videos_dir, suite_name)

