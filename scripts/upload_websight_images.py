#!/usr/bin/env python3
"""
Reorganize WebSight images into subdirectories (HF 10K files/dir limit)
and update JSONL paths, then upload in batches.
"""

import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]  # set HF_TOKEN env var before running
REPO_ID = "Mindigenous/MINDI-1.5-training-data"
IMAGES_DIR = Path("data/websight/images")
FILES_PER_DIR = 10000  # max files per directory on HF

# Step 1: Reorganize images into subdirectories
print("=" * 60)
print("  Step 1: Reorganizing images into subdirectories")
print("=" * 60)

all_images = sorted(IMAGES_DIR.glob("*.jpg"))
print(f"Found {len(all_images)} images in flat directory")

if not all_images:
    # Check if already reorganized
    subdirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
    if subdirs:
        total = sum(len(list(d.glob("*.jpg"))) for d in subdirs)
        print(f"Already reorganized into {len(subdirs)} subdirs with {total} total images")
    else:
        print("ERROR: No images found!")
        exit(1)
else:
    for i, img in enumerate(all_images):
        subdir_idx = i // FILES_PER_DIR
        subdir = IMAGES_DIR / f"{subdir_idx:02d}"
        subdir.mkdir(exist_ok=True)
        shutil.move(str(img), str(subdir / img.name))
        if (i + 1) % 10000 == 0:
            print(f"  Moved {i + 1:,} images...")

    subdirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
    for sd in subdirs:
        count = len(list(sd.glob("*.jpg")))
        print(f"  {sd.name}/: {count:,} images")

# Step 2: Update JSONL files with new paths
print(f"\n{'=' * 60}")
print("  Step 2: Updating JSONL paths")
print("=" * 60)

for jsonl_name in ["train.jsonl", "val.jsonl"]:
    jsonl_path = Path("data/websight") / jsonl_name
    if not jsonl_path.exists():
        print(f"  {jsonl_name}: not found, skipping")
        continue

    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    updated = []
    for line in lines:
        entry = json.loads(line)
        old_path = entry["image_path"]
        filename = os.path.basename(old_path)
        num = int(filename.replace("ws_", "").replace(".jpg", ""))
        subdir_idx = num // FILES_PER_DIR
        new_path = f"data/websight/images/{subdir_idx:02d}/{filename}"
        entry["image_path"] = new_path
        updated.append(json.dumps(entry, ensure_ascii=False))

    jsonl_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"  {jsonl_name}: updated {len(updated):,} entries")

# Step 3: Upload to HF
print(f"\n{'=' * 60}")
print("  Step 3: Uploading to HuggingFace")
print("=" * 60)

api = HfApi(token=TOKEN)

# Upload updated JSONL files first
print("\nUploading updated JSONL files...")
for jsonl_name in ["train.jsonl", "val.jsonl"]:
    jsonl_path = Path("data/websight") / jsonl_name
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=f"websight/{jsonl_name}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  {jsonl_name} uploaded")

# Check which subdirs are already uploaded
import time
repo_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))

# Upload each subdirectory separately
subdirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
for i, subdir in enumerate(subdirs):
    count = len(list(subdir.glob("*.jpg")))
    # Check if this subdir is already fully uploaded
    sample_file = f"websight/images/{subdir.name}/{sorted(subdir.glob('*.jpg'))[0].name}"
    if sample_file in repo_files:
        print(f"\nSubdir {subdir.name}/ ({count:,} images) [{i+1}/{len(subdirs)}] — already uploaded, skipping.")
        continue

    for attempt in range(3):
        try:
            print(f"\nUploading subdir {subdir.name}/ ({count:,} images) [{i+1}/{len(subdirs)}] (attempt {attempt+1})...")
            api.upload_folder(
                folder_path=str(subdir),
                path_in_repo=f"websight/images/{subdir.name}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Add WebSight images subdir {subdir.name} ({count} images)",
            )
            print(f"  Subdir {subdir.name} committed!")
            break
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after 3 attempts. Run script again to resume.")

print(f"\n{'=' * 60}")
print("  ALL DONE! All WebSight data uploaded to HF.")
print("=" * 60)
