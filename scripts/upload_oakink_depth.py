"""
Upload oakink depth data to HuggingFace in batches.
54G, 779 folders -> UCBProject/ProcessedData / depth/ThirdPerson/Oakink
"""
import os
import sys
from huggingface_hub import HfApi

api = HfApi()

local_root = "/home/lyh/Project/V2AP/data_hub/ProcessedData/depth/ThirdPerson/oakink"
repo_id = "UCBProject/ProcessedData"
repo_type = "dataset"
remote_prefix = "depth/ThirdPerson/Oakink"

# Get all top-level subfolders (each is one oakink sequence like A01001_0001_0000)
sequences = sorted(os.listdir(local_root))
total = len(sequences)
print(f"Found {total} sequences to upload.", flush=True)

# Check which already exist on HF to allow resume
try:
    existing = set()
    for f in api.list_repo_files(repo_id=repo_id, repo_type=repo_type):
        # e.g. depth/ThirdPerson/Oakink/A01001_0001_0000/...
        if f.startswith(remote_prefix + "/"):
            parts = f.split("/")
            if len(parts) > 3:
                existing.add(parts[3])  # sequence name
    print(f"Already uploaded: {len(existing)} sequences (will skip).", flush=True)
except Exception as e:
    print(f"Warning: could not list existing files: {e}", flush=True)
    existing = set()

to_upload = [s for s in sequences if s not in existing]
print(f"Remaining to upload: {len(to_upload)} sequences.", flush=True)

for i, seq in enumerate(to_upload):
    local_path = os.path.join(local_root, seq)
    path_in_repo = f"{remote_prefix}/{seq}"
    print(f"[{i+1}/{len(to_upload)}] Uploading {seq} ...", flush=True)
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            commit_message=f"Upload oakink depth: {seq}",
        )
        print(f"  ✅ Done: {seq}", flush=True)
    except Exception as e:
        print(f"  ❌ Failed: {seq} -> {e}", flush=True)

print("🎉 All done!", flush=True)
