# builder.py
import os
from huggingface_hub import snapshot_download

# It will look for an environment variable named HF_TOKEN
token = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id="pillowcushion/res-ai",
    local_dir="/model",
    token=token
)
