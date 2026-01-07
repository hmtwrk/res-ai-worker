# builder.py
from huggingface_hub import snapshot_download

# Replace "hf_xxxxx" with your ACTUAL token you just copied
HF_TOKEN = "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" 

snapshot_download(
    repo_id="pillowcushion/res-ai",
    local_dir="/model",
    token=HF_TOKEN  # <--- This grants permission to access your private repo
)
