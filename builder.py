# builder.py
from huggingface_hub import snapshot_download

# This downloads the model to the folder /model inside the image
snapshot_download(
    repo_id="YOUR_MODEL_REPO_ID", 
    local_dir="/model"
)
