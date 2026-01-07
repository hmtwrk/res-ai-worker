# builder.py
from huggingface_hub import snapshot_download

# This downloads the model to the folder /model inside the image
snapshot_download(
    repo_id="pillowcushion/res-ai", 
    local_dir="/model"
)
