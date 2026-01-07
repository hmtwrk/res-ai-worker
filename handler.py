import runpod
import torch
import base64
import io
import os
import sys
from diffusers import StableAudioPipeline
from huggingface_hub import snapshot_download

# --- 1. PROOF OF LIFE LOGGING ---
# This prints immediately when the container starts
print("--- 🚀 WORKER STARTING UP 🚀 ---", file=sys.stderr, flush=True)

REPO_ID = "pillowcushion/res-ai"
HF_TOKEN = os.environ.get("HF_TOKEN")

pipe = None

def load_model():
    global pipe
    print("--- 📥 Downloading Model... ---", file=sys.stderr, flush=True)
    
    # Download the entire repo (safetensors + config.json)
    model_folder = snapshot_download(repo_id=REPO_ID, token=HF_TOKEN)
    
    # Paths
    checkpoint = f"{model_folder}/model.safetensors"
    config_file = f"{model_folder}/config.json"
    
    print(f"--- 🔍 Loading from: {checkpoint} ---", file=sys.stderr, flush=True)
    
    # Load with EXPLICIT config path
    pipe = StableAudioPipeline.from_single_file(
        checkpoint, 
        config=config_file,
        torch_dtype=torch.float16
    )
    
    pipe = pipe.to("cuda")
    print("--- ✅ Model Loaded Successfully! ---", file=sys.stderr, flush=True)

def handler(job):
    global pipe
    print("--- ⚡ Request Received ---", file=sys.stderr, flush=True)
    
    if pipe is None:
        load_model()

    job_input = job["input"]
    prompt = job_input.get("prompt", "Cinematic drone")
    duration = job_input.get("duration", 10)
    steps = job_input.get("steps", 50)

    audio = pipe(prompt, num_inference_steps=steps, audio_end_in_s=duration).audios

    buffer = io.BytesIO()
    import scipy.io.wavfile as wav
    wav.write(buffer, 44100, audio[0].T.float().cpu().numpy())
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    print("--- 📤 Sending Audio Back ---", file=sys.stderr, flush=True)
    return {"audio_base64": audio_base64}

runpod.serverless.start({"handler": handler})
