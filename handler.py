import runpod
import torch
import base64
import io
import os
import sys
from diffusers import StableAudioPipeline

# --- 1. PROOF OF LIFE LOGGING ---
print("--- 🚀 WORKER STARTING UP 🚀 ---", file=sys.stderr, flush=True)

# 2. USE OFFICIAL REPO 
# This provides the necessary folder structure (model_index.json) automatically
REPO_ID = "stabilityai/stable-audio-open-1.0"
HF_TOKEN = os.environ.get("HF_TOKEN")

pipe = None

def load_model():
    global pipe
    print("--- 📥 Downloading Model... ---", file=sys.stderr, flush=True)
    
    try:
        # 3. Load using from_pretrained (The standard, working method)
        pipe = StableAudioPipeline.from_pretrained(
            REPO_ID, 
            torch_dtype=torch.float16,
            token=HF_TOKEN
        )
        pipe = pipe.to("cuda")
        print("--- ✅ Model Loaded Successfully! ---", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"--- ❌ Model Load Failed: {e} ---", file=sys.stderr, flush=True)
        # Verify user has accepted terms
        print("--- HINT: Ensure you accepted the agreement at https://huggingface.co/stabilityai/stable-audio-open-1.0 ---", file=sys.stderr, flush=True)
        raise e

def handler(job):
    global pipe
    print("--- ⚡ Request Received ---", file=sys.stderr, flush=True)
    
    if pipe is None:
        load_model()

    job_input = job["input"]
    prompt = job_input.get("prompt", "Cinematic drone")
    duration = job_input.get("duration", 10)
    steps = job_input.get("steps", 50)

    # Generate
    audio = pipe(prompt, num_inference_steps=steps, audio_end_in_s=duration).audios

    # Encode
    buffer = io.BytesIO()
    import scipy.io.wavfile as wav
    wav.write(buffer, 44100, audio[0].T.float().cpu().numpy())
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    print("--- 📤 Sending Audio Back ---", file=sys.stderr, flush=True)
    return {"audio_base64": audio_base64}

runpod.serverless.start({"handler": handler})
