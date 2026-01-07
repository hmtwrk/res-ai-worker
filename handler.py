import torch
from diffusers import AudioLDMPipeline
import base64
import io
import sys
import runpod
import scipy.io.wavfile as wav

# 1. Global variable to hold the model in memory (Warm Worker)
pipe = None

def load_model():
    global pipe
    print("--- 🔄 Loading Model from Disk (/model) ---", file=sys.stderr, flush=True)

    # --- THE CRITICAL CHANGE ---
    # We load from "/model" because that is where builder.py saved it.
    # local_files_only=True ensures we don't try to hit the internet.
    pipe = AudioLDMPipeline.from_pretrained(
        "/model",               
        torch_dtype=torch.float16,
        local_files_only=True   
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    print("--- ✅ Model Loaded Successfully ---", file=sys.stderr, flush=True)

def handler(job):
    global pipe
    print("--- ⚡ Request Received ---", file=sys.stderr, flush=True)
    
    # 2. If this is a Cold Start, load the model now
    if pipe is None:
        load_model()

    # 3. Parse Inputs
    job_input = job["input"]
    prompt = job_input.get("prompt", "Cinematic drone")
    duration = job_input.get("duration", 10)
    
    # Read Steps & CFG (with defaults)
    steps = job_input.get("steps", 50)        
    cfg_scale = job_input.get("cfg_scale", 7.0) 

    # 4. Generate
    audio = pipe(
        prompt, 
        num_inference_steps=steps, 
        guidance_scale=cfg_scale, 
        audio_end_in_s=duration
    ).audios

    # 5. Encode to Base64
    buffer = io.BytesIO()
    
    # Note: AudioLDM standard output is usually 16000Hz. 
    # If your audio sounds fast/high-pitched, change 44100 to 16000 below.
    wav.write(buffer, 44100, audio[0].T.float().cpu().numpy())
    
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    print("--- 📤 Sending Audio Back ---", file=sys.stderr, flush=True)
    return {"audio_base64": audio_base64}

# Start the RunPod Serverless Worker
runpod.serverless.start({"handler": handler})
