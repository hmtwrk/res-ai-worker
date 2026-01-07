import runpod
import torch
import base64
import io
import os
from diffusers import StableAudioPipeline
from huggingface_hub import snapshot_download

# 1. SETUP: Define your model info
REPO_ID = "pillowcushion/res-ai"
HF_TOKEN = os.environ.get("HF_TOKEN") # We will set this in RunPod settings later

# 2. GLOBAL VARIABLES: Load model only once (Cold Start)
pipe = None

def load_model():
    global pipe
    print("--- Loading Model... ---")
    
    # 1. Download the repo (which contains model.safetensors and config.json)
    model_folder = snapshot_download(repo_id=REPO_ID, token=HF_TOKEN)
    
    # 2. Point to the specific file
    checkpoint_path = f"{model_folder}/model.safetensors"
    
    # 3. Load using 'from_single_file'
    pipe = StableAudioPipeline.from_single_file(
        checkpoint_path, 
        torch_dtype=torch.float16
    )
    
    pipe = pipe.to("cuda")
    print("--- Model Loaded! ---")

# 3. HANDLER: This runs for every request
def handler(job):
    global pipe
    
    # Load model if it's the first time running
    if pipe is None:
        load_model()

    # Get the input from the API
    job_input = job["input"]
    prompt = job_input.get("prompt", "Cinematic drone")
    duration = job_input.get("duration", 10)
    steps = job_input.get("steps", 50)

    # Generate Audio
    generator = torch.Generator("cuda").manual_seed(42)
    audio = pipe(
        prompt, 
        negative_prompt="Low quality, static, noise",
        num_inference_steps=steps, 
        audio_end_in_s=duration, 
        num_waveforms_per_prompt=1, 
        generator=generator
    ).audios

    # Convert Audio to Base64 (to send back over the internet)
    output = audio[0].T.float().cpu().numpy()
    import scipy.io.wavfile as wav
    import numpy as np
    
    # Save to buffer
    buffer = io.BytesIO()
    wav.write(buffer, 44100, output)
    buffer.seek(0)
    
    # Encode
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {"audio_base64": audio_base64}

# 4. START: Connect to RunPod
runpod.serverless.start({"handler": handler})
