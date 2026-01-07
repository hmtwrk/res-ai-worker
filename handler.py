def handler(job):
    global pipe
    print("--- ⚡ Request Received ---", file=sys.stderr, flush=True)
    
    if pipe is None:
        load_model()

    job_input = job["input"]
    prompt = job_input.get("prompt", "Cinematic drone")
    duration = job_input.get("duration", 10)
    
    # --- NEW: Read Steps & CFG from the Job Input ---
    steps = job_input.get("steps", 50)        # Default to 50 if missing
    cfg_scale = job_input.get("cfg_scale", 7.0) # Default to 7.0 if missing

    # Generate with the new variables
    audio = pipe(
        prompt, 
        num_inference_steps=steps, 
        guidance_scale=cfg_scale, 
        audio_end_in_s=duration
    ).audios

    # Encode
    buffer = io.BytesIO()
    import scipy.io.wavfile as wav
    wav.write(buffer, 44100, audio[0].T.float().cpu().numpy())
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    print("--- 📤 Sending Audio Back ---", file=sys.stderr, flush=True)
    return {"audio_base64": audio_base64}
