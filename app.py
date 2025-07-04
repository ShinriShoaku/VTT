import io
import os
import sys
import time
import torch
import librosa
import soundfile as sf
import numpy as np
import scipy.signal
import webrtcvad
import noisereduce as nr

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from pyngrok import ngrok
import subprocess
import yaml # Tambahkan import ini untuk membaca file YAML

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# --- Validasi Path Model ---
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"Error: Model directory not found at '{LOCAL_MODEL_DIR}'.")
    print("Please ensure the 'model' folder is located in the same directory as app.py,")
    print("or adjust the LOCAL_MODEL_DIR path in app.py to its correct location.")
    sys.exit(1)

print(f"Loading model from: {LOCAL_MODEL_DIR}")

# Load model once at startup
processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

app = FastAPI(title="Wav2Vec2 Indonesian ASR")

# === Konstanta Streaming ===
SR = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(SR * VAD_FRAME_MS / 1000)
VAD = webrtcvad.Vad(2)  # Level 2: moderate aggressiveness

# === Utility: Noise Reduction ===
def reduce_noise(data: np.ndarray, sr: int) -> np.ndarray:
    if len(data) == 0:
        return data
    noise_sample = data[:sr // 2] if len(data) > sr // 2 else data
    return nr.reduce_noise(y=data, y_noise=noise_sample, sr=sr)

# === Utility: Transcribe from audio byte stream ===
def transcribe_audio_bytes(audio_bytes: bytes, sr: int = 16000) -> str:
    try:
        data, orig_sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except Exception as e:
        raise ValueError(f"Could not read audio bytes with soundfile: {e}")

    if data.ndim > 1:
        data = data[:, 0]

    if orig_sr != sr:
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
    
    if data.size > 0:
        data = reduce_noise(data, sr)
    else:
        return ""

    inputs = processor(data, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# === REST Endpoint: Transcribe from Uploaded File ===
@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio")
    body = await file.read()
    try:
        text = transcribe_audio_bytes(body)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
    return JSONResponse({"text": text})

# === WebSocket Endpoint: Streaming Audio (PCM16) ===
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    speech_buffer = bytearray()
    in_speech = False
    last_speech_time = time.time()
    
    transcription_pending = False
    last_buffer_update_time = time.time()
    SILENCE_THRESHOLD = 0.8
    FINAL_TRANSCRIBE_DELAY = 0.5

    try:
        while True:
            data: bytes = await ws.receive_bytes()

            offset = 0
            while offset + VAD_FRAME_SIZE * 2 <= len(data):
                frame = data[offset: offset + VAD_FRAME_SIZE * 2]
                offset += VAD_FRAME_SIZE * 2

                is_speech = VAD.is_speech(frame, SR)

                if is_speech:
                    speech_buffer.extend(frame)
                    in_speech = True
                    last_speech_time = time.time()
                    transcription_pending = True
                elif in_speech:
                    speech_buffer.extend(frame)
                    if time.time() - last_speech_time > SILENCE_THRESHOLD:
                        in_speech = False
                        transcription_pending = False

                        if len(speech_buffer) > 0:
                            pcm = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            if pcm.size > 0:
                                pcm = reduce_noise(pcm, SR)
                            
                            inputs = processor(pcm, sampling_rate=SR, return_tensors="pt", padding=True)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                logits = model(**inputs).logits
                            pred = torch.argmax(logits, dim=-1)
                            text = processor.batch_decode(pred)[0]

                            if text:
                                print(f"📝 recognized: {text}")
                                await ws.send_text(text)
                            
                            speech_buffer.clear()
                else:
                    if len(speech_buffer) > 0 and time.time() - last_speech_time > SILENCE_THRESHOLD + FINAL_TRANSCRIBE_DELAY:
                        speech_buffer.clear()
                        transcription_pending = False

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        if len(speech_buffer) > 0:
            print("Final transcription on disconnect...")
            try:
                pcm = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                if pcm.size > 0:
                    pcm = reduce_noise(pcm, SR)
                inputs = processor(pcm, sampling_rate=SR, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(pred)[0]
                if text:
                    print(f"📝 recognized (final): {text}")
                    await ws.send_text(text)
            except Exception as e:
                print(f"Error during final transcription on disconnect: {e}")

    except Exception as e:
        print(f"❌ Error in websocket: {e}")


# --- Run if standalone ---
if __name__ == "__main__":
    import uvicorn

    FASTAPI_PORT = 8000

    print(f"Starting FastAPI server on port {FASTAPI_PORT}...")

    # Check if running in Google Colab environment
    is_colab = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

    try:
        NGROK_AUTH_TOKEN = None

        # Try to read from ngrok config file first (most persistent for local setups)
        ngrok_config_path = os.path.join(os.path.expanduser("~"), ".config", "ngrok", "ngrok.yml")
        if os.path.exists(ngrok_config_path):
            try:
                with open(ngrok_config_path, 'r') as f:
                    ngrok_config = yaml.safe_load(f)
                    if ngrok_config and 'authtoken' in ngrok_config:
                        NGROK_AUTH_TOKEN = ngrok_config['authtoken']
                        print("ngrok authtoken successfully read from ~/.config/ngrok/ngrok.yml.")
            except Exception as e:
                print(f"WARNING: Could not read ngrok authtoken from config file: {e}")

        # If not found from config file, try environment variable (good for Colab/Docker)
        if NGROK_AUTH_TOKEN is None:
            NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
            if NGROK_AUTH_TOKEN:
                print("ngrok authtoken successfully read from NGROK_AUTH_TOKEN environment variable.")
        
        # If still not found, try Colab secrets (specific to Colab)
        if NGROK_AUTH_TOKEN is None and is_colab:
            try:
                from google.colab import userdata
                NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN')
                if NGROK_AUTH_TOKEN:
                    print("ngrok authtoken successfully read from Google Colab Secrets.")
            except ImportError:
                pass # Not in Colab, or userdata not available

        # --- Instal ngrok CLI jika belum ada ---
        try:
            subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
            print("ngrok CLI is already installed.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ngrok CLI not found. Installing ngrok...")
            # Ini akan dijalankan baik di Colab maupun lokal Linux
            subprocess.run(["curl", "-s", "https://ngrok-builds.s3.amazonaws.com/ngrok_linux_amd64.zip", "-o", "ngrok.zip"], check=True)
            subprocess.run(["unzip", "-o", "ngrok.zip", "-d", "/usr/local/bin"], check=True)
            subprocess.run(["rm", "ngrok.zip"], check=True)
            print("ngrok CLI installed.")

        # --- Autentikasi ngrok dan buat tunnel ---
        if NGROK_AUTH_TOKEN:
            try:
                ngrok.set_auth_token(NGROK_AUTH_TOKEN)
                print("ngrok authtoken set for pyngrok.")
                
                ngrok.kill()
                public_url = ngrok.connect(FASTAPI_PORT).public_url
                print(f"ngrok tunnel created! Public URL: {public_url}")
                print(f"--- Your Public FastAPI ASR URL: {public_url} ---")
                print(f"--- Visit {public_url}/docs for API documentation (Swagger UI) ---")
            except Exception as e:
                print(f"ERROR: Failed to create ngrok tunnel: {e}")
                print("Falling back to local execution. The API will not be publicly accessible via ngrok.")
        else:
            print("WARNING: No ngrok authtoken found from any source.")
            print("The API will run locally, but will not be publicly accessible via ngrok.")
        
        # Jalankan Uvicorn (akan selalu dijalankan, baik dengan atau tanpa ngrok tunnel)
        uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT)

    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        print("Falling back to local execution. The API will not be publicly accessible.")
        uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT)