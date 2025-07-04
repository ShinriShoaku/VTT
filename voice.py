import os
import sys
import time
import argparse
import torch
import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
import collections
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import scipy.signal  
import noisereduce as nr


# === Konfigurasi Model dan Audio ===
if sys.platform.startswith('win'):
    LOCAL_MODEL_DIR = "C:/Users/YourUser/VoiceToText/"
    print("Detected OS: Windows")
elif sys.platform.startswith('linux'):
    LOCAL_MODEL_DIR = "/home/shinri/VoiceToText/"
    print("Detected OS: Linux")
else:
    print("Unsupported OS.")
    sys.exit(1)

SAMPLING_RATE = 16000
BLOCK_SIZE = 1600
CHANNELS = 1
DTYPE = "float32"
AUDIO_BUFFER_MAX_LENGTH = SAMPLING_RATE * 5
audio_buffer = collections.deque(maxlen=AUDIO_BUFFER_MAX_LENGTH)

# === Load Model ===
required_files = ["config.json", "pytorch_model.bin", "vocab.json", "tokenizer_config.json"]
if not all(os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in required_files):
    print("❌ Missing model files in:", LOCAL_MODEL_DIR)
    sys.exit(1)

try:
    processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL_DIR)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"✅ Model loaded on: {device}")

# === Fungsi Transkripsi dari Buffer Mic ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Audio stream warning: {status}", file=sys.stderr)
    audio_buffer.extend(indata[:, 0])

def transcribe_audio_from_buffer():
    if len(audio_buffer) == 0:
        return ""
    audio_array = np.array(audio_buffer, dtype=DTYPE)
    inputs = processor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def reduce_noise(audio_data, sr):
    # Ambil noise profil dari awal 0.5 detik (jika tersedia)
    noise_len = min(len(audio_data), sr // 2)
    noise_sample = audio_data[:noise_len]
    return nr.reduce_noise(y=audio_data, y_noise=noise_sample, sr=sr, prop_decrease=1.0)


# === Fungsi Transkripsi dari File Audio ===
def transcribe_file_audio(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        start = time.time()
        data, sr = sf.read(file_path, always_2d=True)
        data = data[:, 0]  # Ambil 1 channel
        if sr != SAMPLING_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLING_RATE)

        inputs = processor(data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0]
        print(f"\n⏱️  Transcription time: {time.time() - start:.2f} sec")
        return text

    except Exception as e:
        print(f"❌ Failed to process audio: {e}")
        return None



def realtime_transcription(input_device=None):
    print("\n🎤 Starting real-time transcription...\n")
    print("--- Audio Devices ---")
    print(sd.query_devices())
    print("----------------------")

    # Tentukan samplerate stream: pakai default device jika samplerate 16000 tidak didukung
    stream_samplerate = SAMPLING_RATE
    if input_device is not None:
        dev_info = sd.query_devices(input_device, 'input')
        default_sr = int(dev_info['default_samplerate'])
        try:
            # coba 16k dulu
            sd.check_input_settings(device=input_device, samplerate=SAMPLING_RATE)
        except Exception:
            print(f"⚠️ Device {input_device} tidak dukung {SAMPLING_RATE} Hz, pakai {default_sr} Hz")
            stream_samplerate = default_sr

    stream_args = {
        "samplerate": stream_samplerate,
        "blocksize": BLOCK_SIZE,
        "channels": CHANNELS,
        "dtype": DTYPE,
        "callback": audio_callback
    }
    if input_device is not None:
        stream_args["device"] = input_device

    try:
        with sd.InputStream(**stream_args):
            last_trans_time = time.time()
            last_text = ""
            silence_th = 0.7
            last_status = time.time()
            status = "Mendengarkan..."

            buffer_sr = stream_samplerate 
            silence_th = 1.0
            speaking = False
            last_speech_time = time.time()
            last_status_time = 0

            while True:
                now = time.time()
                if len(audio_buffer) >= buffer_sr * 0.5:
                    data = np.array(audio_buffer, dtype=DTYPE)

                    # Resample jika perlu
                    if buffer_sr != SAMPLING_RATE:
                        data = scipy.signal.resample(
                            data,
                            int(len(data) * SAMPLING_RATE / buffer_sr)
                        )

                    # === Noise Suppression ===
                    data = reduce_noise(data, SAMPLING_RATE)

                    rms = np.sqrt(np.mean(data**2))

                    # Deteksi sedang bicara
                    if rms > 0.02:
                        speaking = True
                        last_speech_time = now
                    elif now - last_speech_time > silence_th and speaking:
                        # Selesai bicara
                        speaking = False
                        result = transcribe_audio_from_buffer()
                        if result and result.strip():
                            print(f"\r🗣️  Anda berkata: {result}{' ' * 20}")
                        audio_buffer.clear()

                    if now - last_status_time > 0.5:
                        state = "Berbicara..." if speaking else "Mendengarkan..."
                        print(f"\r{state} (RMS: {rms:.4f})", end="", flush=True)
                        last_status_time = now

                time.sleep(0.05)


    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        print(sd.query_devices())
    finally:
        print("\n🧹 Membersihkan...")
        sd.stop()


# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path ke file audio (mp3, wav, flac, dll)")
    parser.add_argument("--realtime", action="store_true", help="Mode mikrofon real-time")
    parser.add_argument("--device", type=int, help="Index perangkat audio input (mic)")
    args = parser.parse_args()

    if args.file:
        print(f"\n--- Transcribing file: {args.file} ---")
        result = transcribe_file_audio(args.file)
        if result:
            print(f"\n📝 Hasil Transkripsi: {result}")
        else:
            print("\n❌ Gagal mentranskripsi file audio.")
    elif args.realtime:
        realtime_transcription(args.device)
    else:
        print("❗ Gunakan salah satu argumen: --file /path/to/audio atau --realtime")
