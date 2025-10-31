# Module xu ly STT (speech_to_text.py) 
import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import logging

# Bien toan cuc de giu model (chi load 1 lan)
stt_model = None
stt_processor = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Duong dan den model da tai ve (di tu file nay -> services -> backend -> models)
# --- THAY DOI: Cap nhat duong dan de tro vao 'whisper-small-local' ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'whisper-small-local'))

def init_stt_model():
    """
    Load model Whisper (STT) tu thu muc local.
    """
    global stt_model, stt_processor
    
    if stt_model is not None:
        logging.info("Model STT (Whisper) da duoc load.")
        return

    try:
        logging.info(f"Bat dau load model STT tu: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            logging.error(f"Thu muc model khong ton tai hoac bi trong: {MODEL_PATH}")
            logging.error("VUI LONG CHAY SCRIPT 'scripts/prepare_model.py' TRUOC KHI CHAY SERVER.")
            raise FileNotFoundError("Chua co model Whisper local.")
            
        stt_processor = AutoProcessor.from_pretrained(MODEL_PATH)
        stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH).to(device)
        
        logging.info(f"Load model STT (Whisper) thanh cong. Dang chay tren: {device}")
        
    except Exception as e:
        logging.error(f"Loi nghiem trong khi load model STT: {e}")
        raise e

def transcribe_audio(wav_file_path):
    """
    Nhan duong dan file WAV, chuyen thanh text.
    """
    if stt_model is None or stt_processor is None:
        raise Exception("Model STT chua duoc khoi tao.")

    try:
        # 1. Doc file audio va resample ve 16kHz
        speech_array, sampling_rate = librosa.load(wav_file_path, sr=16000)
        
        # 2. Xu ly audio dau vao
        input_features = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # 3. Tao token IDs du doan
        # forced_decoder_ids = stt_processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        
        # --- THAY DOI: Xoa max_length=448 (vi do la cua v3), de model 'small' tu dung config mac dinh ---
        predicted_ids = stt_model.generate(input_features) 
        
        # 4. Giai ma token IDs thanh text
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0].strip() # Tra ve ket qua text
        
    except Exception as e:
        logging.error(f"Loi trong qua trinh transcribe: {e}")
        return f"[Loi STT: {e}]"

