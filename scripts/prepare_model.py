import os
import subprocess
import sys
import importlib

def install_and_import(package):
    """Cài đặt và nhập thư viện."""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Đang cài đặt {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    finally:
        globals()[package] = importlib.import_module(package)

# --- Hàm để chạy lệnh và kiểm tra lỗi ---
def run_command(command):
    """Chạy một lệnh trong shell và in ra output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Lỗi khi chạy lệnh: {command}")
        print(stderr.decode('utf-8'))
    else:
        print(stdout.decode('utf-8'))

# --- Bắt đầu quy trình ---
print("--- Bắt đầu quá trình chuẩn bị model ---")
print("Vui lòng chờ, quá trình này sẽ mất khoảng 5-10 phút tùy thuộc vào tốc độ của Colab.")

# --- Bước 1: Cài đặt các thư viện cần thiết ---
# Đảm bảo các thư viện cần thiết đã được cài đặt
print("\n[Bước 1/4] Đang cài đặt các thư viện cần thiết...")
try:
    install_and_import("transformers")
    install_and_import("torch")
    install_and_import("huggingface_hub")
    print("Cài đặt hoàn tất!")
except Exception as e:
    print(f"Lỗi khi cài đặt thư viện: {e}")
    sys.exit()

# Tải các module sau khi đã cài đặt
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# --- Bước 2: Tải Model và Processor từ Hugging Face ---
# Đây là model AI nhận diện giọng nói tốt nhất và mới nhất
MODEL_NAME = "openai/whisper-large-v3"
print(f"\n[Bước 2/4] Đang tải processor và model 'xịn' nhất từ: {MODEL_NAME}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)
    print("Tải model thành công!")
except Exception as e:
    print(f"Đã xảy ra lỗi khi tải model: {e}")
    sys.exit()

# --- Bước 3: Lưu Model và Processor vào thư mục local ---
SAVE_DIRECTORY = "./whisper-large-v3-local"
print(f"\n[Bước 3/4] Đang lưu model vào thư mục: {SAVE_DIRECTORY}")
try:
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        
    model.save_pretrained(SAVE_DIRECTORY)
    processor.save_pretrained(SAVE_DIRECTORY)
    print("Lưu model thành công!")
except Exception as e:
    print(f"Đã xảy ra lỗi khi lưu model: {e}")
    sys.exit()

# --- Bước 4: Nén thư mục để tải về ---
ZIP_FILE_NAME = "whisper_model.zip"
print(f"\n[Bước 4/4] Đang nén thư mục '{SAVE_DIRECTORY}' thành file '{ZIP_FILE_NAME}'...")
run_command(f"zip -r -q {ZIP_FILE_NAME} {SAVE_DIRECTORY}")

print("\n--- 🎉 HOÀN TẤT! ---")
print(f"Một file tên là '{ZIP_FILE_NAME}' đã được tạo thành công.")
print("-> Hãy tìm nó ở thanh file bên trái của Colab, nhấn chuột phải và chọn 'Download' để tải về máy.")

