# Module goi Gemini (generative_text.py) 
import google.generativeai as genai
import logging

# Bien toan cuc de giu model
feedback_model = None
topic_model = None

# Dinh nghia cac system prompt (huong dan vai tro cho AI)
FEEDBACK_SYSTEM_PROMPT = """
Ban la mot huan luyen vien giao tiep (speaking coach) chuyen nghiep va than thien. 
Nguoi dung (hoc sinh) se cung cap mot doan van ban ma ho vua noi.
Nhiem vu cua ban la:
1. Dua ra 1-2 diem ban ay lam TOT (vi du: dung tu hay, y ro rang).
2. Dua ra 1-2 diem ban ay can CAI THIEN (vi du: cau hoi dai, bi vap, thieu tu noi).
3. Dua ra mot GOI Y cu the (vi du: "Ban thu tach cau do thanh 2 cau ngan hon xem sao?").
Hay tra loi ngan gon, mang tinh xay dung, va than thien. Khong can qua trang trong.
"""

TOPIC_SYSTEM_PROMPT = """
Ban la mot nguoi ban sang tao. Hay tao ra mot chu de giao tiep (speaking topic) ngan gon, thu vi cho nguoi dung luyen tap.
Chu de nen la mot cau hoi mo hoac mot de tai tranh luan nhe nhang.
Vi du: "Ban nghi sao ve viec hoc online?", "Hay ke ve noi ban thich nhat trong thanh pho cua minh."
Chi tra loi duy nhat chu de do, khong them loi dan.
"""

# --- THAY DOI: Ham init chap nhan 2 keys ---
def init_genai_models(feedback_api_key, topic_api_key):
    """
    Khoi tao cac model Gemini cho Text Generation voi API Key rieng biet.
    """
    global feedback_model, topic_model
    
    try:
        # --- Tao model FEEDBACK voi key 1 ---
        logging.info("Dang khoi tao model Feedback (Gemini)...")
        genai.configure(api_key=feedback_api_key)
        feedback_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=FEEDBACK_SYSTEM_PROMPT
        )
        
        # --- Tao model TOPIC voi key 2 ---
        logging.info("Dang khoi tao model Topic (Gemini)...")
        genai.configure(api_key=topic_api_key)
        topic_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=TOPIC_SYSTEM_PROMPT
        )
        
        logging.info("Khoi tao 2 model GenAI (Text) thanh cong.")
    except Exception as e:
        logging.error(f"Loi khi khoi tao Gemini: {e}")
        raise e

def get_feedback_on_speech(transcript_text):
    """
    Goi Gemini de lay feedback ve doan text.
    """
    if feedback_model is None:
        raise Exception("Model Feedback (Gemini) chua duoc khoi tao.")
        
    try:
        # Khoi tao chat history (hoac goi truc tiep neu khong can chat)
        response = feedback_model.generate_content(transcript_text)
        return response.text
    except Exception as e:
        logging.error(f"Loi Gemini (get_feedback): {e}")
        return f"[Loi: Khong the lay feedback tu AI. {e}]"

def generate_new_topic():
    """
    Goi Gemini de tao chu de moi.
    """
    if topic_model is None:
        raise Exception("Model Topic (Gemini) chua duoc khoi tao.")
        
    try:
        # Prompt trong truong hop nay la rong, vi da set het trong system_instruction
        response = topic_model.generate_content("Tao mot chu de moi.")
        return response.text
    except Exception as e:
        logging.error(f"Loi Gemini (generate_topic): {e}")
        return f"[Loi: Khong the tao topic. {e}]"

