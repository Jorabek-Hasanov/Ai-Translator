import streamlit as st
import os
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set environment variable to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Page Configuration ---
st.set_page_config(
    page_title="Ultra Modern En-Uz Translator",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stApp {
        background-image: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title-text {
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
    }
    .subtitle-text {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 40px;
    }
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-size: 1.1rem;
    }
    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.7);
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
</style>
""", unsafe_allow_html=True)

# --- Logic & Model Loading ---

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate(text, tokenizer, model):
    text = text.strip()
    if not text:
        return ""

    inputs = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Corrected method to get token ID for Uzbek (Latin)
    # Using convert_tokens_to_ids as requested to fix lang_code_to_id error
    forced_bos_token_id = tokenizer.convert_tokens_to_ids('uzn_Latn')
    
    outputs = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id, 
        max_new_tokens=128
    )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Cleaning
    cleaned_text = decoded.strip()
    cleaned_text = re.sub(r'^(uzn?_Latn|eng?_Latn|uz|en)\s+', '', cleaned_text, flags=re.IGNORECASE).strip()
    return cleaned_text

# --- Layout ---
st.markdown('<div class="title-text">AI Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">English 🇬🇧 ➝ Uzbek 🇺🇿</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2, gap="large")

    try:
        with st.spinner("Model yuklanmoqda, iltimos kuting... (Loading model, please wait...)"):
            tokenizer, model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    with col1:
        st.markdown("**ENGLISH**")
        source_text = st.text_area("Source Text", height=200, placeholder="Enter text to translate...", label_visibility="collapsed")
        st.caption("TTS Disabled")

    with col2:
        st.markdown("**UZBEK**")
        if "translation" not in st.session_state:
            st.session_state.translation = ""
        st.text_area("Translated Text", value=st.session_state.translation, height=200, label_visibility="collapsed")
        st.caption("TTS Disabled")

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    if st.button("✨ TRANSLATE ✨"):
        if source_text:
            progress_bar = st.progress(0)
            for i in range(1, 101, 10):
                time.sleep(0.01)
                progress_bar.progress(i)
            
            result = translate(source_text, tokenizer, model)
            
            progress_bar.progress(100)
            st.session_state.translation = result
            st.rerun()
        else:
            st.warning("Please enter text first.")
