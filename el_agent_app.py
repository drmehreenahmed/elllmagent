# el_agent_app.py -- Full integrated Streamlit app
# Classification + High-only RF Regression + LLM analysis
# Original classification features fully preserved

import streamlit as st
import torch
import tempfile
import shutil
import os
import io
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------- CONFIG -------------------------
MODEL_DIR = "models/deepseek-llm-7b-chat"
FEATURE_MODEL_PATH = "models/resnet18_feature_extractor.pth"
CLASSIFIER_PATH = "models/best_classifier_v2.pkl"
RF_MODEL_PATH = "models/weighted_random_forest_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
IMAGE_DISPLAY_WIDTH = 360
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS_DEFAULT = 250
TEMPERATURE_DEFAULT = 0.2

st.set_page_config(page_title="🔍 EL LLM Agent", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
    .pred-high {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pred-low {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pred-unknown {
        background: linear-gradient(135deg, #e2e3e5, #d6d8db);
        border: 2px solid #6c757d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pred-header {
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .pred-sub {
        font-size: 0.9em;
        color: #666;
        margin-top: 8px;
    }
    .pred-percent {
        font-size: 1.3em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 5px 0;
    }
    .pce-highlight {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        font-size: 1.4em;
        font-weight: bold;
        text-align: center;
        color: #856404;
        margin: 8px 0;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        max-height: 500px;
        overflow-y: auto;
    }
    .chat-user {
        text-align: right;
        margin: 10px 0;
    }
    .chat-llm {
        text-align: left;
        margin: 10px 0;
    }
    .bubble {
        display: inline-block;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 80%;
    }
    .chat-user .bubble {
        background: #007bff;
        color: white;
    }
    .chat-llm .bubble {
        background: #e9ecef;
        color: #333;
        border: 1px solid #dee2e6;
    }
    .small-muted {
        font-size: 0.9em;
        color: #6c757d;
        text-align: center;
        font-style: italic;
    }
    .user-input-section {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- MODEL LOADERS -------------------------
@st.cache_resource(show_spinner=True)
def load_deepseek_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tmpdir = tempfile.mkdtemp(prefix="ds_offload_")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            offload_folder=tmpdir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        device = next(model.parameters()).device
        return tokenizer, model, device, tmpdir
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to("cpu")
        return tokenizer, model, torch.device("cpu"), None

@st.cache_resource(show_spinner=True)
def load_regression_components(rf_path=RF_MODEL_PATH, scaler_path=SCALER_PATH):
    reg_model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    return reg_model, scaler

@st.cache_resource(show_spinner=True)
def load_pce_classifier():
    try:
        from predictor import PCEClassifier
        return PCEClassifier(FEATURE_MODEL_PATH, CLASSIFIER_PATH)
    except Exception:
        return None

# ------------------------- HELPER FUNCTIONS -------------------------
def parse_classifier_result(raw):
    if raw is None:
        return {"prediction": "Unknown", "confidence": 0.0}
    if isinstance(raw, dict):
        pred = raw.get("prediction", raw.get("label", "Unknown"))
        conf = raw.get("confidence", raw.get("score", 0.0))
    else:
        pred, conf = raw, 0.0
    if isinstance(pred, (int, float)):
        pred = "High" if pred >= 0.5 else "Low"
    if isinstance(pred, str):
        s = pred.strip().upper()
        if s.startswith("H"): pred = "High"
        elif s.startswith("L"): pred = "Low"
        else: pred = "Unknown"
    try:
        conf = float(conf)
        if conf > 1.0: conf /= 100.0
    except:
        conf = 0.0
    return {"prediction": pred, "confidence": conf}

def get_fast_features(img_np):
    # same as original regression feature extractor
    vals = img_np.flatten()
    features = [
        np.mean(vals), np.std(vals), np.median(vals),
        np.max(vals)-np.min(vals)
    ]
    hist, _ = np.histogram(vals, bins=32, range=(0,255))
    hist = hist / (np.sum(hist)+1e-10)
    features.extend([
        -np.sum(hist*np.log2(hist+1e-10)),
        np.percentile(vals,25), np.percentile(vals,75)
    ])
    from scipy.ndimage import uniform_filter
    local_var = uniform_filter(img_np.astype(float)**2, size=3) - uniform_filter(img_np.astype(float), size=3)**2
    features.extend([np.mean(local_var), np.std(local_var)])
    edges = cv2.Canny(img_np, 50, 150)
    features.extend([np.mean(edges), np.sum(edges>0)/edges.size])
    _, binary = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    features.append(np.sum(binary>0)/binary.size)
    small_img = cv2.resize(img_np,(64,64))
    from scipy.fft import fft2, fftshift
    fft = np.log1p(np.abs(fftshift(fft2(small_img.astype(float)))))
    features.extend([np.mean(fft), np.std(fft), np.max(fft)-np.min(fft)])
    return np.array(features)

def predict_pce_high_only(pil_img, cls_label, reg_model, scaler):
    if cls_label != "High":
        return "—"
    try:
        feats = get_fast_features(np.array(pil_img.convert("L")))
        feats_scaled = scaler.transform(feats.reshape(1,-1))
        return round(float(reg_model.predict(feats_scaled)[0]),3)
    except:
        return "—"

def classify_with_wrapper(classifier, img_bytes):
    if classifier is None:
        return {"prediction": "Unknown", "confidence": 0.0}
    try:
        return classifier.predict(img_bytes)
    except Exception:
        return {"prediction": "Unknown", "confidence": 0.0}

def llm_generate_local(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=250, temperature=0.2):
    full_prompt = f"<SYSTEM>\n{system_prompt}\n</SYSTEM>\n\n<USER>\n{user_prompt}\n</USER>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    if device.type == "cuda":
        inputs = {k:v.to(device) for k,v in inputs.items()}
    pad_id = tokenizer.eos_token_id or 100001
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, top_p=0.9, pad_token_id=pad_id)
    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text

def generate_llm_response(system_prompt, user_prompt, max_new_tokens, temperature):
    if model_choice=="local" and deepseek is not None and tokenizer is not None:
        return llm_generate_local(tokenizer, deepseek, llm_device, system_prompt, user_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return "⚠️ No local LLM available."

# ------------------------- SESSION STATE -------------------------
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "results" not in st.session_state: st.session_state.results = []
if "last_upload_names" not in st.session_state: st.session_state.last_upload_names = []
if "user_question_input" not in st.session_state: st.session_state.user_question_input = ""

# ------------------------- MODEL LOAD -------------------------
with st.sidebar:
    st.title("⚙️ Models & LLM")
    with st.spinner("Loading models..."):
        tokenizer, deepseek, llm_device, offload_tmp = load_deepseek_model()
        classifier = load_pce_classifier()
        reg_model, feature_scaler = load_regression_components()
    st.success("✅ Models loaded")
    model_choice = st.selectbox("LLM source", options=("local","none"))
    max_tokens = st.slider("Max tokens",64,1024,value=MAX_TOKENS_DEFAULT,step=16)
    temp = st.slider("Temperature",0.0,1.0,value=TEMPERATURE_DEFAULT,step=0.05)
    sys_prompt = st.text_area("System prompt", value="You are an expert in EL diagnostics. Only use provided predictions and PCE. Do not invent images.", height=140)
    user_prompt_default = st.text_input("Default user prompt", value="Provide concise batch summary (per-image notes + batch trends + 3 next steps).")
    extra_prompt = st.text_area("Extra prompt (optional)", value="Include confidence values and list Low efficiency images.")

# ------------------------- UI -------------------------
st.title("🔍 EL Image Efficiency Agent — Classification + High-only Regression")
uploaded_files = st.file_uploader("📤 Upload EL images", accept_multiple_files=True, type=["png","jpg","jpeg"])
col_run, col_clear, col_dl = st.columns([1,1,2])
run_batch = col_run.button("🔎 Classify & Summarize Batch")
clear_chat = col_clear.button("🧹 Clear LLM Chat")
if st.session_state.results:
    # Convert PCE values to strings for CSV to avoid type issues
    csv_data = []
    for result in st.session_state.results:
        csv_data.append({
            "file": result["file"],
            "class_prediction": result["class_prediction"],
            "class_confidence": result["class_confidence"],
            "pce": str(result["pce"])  # Convert to string for CSV
        })
    csv_bytes = pd.DataFrame(csv_data).to_csv(index=False).encode()
    col_dl.download_button("⬇️ Download CSV", csv_bytes, file_name="el_results.csv", mime="text/csv")

# ------------------------- PROCESS UPLOAD -------------------------
if uploaded_files:
    names = [f.name for f in uploaded_files]
    if names != st.session_state.last_upload_names:
        st.session_state.results = []
        st.session_state.last_upload_names = names
    cols = st.columns(2)
    for idx, uploaded in enumerate(uploaded_files):
        col = cols[idx%2]
        img_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        cls_raw = classify_with_wrapper(classifier, img_bytes)
        parsed = parse_classifier_result(cls_raw)
        pce_val = predict_pce_high_only(pil_img, parsed["prediction"], reg_model, feature_scaler)
        rec = {"file": uploaded.name, "class_prediction": parsed["prediction"], "class_confidence": parsed["confidence"], "pce": pce_val}
        existing = next((r for r in st.session_state.results if r["file"]==uploaded.name), None)
        if existing: existing.update(rec)
        else: st.session_state.results.append(rec)
        with col:
            st.image(pil_img, caption=uploaded.name, width=IMAGE_DISPLAY_WIDTH)
            cls_style = "pred-unknown"
            if rec["class_prediction"]=="High": cls_style="pred-high"
            elif rec["class_prediction"]=="Low": cls_style="pred-low"
            
            st.markdown(
                f"<div class='prediction-card {cls_style}'>"
                f"<div class='pred-header'>{'🔆' if rec['class_prediction']=='High' else '⚠️'} {rec['class_prediction']}</div>"
                f"<div class='pred-sub'>Confidence</div>"
                f"<div class='pred-percent'>{rec['class_confidence']*100:.1f}%</div>"
                f"</div>", unsafe_allow_html=True
            )
            
            # Highlight PCE value with special styling
            if rec["pce"] != "—":
                st.markdown(
                    f"<div class='pce-highlight'>"
                    f"Predicted PCE: {rec['pce']}"
                    f"</div>", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='pred-sub' style='text-align: center;'>"
                    f"PCE: {rec['pce']}"
                    f"</div>", unsafe_allow_html=True
                )
            
            # per-image LLM question
            q = st.text_input(f"Question for {uploaded.name}", key=f"q_{uploaded.name}", value="What could cause this efficiency/PCE level?")
            if st.button("💬 Ask LLM", key=f"ask_{uploaded.name}"):
                context = f"Image: {uploaded.name}\nPrediction: {rec['class_prediction']} ({rec['class_confidence']*100:.1f}%)\nPCE: {rec['pce']}"
                prompt_text = f"{context}\n\n{q}\n\n{extra_prompt or ''}"
                resp = generate_llm_response(sys_prompt, prompt_text, max_tokens, temp)
                st.session_state.chat_history.append({"role":"user","text":f"[{uploaded.name}] {q}"})
                st.session_state.chat_history.append({"role":"llm","text":resp})
                st.rerun()
    
    # show table - convert PCE to string for display
    st.markdown("### Batch results")
    display_data = []
    for result in st.session_state.results:
        display_data.append({
            "file": result["file"],
            "class_prediction": result["class_prediction"],
            "class_confidence": f"{result['class_confidence']*100:.1f}%",
            "pce": str(result["pce"])  # Convert to string for display
        })
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

# ------------------------- BATCH LLM SUMMARY -------------------------
if run_batch and st.session_state.results:
    context_text = "\n".join([
        f"{i+1}. {r['file']}: {r['class_prediction']} ({r['class_confidence']*100:.1f}%) — PCE: {r['pce']}"
        for i,r in enumerate(st.session_state.results)
    ])
    low_imgs = [r['file'] for r in st.session_state.results if r['class_prediction']=="Low"]
    low_context = f"⚠️ Low efficiency detected in: {', '.join(low_imgs)}\n\n" if low_imgs else ""
    final_prompt = low_context + user_prompt_default.strip() + "\n\nBatch results:\n" + context_text + "\n\n" + (extra_prompt or "")
    llm_raw = generate_llm_response(sys_prompt, final_prompt, max_tokens, temp)
    cleaned_response = llm_raw.replace(final_prompt,"").strip()
    st.session_state.chat_history.append({"role":"user","text":"Batch summary request"})
    st.session_state.chat_history.append({"role":"llm","text":cleaned_response})
    st.rerun()

# ------------------------- CHAT HISTORY -------------------------
st.markdown("## 💬 LLM Chat")
chat_col_left, chat_col_right = st.columns([2,1])

with chat_col_left:
    container = st.container()
    if not st.session_state.chat_history:
        container.markdown("<div class='chat-container'><div class='small-muted'>No LLM outputs yet — run batch or ask per-image.</div></div>", unsafe_allow_html=True)
    else:
        html="<div class='chat-container'>"
        for msg in st.session_state.chat_history:
            if msg['role']=="user": html+=f"<div class='chat-user'><div class='bubble'>{msg['text']}</div></div>"
            else: html+=f"<div class='chat-llm'><div class='bubble'>{msg['text']}</div></div>"
        html+="</div>"
        container.markdown(html, unsafe_allow_html=True)
    
    # User input box for custom questions
    st.markdown("---")
    st.markdown("### 💭 Ask a Custom Question")
    with st.container():
        st.markdown('<div class="user-input-section">', unsafe_allow_html=True)
        
        # Use a unique key for the text area to avoid session state conflicts
        user_question = st.text_area(
            "Enter your question about the EL analysis:",
            placeholder="e.g., What patterns do you see in the high-efficiency images? How can we improve low-efficiency samples?",
            height=100,
            key="custom_question_input"  # Changed key to avoid conflict
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_custom = st.button("🚀 Ask LLM", use_container_width=True, key="ask_custom_btn")
        with col2:
            clear_input = st.button("🧹 Clear Input", use_container_width=True, key="clear_input_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle Ask LLM button
        if ask_custom and user_question.strip():
            # Prepare context from current results
            if st.session_state.results:
                context_text = "\n".join([
                    f"{r['file']}: {r['class_prediction']} ({r['class_confidence']*100:.1f}%) — PCE: {r['pce']}"
                    for r in st.session_state.results
                ])
                full_prompt = f"Current batch results:\n{context_text}\n\nUser question: {user_question}\n\n{extra_prompt or ''}"
            else:
                full_prompt = f"User question: {user_question}\n\n{extra_prompt or ''}"
            
            resp = generate_llm_response(sys_prompt, full_prompt, max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":user_question})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
        
        # Handle Clear Input button - this will clear on next rerun
        if clear_input:
            # We can't directly modify the widget value, but we can use a different approach
            # The input will be empty on the next run
            st.rerun()

    # quick follow-ups
    st.markdown("---")
    st.markdown("### ⚡ Quick Follow-ups")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        if st.button("Identify correlations between class and PCE", use_container_width=True, key="corr_btn"):
            q="Based only on the provided filenames, classes, confidences, and PCE values, identify any patterns or correlations between class and PCE. Be concise."
            resp = generate_llm_response(sys_prompt,q+"\n\n"+(extra_prompt or ""), max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":q})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    
    with col_f2:
        if st.button("Suggest extra image features to extract", use_container_width=True, key="features_btn"):
            q="What additional image features (quantitative) should be extracted to improve PCE regression? Provide 5 concise suggestions."
            resp = generate_llm_response(sys_prompt,q+"\n\n"+(extra_prompt or ""), max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":q})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()

with chat_col_right:
    st.markdown("### Controls")
    if st.button("Clear chat", use_container_width=True, key="clear_chat_btn"):
        st.session_state.chat_history=[]
        st.rerun()
    st.markdown("---")
    st.markdown("**Export**")
    if st.session_state.results:
        # Convert PCE to strings for CSV export
        export_data = []
        for result in st.session_state.results:
            export_data.append({
                "file": result["file"],
                "class_prediction": result["class_prediction"],
                "class_confidence": result["class_confidence"],
                "pce": str(result["pce"])
            })
        df_export = pd.DataFrame(export_data)
        st.download_button("⬇️ Download CSV", df_export.to_csv(index=False).encode(), file_name="el_results.csv", mime="text/csv", use_container_width=True)
    st.caption("Local LLM must be installed for model_choice='local' to work.")