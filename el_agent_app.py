# el_agent_app.py -- Optimized version using online models (fixed)
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
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from predictor import PCEClassifier  # should be ONNX-capable

# ------------------------- CONFIG -------------------------
DEEPSEEK_MODEL_ID = "microsoft/DialoGPT-small"  # Much smaller model for deployment
CLASSIFIER_PATH = "models/best_classifier.onnx"
RF_MODEL_PATH = "models/weighted_random_forest_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

IMAGE_DISPLAY_WIDTH = 360
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS_DEFAULT = 150  # Reduced from 250
TEMPERATURE_DEFAULT = 0.3  # Increased from 0.2 for small models

st.set_page_config(page_title="🔍 EL LLM Agent", layout="wide")

# ------------------------- SESSION STATE -------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if "last_upload_names" not in st.session_state:
    st.session_state.last_upload_names = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ ADD ALL MODEL VARIABLES TO SESSION STATE:
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

if "llm_device" not in st.session_state:
    st.session_state.llm_device = torch.device("cpu")

if "classifier" not in st.session_state:
    st.session_state.classifier = None

if "resnet_model" not in st.session_state:
    st.session_state.resnet_model = None

if "reg_model" not in st.session_state:
    st.session_state.reg_model = None

if "feature_scaler" not in st.session_state:
    st.session_state.feature_scaler = None

# ------------------------- STYLES -------------------------
st.markdown("""
<style>
/* (your CSS unchanged) */
.pred-high { background: linear-gradient(135deg, #d4edda, #c3e6cb); border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.pred-low { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 2px solid #dc3545; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.pred-unknown { background: linear-gradient(135deg, #e2e3e5, #d6d8db); border: 2px solid #6c757d; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.pred-error { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 2px solid #dc3545; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.pred-header { font-size: 1.4em; font-weight: bold; margin-bottom: 10px; text-align: center; }
.pred-sub { font-size: 0.9em; color: #666; margin-top: 8px; }
.pred-percent { font-size: 1.3em; font-weight: bold; color: #2c3e50; text-align: center; margin: 5px 0; }
.pce-highlight { background: linear-gradient(135deg, #fff3cd, #ffeaa7); border: 2px solid #ffc107; border-radius: 8px; padding: 12px; font-size: 1.4em; font-weight: bold; text-align: center; color: #856404; margin: 8px 0; }
.chat-container { background: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0; max-height: 500px; overflow-y: auto; }
.chat-user { text-align: right; margin: 10px 0; }
.chat-llm { text-align: left; margin: 10px 0; }
.bubble { display: inline-block; padding: 10px 15px; border-radius: 18px; max-width: 80%; }
.chat-user .bubble { background: #007bff; color: white; }
.chat-llm .bubble { background: #e9ecef; color: #333; border: 1px solid #dee2e6; }
.small-muted { font-size: 0.9em; color: #6c757d; text-align: center; font-style: italic; }
.user-input-section { background: white; border: 2px solid #e9ecef; border-radius: 10px; padding: 15px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ------------------------- MODEL LOADERS -------------------------
@st.cache_resource(show_spinner=True)
def load_deepseek_model():
    """Load the smallest possible model for deployment"""
    model_options = [
        # Try these in order - smallest first
        {"name": "TinyGPT2", "id": "sshleifer/tiny-gpt2", "size": "50M"},
        {"name": "DialoGPT-small", "id": "microsoft/DialoGPT-small", "size": "117M"},
        {"name": "DistilGPT2", "id": "distilgpt2", "size": "82M"},
    ]
    
    for model_info in model_options:
        try:
            with st.spinner(f"🔄 Loading {model_info['name']} ({model_info['size']})..."):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info["id"],
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_info["id"],
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                model.to("cpu")
                st.success(f"✅ {model_info['name']} loaded successfully ({model_info['size']})")
                return tokenizer, model, torch.device("cpu"), None
                
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_info['name']}: {str(e)[:100]}...")
            continue
    
    st.error("❌ All model loading attempts failed")
    return None, None, torch.device("cpu"), None

@st.cache_resource(show_spinner=True)
def load_regression_components(rf_path=RF_MODEL_PATH, scaler_path=SCALER_PATH):
    try:
        reg_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        st.success("✅ Regression models loaded successfully")
        return reg_model, scaler
    except Exception as e:
        st.error(f"❌ Failed to load regression models: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=True)
def load_pce_classifier(path):
    """Load the PCE classifier (ONNX via predictor.PCEClassifier expected)."""
    try:
        classifier = PCEClassifier(path)
        if classifier.is_loaded:
            st.success("✅ PCEClassifier loaded successfully")
            return classifier
        else:
            st.error("❌ PCEClassifier file not found or corrupted")
            return None
    except Exception as e:
        st.error(f"❌ Failed to initialize PCEClassifier: {e}")
        return None

@st.cache_resource(show_spinner=True)
# ------------------------- ResNet Feature Extractor (OLD PIPELINE) -------------------------
def load_old_resnet_feature_extractor(device="cpu"):
    """
    Load ResNet18 feature extractor as in the old training pipeline.
    - pretrained ImageNet weights
    - final FC removed
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # remove final classification layer
    model = model.to(device).eval()
    return model

# Preprocessing matching old pipeline
old_resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
    transforms.ToTensor(),                         # No normalization
])

def extract_resnet_features_old(pil_image, resnet_model, device="cpu"):
    """
    Extract 512-dim ResNet features matching old pipeline.
    Input:
        pil_image: PIL.Image in RGB or grayscale
        resnet_model: loaded ResNet18 feature extractor
    Output:
        1D numpy array of shape (512,)
    """
    # Convert PIL to NumPy grayscale if needed
    img_np = np.array(pil_image.convert("L"))
    # Convert to 3-channel RGB as in old pipeline
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    # Apply transforms
    img_tensor = old_resnet_transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats = resnet_model(img_tensor)
    
    # Flatten to 1D NumPy array
    return feats.cpu().numpy().squeeze().astype(np.float32)


# ------------------------- FEATURE / CLASSIFY WRAPPERS -------------------------
# def extract_resnet_features(pil_image, resnet_model):
#     """
#     Drop-in replacement that replicates the old ResNet pipeline:
#     - Grayscale → 3-channel RGB
#     - Resize (224,224)
#     - No normalization
#     - Output: 512-dim feature vector
#     """
#     img_np = np.array(pil_image.convert("L"))
#     img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
#     img_tensor = old_resnet_transform(img_rgb).unsqueeze(0).to("cpu")  # CPU for Streamlit
#     with torch.no_grad():
#         feats = resnet_model(img_tensor)
#     return feats.cpu().numpy().squeeze().astype(np.float32)

def classify_with_wrapper(classifier, pil_image, resnet_model):
    if classifier is None or resnet_model is None:
        return None

    try:
        feats = extract_resnet_features_old(pil_image, resnet_model)
        if feats is None:
            return None

        # ALWAYS reshape for ONNX/sklearn: (1,512)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        # Use classifier prediction exactly as old pipeline
        return classifier.predict_from_features(feats)

    except Exception as e:
        st.error(f"❌ Classifier error: {e}")
        return None

# ------------------------- AUXILIARY FUNCTIONS -------------------------
def parse_classifier_result(raw):
    if raw is None:
        return {"prediction": "Error"}
    if isinstance(raw, dict):
        pred = raw.get("prediction", raw.get("label", "Unknown"))
    else:
        pred = raw
    if isinstance(pred, (int, float, np.number)):
        pred = "High" if float(pred) >= 0.5 else "Low"
    if isinstance(pred, str):
        s = pred.strip().upper()
        if s.startswith("H"): pred = "High"
        elif s.startswith("L"): pred = "Low"
        elif s.startswith("E"): pred = "Error"
        else: pred = "Unknown"
    return {"prediction": pred}

def get_fast_features(img_np):
    vals = img_np.flatten()
    features = [np.mean(vals), np.std(vals), np.median(vals), np.max(vals)-np.min(vals)]
    hist, _ = np.histogram(vals, bins=32, range=(0,255))
    hist = hist / (np.sum(hist)+1e-10)
    features.extend([-np.sum(hist*np.log2(hist+1e-10)), np.percentile(vals,25), np.percentile(vals,75)])
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
    return np.array(features, dtype=np.float32)

def predict_pce_high_only(pil_img, cls_label, reg_model, scaler):
    """
    Predict PCE using old 15-feature fast features, not ResNet features.
    PCE is returned as percentage with proper formatting.
    """
    if cls_label != "High" or reg_model is None or scaler is None:
        return "—"
    try:
        # Use old handcrafted features for regression
        feats = get_fast_features(np.array(pil_img.convert("L")))
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        pce_percentage = float(reg_model.predict(feats_scaled)[0])
        
        # The regression model already outputs percentage values (10-18 range)
        # Just round to 2 decimal places
        return round(pce_percentage, 2)
        
    except Exception as e:
        st.error(f"❌ Regression error: {e}")
        return "—"

def llm_generate_local(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=150, temperature=0.3):
    """Optimized generation with support for DeepSeek chat format"""
    try:
        st.info("🔄 Starting LLM generation...")
        
        # Check if it's a DeepSeek model and use appropriate prompt format
        model_name = str(type(model)).lower()
        if "deepseek" in model_name:
            # DeepSeek chat format
            full_prompt = f"<s>System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            # Default format for other models
            full_prompt = f"{system_prompt}\n\nQuestion: {user_prompt}\nAnswer:"
        
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,  # Increased for larger models
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generation with model-specific parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        st.success(f"✅ Generated {len(text)} characters")
        return text.strip()
        
    except Exception as e:
        st.error(f"❌ Generation error: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return f"⚠️ LLM Error: {str(e)}"
        
def adapt_prompt_for_model(system_prompt, user_prompt, model_name="unknown"):
    """
    Adapt prompts based on specific model capabilities
    """
    if "deepseek" in model_name.lower():
        # DeepSeek can handle complex prompts
        return system_prompt, user_prompt
        
    elif "gpt2" in model_name.lower():
        # GPT2 needs very simple prompts
        simplified_system = "You analyze solar cell EL images. High efficiency = good. Low efficiency = defects. Be very direct."
        simple_user = user_prompt
        if len(user_prompt.split()) > 15:
            simple_user = "Summarize the solar cell analysis results clearly."
        return simplified_system, simple_user
        
    elif "opt" in model_name.lower():
        # OPT can handle moderate complexity
        simplified_system = "You are a solar cell EL analysis expert. Use classification results and PCE values."
        return simplified_system, user_prompt
        
    else:
        # Default
        return system_prompt, user_prompt

def build_context_aware_prompt(results_data, user_question):
    """Build prompts with dynamic context based on the question type"""
    
    base_context = build_llm_context(results_data)
    question_lower = user_question.lower()
    
    # Add question-specific guidance
    if any(term in question_lower for term in ['defect', 'crack', 'problem', 'issue']):
        guidance = """
DEFECT ANALYSIS FOCUS:
- Analyze dark areas, cracks, and non-uniform patterns in low-efficiency images
- Identify potential root causes: microcracks, shunts, finger interruptions
- Assess defect severity and distribution patterns
- Recommend specific inspection techniques for identified defects
"""
    elif any(term in question_lower for term in ['pce', 'efficiency', 'performance']):
        guidance = """
PCE ANALYSIS FOCUS:
- Evaluate PCE distribution and consistency
- Identify outliers and performance variations
- Assess correlation between efficiency classification and PCE values
- Provide insights on cell quality based on PCE metrics
"""
    elif any(term in question_lower for term in ['summary', 'overview', 'batch']):
        guidance = """
BATCH ANALYSIS FOCUS:
- Provide comprehensive batch quality assessment
- Highlight key patterns and trends
- Identify both strengths and areas for improvement
- Give overall technical assessment with specific metrics
"""
    else:
        guidance = """
GENERAL ANALYSIS:
- Provide technical insights based on all available data
- Focus on both high and low efficiency patterns
- Offer practical engineering perspectives
- Suggest data-driven conclusions
"""
    
    return f"{base_context}\n\n{guidance}\n\nQuestion: {user_question}"

def clean_llm_response(response, model_name):
    """Clean and fix common model response issues with strict filtering"""
    if not response or len(response.strip()) < 5:
        return "No meaningful response generated."
    
    response = response.strip()
    
    # Strict filtering for irrelevant technical content
    irrelevant_patterns = [
        'laser', 'e-ink', 'optical sensor', 'photonic', 'olympus', 'electron',
        'equation', 'calculation', 'sensor', 'detector', 'device', 'system',
        'power consumption', 'energy', 'light output', 'watts', 'voltage',
        'http://', 'https://', 'www.', '.com', '.org', '[1]', '[2]', '[3]'
    ]
    
    # Check if response contains irrelevant technical terms
    irrelevant_count = sum(1 for pattern in irrelevant_patterns if pattern in response.lower())
    if irrelevant_count >= 2:
        return "The model provided an irrelevant technical response. Please try asking about solar cell EL analysis specifically."
    
    # Check for meaningful solar cell content
    relevant_terms = [
        'solar', 'cell', 'pv', 'electroluminescence', 'el', 'efficiency',
        'defect', 'crack', 'shunt', 'dark area', 'uniform', 'luminescence',
        'pce', 'high', 'low', 'quality', 'module', 'photovoltaic'
    ]
    
    relevant_count = sum(1 for term in relevant_terms if term in response.lower())
    
    # If response has more irrelevant than relevant content, reject it
    if irrelevant_count > relevant_count:
        return "Response appears off-topic. Please ask specific questions about solar cell EL image analysis results."
    
    # Ensure minimum relevance
    if relevant_count < 2:
        return "Response lacks relevant solar cell analysis content. Please try a more specific question."
    
    return response
def build_llm_context(results_data):
    """Build rich context from classifier and regression results for the LLM"""
    if not results_data:
        return "No analysis results available."
    
    # Calculate statistics
    high_count = len([r for r in results_data if r["class_prediction"] == "High"])
    low_count = len([r for r in results_data if r["class_prediction"] == "Low"])
    total_count = len(results_data)
    
    # Get PCE values for high efficiency images (as corrected percentages)
    high_pces = [float(r["pce"]) for r in results_data if r["pce"] != "—" and r["class_prediction"] == "High"]
    low_images = [r["file"] for r in results_data if r["class_prediction"] == "Low"]
    
    # Build context
    context_parts = []
    
    # Overall statistics
    context_parts.append(f"EL IMAGE ANALYSIS RESULTS:")
    context_parts.append(f"- Total images analyzed: {total_count}")
    context_parts.append(f"- High efficiency solar cells: {high_count} ({high_count/total_count*100:.1f}%)")
    context_parts.append(f"- Low efficiency solar cells: {low_count} ({low_count/total_count*100:.1f}%)")
    
    # PCE analysis for high efficiency (as corrected percentages)
    if high_pces:
        context_parts.append(f"- PCE range (high efficiency): {min(high_pces):.2f}% to {max(high_pces):.2f}%")
        context_parts.append(f"- Average PCE (high efficiency): {sum(high_pces)/len(high_pces):.2f}%")
        # Add PCE quality assessment
        good_pce_count = sum(1 for pce in high_pces if pce >= 10.0)
        context_parts.append(f"- PCE QUALITY: {good_pce_count}/{len(high_pces)} high efficiency cells ≥10%")
    
    # Low efficiency details
    if low_images:
        context_parts.append(f"- Low efficiency images: {', '.join(low_images[:5])}{'...' if len(low_images) > 5 else ''}")
    
    # Quality assessment based on distribution
    efficiency_ratio = high_count / total_count if total_count > 0 else 0
    if efficiency_ratio > 0.7:
        context_parts.append("- BATCH QUALITY: Good batch with majority high efficiency")
    elif efficiency_ratio < 0.3:
        context_parts.append("- BATCH QUALITY: Poor batch with majority low efficiency")
    else:
        context_parts.append("- BATCH QUALITY: Mixed batch with significant variation")
    
    # Individual image details
    context_parts.append("\nDETAILED RESULTS:")
    for i, result in enumerate(results_data[:10]):  # Limit to first 10 for context length
        pce_display = f"{result['pce']}%" if result["pce"] != "—" else "N/A"
        context_parts.append(f"{i+1}. {result['file']}: {result['class_prediction']} efficiency, PCE: {pce_display}")
    
    if len(results_data) > 10:
        context_parts.append(f"... and {len(results_data) - 10} more images")
    
    return "\n".join(context_parts)

def generate_llm_response(system_prompt, user_prompt, max_new_tokens, temperature):
    try:
        if (hasattr(st.session_state, 'llm_model') and 
            st.session_state.llm_model is not None):
            
            # Detect model type for prompt adaptation
            model_type = str(type(st.session_state.llm_model)).lower()
            if "deepseek" in model_type:
                model_name = "deepseek"
                max_new_tokens = min(max_new_tokens, 500)
                temperature = min(temperature, 0.7)
            elif "gpt2" in model_type:
                model_name = "gpt2"
                max_new_tokens = min(max_new_tokens, 80)
                temperature = min(temperature, 0.4)
            elif "opt" in model_type:
                model_name = "opt"
                max_new_tokens = min(max_new_tokens, 120)
                temperature = min(temperature, 0.5)
            else:
                model_name = "unknown"
            
            # Adapt prompts for the specific model
            adapted_system, adapted_user = adapt_prompt_for_model(
                system_prompt, user_prompt, model_name
            )
            
            response = llm_generate_local(
                st.session_state.tokenizer, 
                st.session_state.llm_model, 
                st.session_state.llm_device, 
                adapted_system,
                adapted_user,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Post-process response - ONLY 2 ARGUMENTS NOW
            cleaned_response = clean_llm_response(response, model_name)
            return cleaned_response
            
        else:
            return "⚠️ No LLM loaded. Please select and load a model in the sidebar."
            
    except Exception as e:
        return f"⚠️ LLM generation error: {str(e)}"
        
def test_model_loading(model_id, model_name):
    """Test if a specific model can be loaded successfully"""
    try:
        with st.spinner(f"🔄 Testing {model_name}..."):
            # Test tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Test model loading
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Test basic generation
            test_input = "Say 'Hello World' in 3 words:"
            inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=50)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return True, f"✅ {model_name} loaded successfully", response
            
    except Exception as e:
        return False, f"❌ {model_name} failed: {str(e)[:100]}", None

# ------------------------- MODEL INITIALIZATION (SIDEBAR) -------------------------
with st.sidebar:
    st.title("⚙️ Models & LLM")
    
    # Model testing section
    st.markdown("### 🧪 Model Testing")
    
    models_to_test = [
        {"id": "deepseek-ai/deepseek-llm-7b-chat", "name": "DeepSeek-7B-Chat", "size": "7B"},
        {"id": "deepseek-ai/deepseek-coder-1.3b-base", "name": "DeepSeek-Coder-1.3B", "size": "1.3B"},
        {"id": "facebook/opt-125m", "name": "OPT-125M", "size": "125M"},
        {"id": "gpt2", "name": "GPT2", "size": "124M"},
    ]
    
    test_results = []
    
    for model_info in models_to_test:
        if st.button(f"Test {model_info['name']}", key=f"test_{model_info['id'].replace('/', '_')}"):  # ✅ UNIQUE KEY
            success, message, response = test_model_loading(model_info["id"], model_info["name"])
            test_results.append({
                "model": model_info["name"],
                "success": success,
                "message": message,
                "response": response,
                "size": model_info["size"]
            })
    
    # Display test results
    if test_results:
        st.markdown("### 📊 Test Results")
        for result in test_results:
            if result["success"]:
                st.success(f"{result['model']} ({result['size']}): {result['message']}")
                st.text(f"Test response: {result['response']}")
            else:
                st.error(f"{result['model']}: {result['message']}")
    
    # Production model loading
    st.markdown("---")
    st.markdown("### 🚀 Production Model")
    
    # Add memory warning
    st.info("💡 **Memory Notes**:\n- DeepSeek-7B: ~14GB RAM required\n- DeepSeek-1.3B: ~3GB RAM required\n- OPT/GPT2: ~500MB RAM required")
    
    selected_model = st.selectbox(
        "Choose model for production:",
        options=[m["id"] for m in models_to_test],
        format_func=lambda x: next((m["name"] for m in models_to_test if m["id"] == x), x),
        key="model_selector"
    )
    
    if st.button("Load Selected Model", key="load_model_btn"):
        with st.spinner(f"Loading {selected_model}..."):
            try:
                # Clear any previously loaded model from memory
                if hasattr(st.session_state, 'llm_model') and st.session_state.llm_model is not None:
                    del st.session_state.llm_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                    selected_model,
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if st.session_state.tokenizer.pad_token is None:
                    st.session_state.tokenizer.pad_token = st.session_state.tokenizer.eos_token
                
                # Load model with optimized settings for larger models
                st.session_state.llm_model = AutoModelForCausalLM.from_pretrained(
                    selected_model,
                    torch_dtype=torch.float16,  # Use half precision to save memory
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                
                # If no GPU, move to CPU explicitly
                if not torch.cuda.is_available():
                    st.session_state.llm_model = st.session_state.llm_model.to("cpu")
                
                st.session_state.llm_device = st.session_state.llm_model.device
                st.success(f"✅ {selected_model} loaded for production!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to load {selected_model}: {e}")
                # If DeepSeek fails, try with even lower precision
                if "deepseek" in selected_model.lower():
                    st.info("🔄 Trying with lower precision...")
                    try:
                        st.session_state.llm_model = AutoModelForCausalLM.from_pretrained(
                            selected_model,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map="cpu",
                        )
                        st.session_state.llm_model = st.session_state.llm_model.to("cpu")
                        st.session_state.llm_device = torch.device("cpu")
                        st.success(f"✅ {selected_model} loaded on CPU with float32!")
                        st.rerun()
                    except Exception as e2:
                        st.error(f"❌ DeepSeek loading failed even with CPU: {e2}")
    
    # Load other models and store in session state
    with st.spinner("Loading analysis models..."):
        if st.session_state.resnet_model is None:
            st.session_state.resnet_model = load_old_resnet_feature_extractor(device="cpu")
        
        if st.session_state.reg_model is None or st.session_state.feature_scaler is None:
            st.session_state.reg_model, st.session_state.feature_scaler = load_regression_components()
        
        if st.session_state.classifier is None:
            st.session_state.classifier = load_pce_classifier(CLASSIFIER_PATH)

    # Model status dashboard (using session state)
    st.markdown("### 📊 Model Status")
    
    model_status = {
        "LLM": "✅ Loaded" if st.session_state.llm_model is not None else "❌ Not loaded",
        "Classifier": "✅ Loaded" if st.session_state.classifier is not None else "❌ Failed", 
        "ResNet": "✅ Loaded" if st.session_state.resnet_model is not None else "❌ Failed",
        "Regression": "✅ Loaded" if st.session_state.reg_model is not None else "❌ Failed"
    }
    
    for model, status in model_status.items():
        st.write(f"{model}: {status}")
    
    # ✅ ADD GENERATION SETTINGS SECTION
    st.markdown("---")
    st.markdown("### 🔧 Generation Settings")
    
    max_tokens = st.slider("Max tokens", 64, 1024, value=150, step=16, key="max_tokens_slider")
    temp = st.slider("Temperature", 0.0, 1.0, value=0.3, step=0.05, key="temp_slider")
    sys_prompt = st.text_area("System prompt", 
        value="""You are an expert photovoltaic (PV) electroluminescence (EL) imaging analyst with 15 years of experience in solar cell defect detection.
    
    DOMAIN EXPERTISE CONTEXT:
    - EL imaging reveals internal defects in solar cells through luminescence patterns
    - HIGH efficiency cells show uniform, bright luminescence indicating good quality
    - LOW efficiency cells show dark areas, cracks, shunts, or non-uniform patterns indicating defects
    - PCE (Power Conversion Efficiency) measures solar cell performance (10-18% = good, <10% = poor)
    - Common defects: microcracks, finger interruptions, shunt resistance, inactive regions
    
    ANALYSIS FRAMEWORK:
    1. Examine efficiency distribution across the batch
    2. Identify defect patterns in low-efficiency images  
    3. Assess PCE consistency in high-efficiency group
    4. Provide quality assessment and technical insights
    5. Recommend practical next steps for investigation
    
    RESPONSE REQUIREMENTS:
    - Be analytical and data-driven using the provided statistics
    - Focus on actual results, not general knowledge
    - Use proper solar cell terminology
    - Provide actionable insights for engineers
    - Structure responses clearly with specific observations
    
    CRITICAL: Always base your analysis on the provided EL image analysis results.""", 
        height=180, key="sys_prompt_area")
    # Quick test with loaded model - test context understanding
    if st.session_state.llm_model is not None:
        if st.button("🧪 Test Current LLM", key="test_current_model"):
            # Create a test context similar to real analysis
            test_context = """EL IMAGE ANALYSIS RESULTS:
            - Total images analyzed: 5
            - High efficiency solar cells: 3 (60.0%)
            - Low efficiency solar cells: 2 (40.0%)
            - PCE range (high efficiency): 12.50% to 18.20%
            - Average PCE (high efficiency): 15.23%
            - PCE QUALITY: 3/3 high efficiency cells ≥10%
            - Low efficiency images: image3.jpg, image5.jpg
            - BATCH QUALITY: Mixed batch with significant variation
            
            DETAILED RESULTS:
            1. image1.jpg: High efficiency, PCE: 12.50%
            2. image2.jpg: High efficiency, PCE: 14.80%
            3. image3.jpg: Low efficiency, PCE: N/A
            4. image4.jpg: High efficiency, PCE: 18.20%
            5. image5.jpg: Low efficiency, PCE: N/A"""
            with st.spinner("Testing context understanding..."):
                test_response = generate_llm_response(
                    sys_prompt,
                    f"{test_context}\n\nQuestion: How many high efficiency solar cells were found?",
                    max_new_tokens=30,
                    temperature=0.1
                )
            st.info(f"Test Response: '{test_response}'")
            
            # Check if LLM understood the context
            if "3" in test_response or "three" in test_response.lower():
                st.success("✅ LLM understands analysis context!")
            else:
                st.warning("⚠️ LLM may not be using the provided context properly")
# ------------------------- UI -------------------------
st.title("🔍 EL Image Efficiency Agent")
uploaded_files = st.file_uploader("📤 Upload EL images", accept_multiple_files=True, type=["png","jpg","jpeg"])

# ✅ ADD THIS WARNING MESSAGE:
if not hasattr(st.session_state, 'llm_model') or st.session_state.llm_model is None:
    st.warning("""
    ⚠️ **LLM Note**: No language model is currently loaded. 
    - Image classification and analysis will still work
    - LLM features (summaries, Q&A) are disabled
    - Please select and load a model in the sidebar
    """)
    
col_run, col_clear, col_dl = st.columns([1,1,2])
run_batch = col_run.button("🔎 Classify & Summarize Batch", width=180)
clear_chat = col_clear.button("🧹 Clear LLM Chat", width=180)

# CSV download (sidebar area)
if st.session_state.results:
    csv_data = []
    for result in st.session_state.results:
        pce_display = f"{result['pce']}%" if result["pce"] != "—" else "N/A"
        csv_data.append({
            "file": result["file"],
            "class_prediction": result["class_prediction"],
            "pce": pce_display
        })
    csv_bytes = pd.DataFrame(csv_data).to_csv(index=False).encode()
    col_dl.download_button("⬇️ Download CSV", csv_bytes, file_name="el_results.csv", 
                         mime="text/csv", width=220, key="main_csv_download")
    
# ------------------------- PROCESS UPLOAD -------------------------
if uploaded_files:
    names = [f.name for f in uploaded_files]
    if names != st.session_state.last_upload_names:
        st.session_state.results = []
        st.session_state.last_upload_names = names
    
    if st.session_state.classifier is None or st.session_state.resnet_model is None:
        st.error("❌ Cannot process images: Required models (Classifier or ResNet) are not loaded")
    else:
        cols = st.columns(2)
        for idx, uploaded in enumerate(uploaded_files):
            col = cols[idx % 2]
            img_bytes = uploaded.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # ---------------- Old ResNet Feature Extraction ----------------
            feats_cls = extract_resnet_features_old(pil_img, st.session_state.resnet_model, device="cpu").reshape(1, -1)  # ✅ FIXED

            # ---------------- Classification ----------------
            cls_raw = st.session_state.classifier.predict_from_features(feats_cls)  # ✅ FIXED
            parsed = parse_classifier_result(cls_raw)

            # ---------------- Regression (PCE) ----------------
            pce_val = predict_pce_high_only(pil_img, parsed["prediction"], st.session_state.reg_model, st.session_state.feature_scaler)

            # ---------------- Store Result ----------------
            rec = {
                "file": uploaded.name,
                "class_prediction": parsed["prediction"],
                "pce": pce_val
            }
            existing = next((r for r in st.session_state.results if r["file"] == uploaded.name), None)
            if existing:
                existing.update(rec)
            else:
                st.session_state.results.append(rec)
            # ---------------- Display ----------------
            with col:
                st.image(pil_img, caption=uploaded.name, width=IMAGE_DISPLAY_WIDTH)

                # Choose styling
                if rec["class_prediction"] == "High":
                    cls_style = "pred-high"
                    icon = "🔆"
                elif rec["class_prediction"] == "Low":
                    cls_style = "pred-low"
                    icon = "⚠️"
                elif rec["class_prediction"] == "Error":
                    cls_style = "pred-error"
                    icon = "❌"
                else:
                    cls_style = "pred-unknown"
                    icon = "❓"

                st.markdown(
                    f"<div class='prediction-card {cls_style}'>"
                    f"<div class='pred-header'>{icon} {rec['class_prediction']}</div>"
                    # f"<div class='pred-sub'>Confidence</div>"
                    # f"<div class='pred-percent'>{rec['class_confidence']*100:.1f}%</div>"
                    f"</div>", unsafe_allow_html=True
                )

                # PCE display
                if rec["pce"] != "—":
                    st.markdown(f"<div class='pce-highlight'>Predicted PCE: {rec['pce']}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='pred-sub' style='text-align: center;'>PCE: {rec['pce']}</div>", unsafe_allow_html=True)
                # ---------------- Per-image LLM question ----------------
                q = st.text_input(f"Question for {uploaded.name}", key=f"q_{uploaded.name}_{idx}", 
                                  value="What could cause this efficiency level?")
                if st.button("💬 Ask LLM", key=f"ask_{uploaded.name}_{idx}", width=200):
                    # Build rich context for this specific image
                    image_context = f"""
                SPECIFIC IMAGE ANALYSIS:
                - Image: {uploaded.name}
                - Classification: {rec['class_prediction']} efficiency
                - PCE: {rec['pce']}
                - Batch context: {len(st.session_state.results)} total images analyzed
                """
                    
                    # For single image questions, also include batch context
                    batch_context = build_llm_context(st.session_state.results)
                    full_context = f"{batch_context}\n\n{image_context}"
                    
                    prompt_text = f"{full_context}\n\nQuestion: {q}"
                    resp = generate_llm_response(sys_prompt, prompt_text, max_tokens, temp)
                    st.session_state.chat_history.append({"role":"user","text":f"[{uploaded.name}] {q}"})
                    st.session_state.chat_history.append({"role":"llm","text":resp})
                    st.rerun()

# ---------------- Batch LLM Summary ----------------
if run_batch and st.session_state.results:
    if st.session_state.classifier is None:
        st.error("❌ Cannot generate summary: Classifier model not loaded")
    else:
        context_text = "\n".join([
            f"{i+1}. {r['file']}: {r['class_prediction']}— PCE: {r['pce']}"
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

    # show table - convert PCE to string for display
    # ---------------- Display Batch Table ----------------
    if st.session_state.results:
        st.markdown("### Batch results")
        display_data = []
        for result in st.session_state.results:
            display_data.append({
                "file": result["file"],
                "class_prediction": result["class_prediction"],
                "pce": f"{result['pce']}%" if result["pce"] != "—" else "N/A"
            })
        st.dataframe(pd.DataFrame(display_data), width=900)

# ------------------------- BATCH LLM SUMMARY -------------------------
if run_batch and st.session_state.results:
    if st.session_state.classifier is None:
        st.error("❌ Cannot generate summary: Classifier model not loaded")
    else:
        # Build comprehensive context from ALL analysis results
        rich_context = build_llm_context(st.session_state.results)
        
        final_prompt = f"{rich_context}\n\nQuestion: {user_prompt_default}"
        
        llm_raw = generate_llm_response(sys_prompt, final_prompt, max_tokens, temp)
        # Don't clean the response as aggressively for batch summaries
        cleaned_response = llm_raw
        
        st.session_state.chat_history.append({"role":"user","text":"Batch analysis summary request"})
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

    st.markdown("---")
    st.markdown("### 💭 Ask a Question")
    with st.container():
        st.markdown('<div class="user-input-section">', unsafe_allow_html=True)
        st.info("💡 **For best results**: Ask specific questions about solar cell EL analysis. Examples: 'What defects are visible?', 'How does efficiency vary?', 'Compare high vs low efficiency patterns.'")
        user_question = st.text_area(
            "Enter your question:",
            placeholder="e.g., How many high efficiency images? List low efficiency ones. What are the PCE values?",
            height=80,
            key="custom_question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_custom = st.button("🚀 Ask LLM", key="ask_custom_btn", width=160)
        with col2:
            clear_input = st.button("🧹 Clear Input", key="clear_input_btn", width=120)
        st.markdown('</div>', unsafe_allow_html=True)
    
        # ✅ CUSTOM QUESTION HANDLER - ADDED HERE
        if ask_custom and user_question.strip():
            if st.session_state.results:
                # Use context-aware prompt building
                context_prompt = build_context_aware_prompt(st.session_state.results, user_question)
                full_prompt = context_prompt
            else:
                full_prompt = user_question
            
            resp = generate_llm_response(sys_prompt, full_prompt, max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":user_question})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    
        if clear_input:
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ EL Analysis Questions")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("Defect Analysis", key="defect_analysis_btn", width=220):
            if st.session_state.results:
                context_prompt = build_context_aware_prompt(
                    st.session_state.results, 
                    "Analyze the defect patterns in low efficiency solar cells and suggest root causes."
                )
                resp = generate_llm_response(sys_prompt, context_prompt, max_tokens, temp)
            else:
                resp = generate_llm_response(sys_prompt, "Analyze solar cell defect patterns.", max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":"Defect analysis"})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    
    with col_f2:
        if st.button("PCE Analysis", key="pce_analysis_btn", width=220):
            if st.session_state.results:
                context_prompt = build_context_aware_prompt(
                    st.session_state.results,
                    "Analyze the PCE distribution and performance quality of high efficiency solar cells."
                )
                resp = generate_llm_response(sys_prompt, context_prompt, max_tokens, temp)
            else:
                resp = generate_llm_response(sys_prompt, "Analyze PCE values.", max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":"PCE analysis"})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        if st.button("Batch Summary", key="batch_summary_btn", width=220):
            if st.session_state.results:
                context_prompt = build_context_aware_prompt(
                    st.session_state.results,
                    "Provide a comprehensive summary of the EL image analysis batch results with quality assessment."
                )
                resp = generate_llm_response(sys_prompt, context_prompt, max_tokens, temp)
            else:
                resp = generate_llm_response(sys_prompt, "Summarize the analysis results.", max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":"Batch summary"})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    
    with col_f4:
        if st.button("Next Steps", key="next_steps_btn", width=220):
            if st.session_state.results:
                context_prompt = build_context_aware_prompt(
                    st.session_state.results,
                    "Recommend practical next steps for investigation based on the EL analysis results."
                )
                resp = generate_llm_response(sys_prompt, context_prompt, max_tokens, temp)
            else:
                resp = generate_llm_response(sys_prompt, "Suggest next steps.", max_tokens, temp)
            st.session_state.chat_history.append({"role":"user","text":"Next steps"})
            st.session_state.chat_history.append({"role":"llm","text":resp})
            st.rerun()
    with chat_col_right:
        st.markdown("### Controls")
        if st.button("Clear chat", key="clear_chat_history", width=160):
            st.session_state.chat_history=[]
            st.rerun()
        st.markdown("---")
        st.markdown("**Export**")
        if st.session_state.results:
            export_data = []
            for result in st.session_state.results:
                export_data.append({
                    "file": result["file"],
                    "class_prediction": result["class_prediction"],
                    "pce": f"{result['pce']}%" if result["pce"] != "—" else "N/A"
                })
            df_export = pd.DataFrame(export_data)
            st.download_button("⬇️ Download CSV", df_export.to_csv(index=False).encode(), 
                             file_name="el_results.csv", mime="text/csv", width=260,
                             key="controls_csv_download")
        st.caption("Select and load a model in the sidebar to enable LLM features.")
