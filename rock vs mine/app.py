import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="SONAR - Rock vs Mine Detector",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Futuristic Underwater CSS
st.markdown("""
<style>
    /* Main Background with Dark Submarine Gradient */
    .stApp {
        background: linear-gradient(135deg, #02111B 0%, #001E3D 50%, #050515 100%);
        color: #E2E8F0;
    }
    
    /* Submarine Radar Glassmorphic Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* Glow Text Header */
    .glow-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
        margin-bottom: 10px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid #00F2FE;
        color: #00F2FE;
    }

    /* Input Field Customization */
    .stTextArea textarea {
        background: rgba(2, 17, 27, 0.8) !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 12px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00F2FE !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    }

    /* Custom Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.6);
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
    }

    /* Result Cards */
    .result-box-rock {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        color: #34D399;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
    }

    .result-box-mine {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        color: #FCA5A5;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('rock_vs_mine_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model file (`rock_vs_mine_model.pkl`) load nahi ho paayi. Kripya check karein ki file same folder me maujood hai.")
    st.stop()

# Sidebar Setup
with st.sidebar:
    st.markdown("### 🚢 Submarine Navigation")
    st.markdown("---")
    st.markdown("""
    **System Status:** <span class="status-badge">ONLINE 🟢</span>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("**Sonar Radar Specs:**")
    st.markdown("- **Frequency Channels:** 60")
    st.markdown("- **Classifier:** Logistic Regression")
    st.markdown("- **Object Detection:** Underwater Rock / Naval Mine")
    st.markdown("---")
    st.info("💡 **Tip:** Sample data test karne ke liye niche buttons ka upyog karein.")

# Main Header
st.markdown('<h1 class="glow-header">⚓ SONAR SUBMARINE RADAR</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94A3B8; margin-bottom: 30px;">Deep Ocean Signal Analysis & Object Classification System</p>', unsafe_allow_html=True)

# Main Container
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #00F2FE; margin-top:0;">📡 SONAR Signal Input</h3>
        <p style="color: #94A3B8; font-size: 0.9rem;">Comma se separated 60 numerical sonar frequency values daalein:</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample Data Options
    sample_rock = "0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.2114,0.2281,0.2431,0.3139,0.3484,0.3921,0.4286,0.4902,0.5799,0.6622,0.6916,0.7630,0.8800,0.9455,0.9922,0.9835,0.9561,0.8961,0.7375,0.5945,0.5309,0.4184,0.3424,0.2156,0.1850,0.1922,0.1301,0.0813,0.0798,0.1190,0.1055,0.0812,0.0733,0.0630,0.0435,0.0216,0.0247,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032"
    sample_mine = "0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055"

    btn_col1, btn_col2, _ = st.columns([1, 1, 1])
    
    if 'input_val' not in st.session_state:
        st.session_state['input_val'] = sample_mine

    if btn_col1.button("🪨 Fill Rock Data"):
        st.session_state['input_val'] = sample_rock
        st.rerun()

    if btn_col2.button("💣 Fill Mine Data"):
        st.session_state['input_val'] = sample_mine
        st.rerun()

    input_data_str = st.text_area(
        label="Sonar Frequencies",
        value=st.session_state['input_val'],
        height=180,
        placeholder="Enter 60 floating point values..."
    )

    predict_clicked = st.button("🔍 ANALYZE SONAR SIGNALS")

with col2:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #00F2FE; margin-top:0;">📊 Radar Analysis Output</h3>
    """, unsafe_allow_html=True)
    
    if predict_clicked:
        if input_data_str.strip():
            try:
                input_data = [float(x.strip()) for x in input_data_str.split(',') if x.strip()]
                
                if len(input_data) != 60:
                    st.error(f"❌ Submarine Sensor Fault: Expected 60 frequencies, received **{len(input_data)}**.")
                else:
                    input_data_numpy = np.asarray(input_data).reshape(1, -1)
                    
                    with st.spinner("Processing ocean acoustic signals..."):
                        prediction = model.predict(input_data_numpy)
                        probabilities = model.predict_proba(input_data_numpy)[0] if hasattr(model, "predict_proba") else None

                    st.markdown("<br>", unsafe_allow_html=True)
                    if prediction[0] == 'R':
                        st.markdown("""
                        <div class="result-box-rock">
                            <h2>🪨 ROCK DETECTED</h2>
                            <p style="font-size: 1.1rem; margin:0;">Safe Navigation Route Clear</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-box-mine">
                            <h2>💣 NAVAL MINE DETECTED</h2>
                            <p style="font-size: 1.1rem; margin:0;">⚠️ DANGER: Explosive Threat Ahead</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if probabilities is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Prediction Confidence:**")
                        st.progress(float(probabilities[0] if prediction[0] == 'M' else probabilities[1]))
            except ValueError:
                st.error("⚠️ Invalid Data Format: Please enter numbers separated by commas.")
        else:
            st.info("👈 Enter frequency values on the left panel to scan.")
    else:
        st.info("👈 Awaiting Sonar Signal Input...")

    st.markdown("</div>", unsafe_allow_html=True)