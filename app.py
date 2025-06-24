import streamlit as st
from utils.transcribe import AudioTranscriber
from utils.extract_info import InfoExtractor
from qa_system import QASystem
import tempfile
import os
import torch
import logging
from typing import Optional, Dict, Any

# --------------------------
# CONFIGURATION
# --------------------------
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'hf_models')

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# --------------------------
# CORE FUNCTIONS
# --------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> Optional[Dict[str, Any]]:
    """Initialize and cache all AI models with comprehensive error handling"""
    try:
        logger.info("Loading models...")
        transcriber = AudioTranscriber("base")
        transcriber.set_progress_callback(lambda p: None)  # Initialize empty callback
        return {
            "transcriber": transcriber,
            "extractor": InfoExtractor(),
            "qa": QASystem()
        }
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        st.error("""
        ❌ System initialization failed. Common fixes:
        1. Restart the app
        2. Check internet connection
        3. Verify sufficient disk space
        """)
        st.stop()
        return None

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Safely handle file uploads with validation"""
    try:
        # Create dedicated temp directory
        temp_dir = tempfile.mkdtemp(prefix="audio_")
        audio_path = os.path.join(temp_dir, f"audio_input_{os.urandom(4).hex()}.mp3")
        
        # Write with verification
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate file
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Empty file uploaded")
            
        logger.info(f"File saved to: {audio_path} (Size: {os.path.getsize(audio_path)} bytes)")
        return audio_path
        
    except Exception as e:
        logger.error(f"File handling error: {str(e)}", exc_info=True)
        st.error("⚠️ File upload failed. Please try another file.")
        return None

def transcribe_audio(models: Dict[str, Any], audio_path: str) -> Optional[str]:
    """Robust audio transcription with progress tracking"""
    try:
        progress_bar = st.progress(0)
        
        def progress_callback(progress: float):
            """Handle progress updates for Streamlit"""
            progress_bar.progress(min(1.0, max(0.0, progress)))
        
        # Set the callback before transcription
        models["transcriber"].set_progress_callback(progress_callback)
        
        with st.spinner("🔊 Processing audio (this may take a few minutes)..."):
            result = models["transcriber"].transcribe(audio_path)
            progress_bar.progress(1.0)
            return result
            
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        progress_bar.progress(1.0)  # Ensure progress bar completes
        st.error(f"""
        🚨 Transcription Error:
        {str(e)}
        
        Try:
        1. Shorter audio files (<5 mins)
        2. Clearer recordings
        3. Different file format
        """)
        return None

# --------------------------
# UI COMPONENTS
# --------------------------
def init_sidebar():
    """Initialize sidebar with useful info"""
    with st.sidebar:
        st.markdown("""
        ## 🎤 About
        This tool analyzes audio to:
        - Convert speech to text
        - Extract contact information
        - Answer questions about content
        
        ## 💡 Tips
        - Use clear recordings
        - Ideal length: 30s-5min
        - Supported formats: MP3, WAV
        """)

def display_transcript(text: str):
    """Interactive transcript display"""
    with st.expander("📜 Full Transcript", expanded=True):
        edited_text = st.text_area(
            "Edit transcript (corrections will improve analysis)",
            value=text,
            height=250,
            key="transcript_editor"
        )
    return edited_text

def display_analysis_results(info):
    """Visualize extracted information"""
    if not info:
        return
        
    st.subheader("🔍 Extracted Information")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Phone Numbers", len(info.phones))
        if info.phones:
            st.code("\n".join(info.phones))
        else:
            st.caption("None detected")
    
    with cols[1]:
        st.metric("Email Addresses", len(info.emails))
        if info.emails:
            st.code("\n".join(info.emails))
        else:
            st.caption("None detected")
            
    with cols[2]:
        st.metric("Person Names", len(info.names))
        if info.names:
            st.code("\n".join(info.names))
        else:
            st.caption("None detected")

# --------------------------
# MAIN APP
# --------------------------
def main():
    # App configuration
    st.set_page_config(
        page_title="AI Speech Analyzer",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize UI
    st.title("🎤 AI Speech Analyzer")
    init_sidebar()
    
    # Load models (cached)
    models = load_models()
    if not models:
        return
    
    # File upload section
    st.subheader("📁 Upload Audio")
    audio_file = st.file_uploader(
        "Drag file here or click to browse",
        type=["mp3", "wav"],
        accept_multiple_files=False,
        help="Maximum file size: 50MB"
    )
    
    if audio_file:
        # Process file upload
        audio_path = save_uploaded_file(audio_file)
        if not audio_path:
            return
            
        try:
            # Transcription pipeline
            text = transcribe_audio(models, audio_path)
            if not text:
                return
                
            edited_text = display_transcript(text)
            
            # Information extraction
            with st.spinner("🔎 Analyzing content..."):
                info = extract_information(models, edited_text)
                display_analysis_results(info)
            
            # Question answering
            st.subheader("❓ Ask About the Content")
            question = st.text_input(
                "Example: 'What was the phone number mentioned?'",
                key="question_input"
            )
            
            if question and st.button("Get Answer", type="primary"):
                with st.spinner("💭 Processing question..."):
                    answer = models["qa"].answer(edited_text, question)
                    if answer:
                        st.success(f"**Answer:** {answer}")
                    else:
                        st.info("No relevant answer found in the text")
                        
        finally:
            # Cleanup resources
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    temp_dir = os.path.dirname(audio_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Cleanup warning: {str(e)}")

# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == "__main__":
    # UI enhancements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Run main app
    main()