import os
import glob
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import pickle

from core.fingerprint import Fingerprint
from core.matchers import HoughMatcher, GeneticMatcher, BaseMatcher

# --- Internal Database Class ---
class FingerprintDatabase:
    """Handles 1:N Enrollment and Identification."""
    def __init__(self):
        self.templates: Dict[str, Fingerprint] = {}

    def save_to_disk(self, filepath: str = "database.pkl"):
        """Serializes the enrolled templates to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.templates, f)

    def load_from_disk(self, filepath: str = "database.pkl") -> bool:
        """Loads templates from a serialized file. Returns True if successful."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.templates = pickle.load(f)
            return True
        return False

    def enroll_directory(self, db_path: str):
        self.templates.clear()
        supported_exts = ('*.tif', '*.png', '*.jpg', '*.bmp')
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(db_path, ext)))

        for file_path in image_files:
            name = os.path.basename(file_path)
            # Load and process the fingerprint immediately
            raw_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is not None:
                fp = Fingerprint(raw_img, name=name)
                fp.extract_features()
                self.templates[name] = fp

    def identify(self, probe: Fingerprint, matcher: BaseMatcher, threshold: float, top_k: int = 2):
        """1:N Matching: Compares the probe against all enrolled templates and returns the top K matches."""
        results = []

        for name, gallery_fp in self.templates.items():
            score = matcher.match(probe, gallery_fp)
            if score >= threshold:
                results.append((name, score))

        # Sort the results by score in descending order (highest score first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the top_k elements
        return results[:top_k]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="1:N Fingerprint Identification", layout="wide")

# Initialize Session State Database
if 'db' not in st.session_state:
    st.session_state.db = FingerprintDatabase()

st.title("🔍 1:N Fingerprint Identification System")
st.markdown("""
This application mimics a real-world Automated Fingerprint Identification System (AFIS). 
First, enroll a database of known subjects. Then, upload a probe image to search the database for a match[cite: 2].
""")

# --- Sidebar Controls ---
st.sidebar.header("1. Database Enrollment")

# Auto-load the database on startup if the file exists
if 'db_loaded' not in st.session_state:
    if st.session_state.db.load_from_disk("database.pkl"):
        st.session_state.db_loaded = True
        st.sidebar.success(f"✅ Loaded {len(st.session_state.db.templates)} records from disk.")
    else:
        st.session_state.db_loaded = False

db_path = st.sidebar.text_input("FVC 2000 Database Path", value="data/DB1")

if st.sidebar.button("Enroll Directory (Heavy CPU)", type="secondary"):
    if os.path.exists(db_path) and os.path.isdir(db_path):
        with st.spinner("Extracting minutiae..."):
            st.session_state.db.enroll_directory(db_path)
            # Automatically save to disk after enrolling
            st.session_state.db.save_to_disk("database.pkl")
            st.session_state.db_loaded = True
        st.sidebar.success(f"Enrolled and saved {len(st.session_state.db.templates)} fingerprints.")
    else:
        st.sidebar.error("Directory not found.")

st.sidebar.header("2. Identification Configuration")
st.sidebar.markdown("**Matching Algorithm:** Generalized Hough Transform")

st.sidebar.header("3. Upload Probe")
probe_file = st.sidebar.file_uploader("Upload a Probe Image to Search", type=["png", "jpg", "tif", "bmp"])

# --- Main Logic ---
if probe_file:
    # Validate database has templates
    if len(st.session_state.db.templates) == 0:
        st.warning("⚠️ The database is currently empty. Please 'Enroll Database' using the sidebar first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        # Load Probe Image (using OpenCV byte decoder for stable .tif support)
        file_bytes = np.asarray(bytearray(probe_file.read()), dtype=np.uint8)
        probe_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        with col1:
            st.subheader("Probe Fingerprint")
            st.image(probe_img, use_container_width=True, caption="Raw Input")

        with col2:
            st.subheader("Action")
            if st.button("Identify Subject in Database", type="primary"):
                with st.spinner(f"Searching {len(st.session_state.db.templates)} records using Hough Transform..."):
                    
                    # 1. Extract Probe Features
                    probe_fp = Fingerprint(probe_img, name="Probe")
                    probe_fp.extract_features()
                    
                    # 2. Select Strategy and Threshold
                    matcher = HoughMatcher()
                    threshold = 35.0
                    
                    # 3. Identify 1:N (Fetching top 2 matches)
                    top_matches = st.session_state.db.identify(probe_fp, matcher, threshold, top_k=2)
                    
                    # 4. Display Results
                    st.markdown("---")
                    st.subheader("Search Results")
                    
                    if top_matches:
                        st.success(f"✅ **SUBJECT IDENTIFIED:** Found {len(top_matches)} matches above threshold.")
                        
                        # Create columns dynamically based on how many matches were found
                        match_cols = st.columns(len(top_matches))
                        
                        for idx, (match_name, score) in enumerate(top_matches):
                            with match_cols[idx]:
                                st.metric(
                                    label=f"Rank {idx+1} Match", 
                                    value=match_name, 
                                    delta=f"Score: {score:.2f}", 
                                    delta_color="off"
                                )
                                
                                matched_fp_obj = st.session_state.db.templates[match_name]
                                
                                # Draw Minutiae on Matched DB Image
                                matched_vis = cv2.cvtColor(matched_fp_obj.raw_image, cv2.COLOR_GRAY2RGB)
                                for m in matched_fp_obj.minutiae:
                                    cv2.circle(matched_vis, (m.x, m.y), 3, (255, 0, 0), -1)
                                
                                st.image(matched_vis, caption=f"{match_name} Minutiae", use_container_width=True)
                                st.image(matched_fp_obj.thinned_image * 255.0, caption=f"{match_name} Skeleton", use_container_width=True, clamp=True)
                                
                        st.markdown("### Probe Verification")
                        with st.expander("View Probe Minutiae Map"):
                            # Draw Minutiae on Probe Image
                            probe_vis = cv2.cvtColor(probe_fp.raw_image, cv2.COLOR_GRAY2RGB)
                            for m in probe_fp.minutiae:
                                cv2.circle(probe_vis, (m.x, m.y), 3, (255, 0, 0), -1)
                            
                            st.image(probe_vis, caption=f"Probe Minutiae ({len(probe_fp.minutiae)} points)", use_container_width=True)
                            st.image(probe_fp.thinned_image * 255.0, caption="Probe Skeleton", use_container_width=True, clamp=True)
                            
                    else:
                        st.error("❌ **UNKNOWN SUBJECT:** No matching profile found in the database.")
else:
    st.info("Upload a probe fingerprint image to begin identification.")