import os
import glob
from typing import Dict, Tuple
from core.fingerprint import Fingerprint
from core.matchers import BaseMatcher
import pickle

class FingerprintDatabase:
    def __init__(self):
        self.templates: Dict[str, Fingerprint] = {}

    def enroll_directory(self, db_path: str):
        """Extracts and caches minutiae for all images in the database."""
        self.templates.clear()
        supported_exts = ('*.tif', '*.png', '*.jpg', '*.bmp')
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(db_path, ext)))

        for file_path in image_files:
            name = os.path.basename(file_path)
            fp = Fingerprint(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), name=name)
            fp.extract_features()
            self.templates[name] = fp

    def identify(self, probe: Fingerprint, matcher: BaseMatcher, threshold: float) -> Tuple[str, float]:
        """1:N Matching: Compares the probe against all enrolled templates."""
        best_score = 0.0
        best_match = None

        for name, gallery_fp in self.templates.items():
            score = matcher.match(probe, gallery_fp)
            if score > best_score:
                best_score = score
                best_match = name
            
            # Early stopping for extreme confidence
            if best_score > threshold * 1.5:
                break

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

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