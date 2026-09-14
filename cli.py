import sys
import argparse
import cv2
from core.fingerprint import Fingerprint
from core.matchers import HoughMatcher, GeneticMatcher

def main():
    parser = argparse.ArgumentParser(description="Fingerprint Recognition CLI Tester")
    parser.add_argument("-p", "--probe", required=True, help="Path to the probe fingerprint image")
    parser.add_argument("-g", "--gallery", required=True, help="Path to the gallery fingerprint image")
    parser.add_argument("-m", "--matcher", choices=["hough", "genetic"], default="hough", 
                        help="Choose the matching algorithm (default: hough)")
    
    args = parser.parse_args()

    print(f"[*] Loading Probe: {args.probe}")
    probe_img = cv2.imread(args.probe, cv2.IMREAD_GRAYSCALE)
    if probe_img is None:
        print(f"[!] Error: Could not load probe image at {args.probe}")
        sys.exit(1)

    print(f"[*] Loading Gallery: {args.gallery}")
    gallery_img = cv2.imread(args.gallery, cv2.IMREAD_GRAYSCALE)
    if gallery_img is None:
        print(f"[!] Error: Could not load gallery image at {args.gallery}")
        sys.exit(1)

    # 1. Initialize Fingerprint Objects
    probe_fp = Fingerprint(probe_img, name="Probe")
    gallery_fp = Fingerprint(gallery_img, name="Gallery")

    # 2. Extract Features
    print("[*] Extracting features from Probe...")
    probe_fp.extract_features()
    print(f"    -> Extracted {len(probe_fp.minutiae)} minutiae.")

    print("[*] Extracting features from Gallery...")
    gallery_fp.extract_features()
    print(f"    -> Extracted {len(gallery_fp.minutiae)} minutiae.")

    # 3. Match
    print(f"[*] Running {args.matcher.upper()} matching algorithm...")
    if args.matcher == "hough":
        matcher = HoughMatcher()
        threshold = 10.0
    else:
        matcher = GeneticMatcher()
        threshold = 50.0

    score = matcher.match(probe_fp, gallery_fp)

    # 4. Results
    print("\n" + "="*40)
    print(f"FINAL SCORE: {score:.2f} (Threshold: {threshold})")
    if score >= threshold:
        print("VERDICT: ✅ MATCH (Genuine)")
    else:
        print("VERDICT: ❌ NO MATCH (Impostor)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()