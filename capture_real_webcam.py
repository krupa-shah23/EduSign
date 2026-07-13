"""
capture_real_webcam.py

Quick script to capture ~30 seconds of real signing from a webcam
and export it using the standard pipeline.

Usage:
    python capture_real_webcam.py

This creates sample_landmarks/real_webcam_signing_*.{npy,json} files.
The script captures at 30 FPS for 900 frames (~30 seconds).
"""

from export_landmarks import export_from_webcam

if __name__ == "__main__":
    print("Starting real webcam capture...")
    print("Recording for ~30 seconds (900 frames at 30 FPS).")
    print("Please ensure good lighting and position yourself in frame.")
    print()
    
    try:
        export_from_webcam(
            name="real_webcam_signing",
            output_dir="sample_landmarks",
            max_frames=900,  # ~30 seconds at 30 FPS
            device_index=0,
        )
        print("\n✓ Webcam capture complete. Files saved to sample_landmarks/")
    except KeyboardInterrupt:
        print("\n⚠ Capture interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Please check that:")
        print("  - A webcam is connected and accessible")
        print("  - mediapipe and opencv-python are installed")
        print("  - No other application is using the webcam")
