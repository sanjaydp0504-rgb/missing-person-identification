
import cv2
import os

detector_path = "face_detection_yunet_2023mar.onnx"
recognizer_path = "face_recognition_sface_2021dec.onnx"

print("Checking model files...")

if not os.path.exists(detector_path):
    print("Face detector model is missing!")
    raise SystemExit

if not os.path.exists(recognizer_path):
    print("Face recognizer model is missing!")
    raise SystemExit

print("Both model files exist.")

try:
    detector = cv2.FaceDetectorYN.create(
        detector_path,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        recognizer_path,
        ""
    )

    print("Face detector loaded successfully!")
    print("Face recognizer loaded successfully!")
    print("Model setup is ready.")

except Exception as error:
    print("Model loading failed:")
    print(error)
