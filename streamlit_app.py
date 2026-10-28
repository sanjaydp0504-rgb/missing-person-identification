
import os
import sqlite3
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "images")
DATABASE = os.path.join(BASE_DIR, "missing_persons.db")

DETECTOR_MODEL = os.path.join(
    BASE_DIR, "face_detection_yunet_2023mar.onnx"
)
RECOGNIZER_MODEL = os.path.join(
    BASE_DIR, "face_recognition_sface_2021dec.onnx"
)

st.set_page_config(
    page_title="Missing Person Identification",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Missing Person Identification System")
st.write(
    "Upload a clear photo to search for possible matches "
    "in the available missing-person records."
)

# Check required files
required_files = [DETECTOR_MODEL, RECOGNIZER_MODEL]

missing_files = [
    os.path.basename(path)
    for path in required_files
    if not os.path.isfile(path)
]

if missing_files:
    st.error("Required model file(s) not found: " + ", ".join(missing_files))
    st.stop()


# Database setup
def setup_database():
    with sqlite3.connect(DATABASE) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE,
                name TEXT NOT NULL,
                age INTEGER,
                last_seen TEXT,
                image TEXT
            )
        """)

        columns = [
            row[1] for row in db.execute(
                "PRAGMA table_info(persons)"
            ).fetchall()
        ]

        if "image" not in columns:
            db.execute("ALTER TABLE persons ADD COLUMN image TEXT")

        count = db.execute(
            "SELECT COUNT(*) FROM persons"
        ).fetchone()[0]

        if count == 0:
            samples = [
                ("MP001", "Sample Person A", 25, "Chennai", "person1.jpeg"),
                ("MP002", "Sample Person B", 30, "Coimbatore", "person2.jpeg"),
                ("MP003", "Sample Person C", 22, "Madurai", "person3.jpeg"),
            ]

            db.executemany("""
                INSERT INTO persons
                (case_id, name, age, last_seen, image)
                VALUES (?, ?, ?, ?, ?)
            """, samples)


def get_person_records():
    with sqlite3.connect(DATABASE) as db:
        db.row_factory = sqlite3.Row
        return db.execute(
            "SELECT * FROM persons ORDER BY id"
        ).fetchall()


# Initialize face models
@st.cache_resource
def load_face_models():
    detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL, "", (320, 320), 0.6, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(
        RECOGNIZER_MODEL, ""
    )
    return detector, recognizer


def extract_face_features(image, detector, recognizer):
    height, width = image.shape[:2]

    if width < 20 or height < 20:
        return None

    scales = [1.0]
    if max(width, height) < 1000:
        scales.append(min(2.0, 1000 / max(width, height)))

    for scale in scales:
        if scale == 1.0:
            test_image = image
        else:
            test_image = cv2.resize(
                image, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        h, w = test_image.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(test_image)

        if faces is None or len(faces) == 0:
            continue

        face = max(faces, key=lambda item: float(item[2] * item[3]))
        aligned_face = recognizer.alignCrop(test_image, face)
        features = recognizer.feature(aligned_face)

        if features is not None:
            return features

    return None


def image_to_bgr(pil_image):
    rgb = np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compare_faces(features1, features2, recognizer):
    return float(
        recognizer.match(
            features1,
            features2,
            cv2.FaceRecognizerSF_FR_COSINE
        )
    )


# Start application
setup_database()
detector, recognizer = load_face_models()

st.subheader("Upload a photo")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("Please upload an image smaller than 5 MB.")
        st.stop()

    try:
        uploaded_pil = Image.open(uploaded_file)
        uploaded_pil.verify()
        uploaded_file.seek(0)
        uploaded_pil = Image.open(uploaded_file)
        uploaded_image = image_to_bgr(uploaded_pil)

        st.image(uploaded_pil, caption="Uploaded photo", width=250)

        if st.button("Search for Possible Matches", type="primary"):
            with st.spinner("Analyzing the photo..."):
                uploaded_features = extract_face_features(
                    uploaded_image, detector, recognizer
                )

                if uploaded_features is None:
                    st.warning(
                        "No face detected. Please upload a clear photo "
                        "with a visible face."
                    )
                else:
                    best_person = None
                    best_score = float("-inf")

                    for person in get_person_records():
                        image_name = person["image"]

                        if not image_name:
                            continue

                        if os.path.basename(image_name) != image_name:
                            continue

                        stored_path = os.path.join(
                            IMAGE_FOLDER, image_name
                        )

                        if not os.path.isfile(stored_path):
                            continue

                        try:
                            with Image.open(stored_path) as stored_pil:
                                stored_image = image_to_bgr(stored_pil)

                            stored_features = extract_face_features(
                                stored_image, detector, recognizer
                            )

                            if stored_features is None:
                                continue

                            score = compare_faces(
                                uploaded_features,
                                stored_features,
                                recognizer
                            )

                            if score > best_score:
                                best_score = score
                                best_person = person

                        except Exception as error:
                            print("Comparison error:", error)

                    # Starting threshold; requires validation on real data.
                    MATCH_THRESHOLD = 0.363

                    if best_person is None:
                        st.warning(
                            "No usable faces were found in the stored "
                            "record photos."
                        )
                    elif best_score >= MATCH_THRESHOLD:
                        st.success("Possible candidate found")

                        st.write("**Case ID:**", best_person["case_id"])
                        st.write("**Name:**", best_person["name"])
                        st.write("**Age:**", best_person["age"])
                        st.write("**Last seen / location:**",
                                  best_person["last_seen"])
                        st.write("**Similarity score:**",
                                 round(best_score, 3))

                        matched_image = os.path.join(
                            IMAGE_FOLDER, best_person["image"]
                        )
                        if os.path.isfile(matched_image):
                            st.image(
                                matched_image,
                                caption="Record photo",
                                width=250
                            )

                        st.warning(
                            "This is only a possible match, not a confirmed "
                            "identity. A qualified person must verify it "
                            "independently."
                        )
                    else:
                        st.info(
                            "No matching record found above the configured "
                            "threshold."
                        )

    except Exception:
        st.error("Unable to process this image. Please try a valid photo.")


st.divider()
st.subheader("Available Missing-Person Records")

search = st.text_input("Search by case ID or name")

records = get_person_records()
filtered_records = [
    person for person in records
    if search.lower() in str(person["case_id"]).lower()
    or search.lower() in str(person["name"]).lower()
]

if not filtered_records:
    st.info("No records found.")
else:
    for person in filtered_records:
        with st.expander(
            f'{person["case_id"]} — {person["name"]}'
        ):
            st.write("Age:", person["age"])
            st.write("Last seen / location:", person["last_seen"])

            image_name = person["image"]
            if image_name and os.path.basename(image_name) == image_name:
                image_path = os.path.join(IMAGE_FOLDER, image_name)
                if os.path.isfile(image_path):
                    st.image(image_path, width=180)

st.caption(
    "Demonstration project. Sample records are fictional. "
    "Face similarity alone must never be used to establish identity."
)
