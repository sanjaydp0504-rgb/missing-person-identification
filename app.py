
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import cv2
import os
import sqlite3
import uuid

app = Flask(__name__)

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "images")
DATABASE = os.path.join(BASE_DIR, "missing_persons.db")

# Face model paths
DETECTOR_MODEL = os.path.join(
    BASE_DIR, "face_detection_yunet_2023mar.onnx"
)
RECOGNIZER_MODEL = os.path.join(
    BASE_DIR, "face_recognition_sface_2021dec.onnx"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Starting threshold; test it with your own photos.
MATCH_THRESHOLD = 0.363

# Load face models
face_detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,       # Lower detection threshold
    0.3,
    5000
)

face_recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def setup_database():
    with get_db() as db:
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

        columns = db.execute(
            "PRAGMA table_info(persons)"
        ).fetchall()

        column_names = [column["name"] for column in columns]

        if "image" not in column_names:
            db.execute(
                "ALTER TABLE persons ADD COLUMN image TEXT"
            )

        count = db.execute(
            "SELECT COUNT(*) FROM persons"
        ).fetchone()[0]

        # Insert fictional examples only if the database is empty.
        if count == 0:
            samples = [
                ("MP001", "Sample Person A", 25, "Chennai",
                 "person1.jpeg"),
                ("MP002", "Sample Person B", 30, "Coimbatore",
                 "person2.jpeg"),
                ("MP003", "Sample Person C", 22, "Madurai",
                 "person3.jpeg"),
            ]

            db.executemany("""
                INSERT INTO persons
                (case_id, name, age, last_seen, image)
                VALUES (?, ?, ?, ?, ?)
            """, samples)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def read_image(image_path):
    """Read an image and correct its orientation when possible."""
    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")

            # Save an RGB copy for OpenCV processing.
            import numpy as np
            rgb_image = np.array(pil_image)

        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    except Exception as error:
        print("Image reading error:", image_path, error)
        return None


def extract_face_features(image_path):
    """
    Detect the largest face and return its SFace features.
    Return None if no face is detected.
    """
    image = read_image(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]

    if width < 20 or height < 20:
        return None

    # Try several image scales because very small faces may be missed.
    scales = [1.0]

    if max(width, height) < 1000:
        scales.append(min(2.0, 1000 / max(width, height)))

    for scale in scales:
        if scale == 1.0:
            test_image = image
        else:
            test_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        test_height, test_width = test_image.shape[:2]
        face_detector.setInputSize((test_width, test_height))

        _, faces = face_detector.detect(test_image)

        if faces is None or len(faces) == 0:
            continue

        # Use the largest detected face.
        face = max(
            faces,
            key=lambda item: float(item[2] * item[3])
        )

        aligned_face = face_recognizer.alignCrop(
            test_image, face
        )

        features = face_recognizer.feature(aligned_face)

        if features is not None:
            return features

    return None


def compare_faces(features1, features2):
    """Higher cosine similarity indicates greater feature similarity."""
    score = face_recognizer.match(
        features1,
        features2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(score)


def get_person_records():
    with get_db() as db:
        return db.execute("""
            SELECT * FROM persons
            ORDER BY id
        """).fetchall()


@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    match_result = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            message = "Please select a photo."

        elif not allowed_file(uploaded_file.filename):
            message = "Please upload a JPG, JPEG, or PNG image."

        else:
            extension = uploaded_file.filename.rsplit(".", 1)[1].lower()

            filename = secure_filename(
                f"{uuid.uuid4().hex}.{extension}"
            )

            upload_path = os.path.join(
                UPLOAD_FOLDER, filename
            )

            try:
                # Validate the uploaded file.
                with Image.open(uploaded_file.stream) as image:
                    image.verify()

                uploaded_file.stream.seek(0)
                uploaded_file.save(upload_path)

                uploaded_features = extract_face_features(
                    upload_path
                )

                if uploaded_features is None:
                    message = (
                        "No face was detected. Please use a clear "
                        "photo with a visible face."
                    )

                else:
                    best_match = None
                    best_score = float("-inf")

                    persons = get_person_records()

                    for person in persons:
                        image_name = person["image"]

                        if not image_name:
                            continue

                        # Keep file access inside the image folder.
                        if os.path.basename(image_name) != image_name:
                            continue

                        sample_path = os.path.join(
                            IMAGE_FOLDER, image_name
                        )

                        if not os.path.isfile(sample_path):
                            continue

                        try:
                            stored_features = extract_face_features(
                                sample_path
                            )

                            if stored_features is None:
                                print(
                                    "No face detected in stored image:",
                                    image_name
                                )
                                continue

                            score = compare_faces(
                                uploaded_features,
                                stored_features
                            )

                            print(
                                "Case:", person["case_id"],
                                "Similarity:", round(score, 3)
                            )

                            if score > best_score:
                                best_score = score
                                best_match = person

                        except Exception as error:
                            print(
                                "Comparison failed for",
                                person["case_id"],
                                error
                            )

                    if (
                        best_match is not None
                        and best_score >= MATCH_THRESHOLD
                    ):
                        match_result = {
                            "case_id": best_match["case_id"],
                            "name": best_match["name"],
                            "age": best_match["age"],
                            "last_seen": best_match["last_seen"],
                            "image": best_match["image"],
                            "score": round(best_score, 3)
                        }

                        message = (
                            "Possible candidate found. "
                            "Please verify independently."
                        )

                    elif best_match is None:
                        message = (
                            "No usable faces were found in the "
                            "stored record photos. Check the images "
                            "inside static/images."
                        )

                    else:
                        message = (
                            "No matching record found above the "
                            "configured threshold."
                        )

            except Exception as error:
                print("Upload processing error:", error)
                message = (
                    "The image could not be processed. "
                    "Please try a valid JPG or PNG image."
                )

            finally:
                if os.path.exists(upload_path):
                    try:
                        os.remove(upload_path)
                    except OSError:
                        pass

    search = request.args.get("search", "").strip()

    with get_db() as db:
        if search:
            persons = db.execute("""
                SELECT * FROM persons
                WHERE case_id LIKE ? OR name LIKE ?
                ORDER BY id
            """, (f"%{search}%", f"%{search}%")).fetchall()
        else:
            persons = db.execute("""
                SELECT * FROM persons
                ORDER BY id
            """).fetchall()

    return render_template(
        "index.html",
        message=message,
        match_result=match_result,
        persons=persons,
        search=search
    )


@app.errorhandler(413)
def file_too_large(error):
    return (
        "Image too large. Please upload a file under 5 MB.",
        413
    )


if __name__ == "__main__":
    setup_database()
    app.run(debug=True)
