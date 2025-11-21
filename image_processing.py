import cv2
import numpy as np
from pathlib import Path
import sqlite3

DB_PATH = Path("patient_health.db")


# ----------------- Basic X-ray / Medical Image Processing -----------------
def load_image(path):
    """
    Load an image in grayscale
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def enhance_contrast(img):
    """
    Enhance image contrast using CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def detect_edges(img):
    """
    Apply Canny edge detection
    """
    return cv2.Canny(img, 50, 150)


def blur_image(img, ksize=(5, 5)):
    """
    Apply Gaussian blur
    """
    return cv2.GaussianBlur(img, ksize, 0)


def threshold_image(img, thresh_val=127):
    """
    Apply binary thresholding
    """
    _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh


def store_image_metadata(patient_id, path, modality, processed_steps):
    """
    Store image metadata into the database
    """
    # Ensure table exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        path TEXT,
        modality TEXT,
        metadata TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """
    metadata = {'processed_steps': processed_steps, 'size': load_image(path).shape}
    conn = sqlite3.connect(DB_PATH)
    conn.execute(create_table_sql)
    conn.execute(
        "INSERT INTO images (patient_id, path, modality, metadata) VALUES (?,?,?,?)",
        (patient_id, path, modality, str(metadata))
    )
    conn.commit()
    conn.close()


# ----------------- Script / Demo -----------------
if __name__ == '__main__':
    sample_path = Path('sample_data/example_xray.png')
    if sample_path.exists():
        img = load_image(sample_path)
        print("Original image shape:", img.shape)

        enhanced = enhance_contrast(img)
        edges = detect_edges(enhanced)
        blurred = blur_image(enhanced)
        thresholded = threshold_image(enhanced)

        print("Processing done:")
        print("Edges shape:", edges.shape)
        print("Blurred shape:", blurred.shape)
        print("Thresholded shape:", thresholded.shape)
    else:
        print('No sample image found; put an image at', sample_path)
