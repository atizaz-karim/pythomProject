from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path

# Load image as grayscale
def load_image(path):
    """
    Load an image from the given path and convert it to grayscale.
    """
    img = Image.open(path).convert("L")  # 'L' mode is grayscale
    return img

# Enhance contrast
def enhance_contrast(img):
    """
    Enhance contrast using PIL ImageEnhance.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(2.0)  # increase contrast by factor of 2

# Edge detection
def detect_edges(img):
    """
    Apply simple edge detection using PIL's FIND_EDGES filter.
    """
    return img.filter(ImageFilter.FIND_EDGES)

if __name__ == "__main__":
    p = Path("sample_data/example_xray.png")
    if p.exists():
        img = load_image(p)
        enhanced = enhance_contrast(img)
        edges = detect_edges(enhanced)
        print("Image processed successfully")
        # Optionally show the images locally
        img.show(title="Original")
        enhanced.show(title="Enhanced")
        edges.show(title="Edges")
    else:
        print("No sample image found at", p)