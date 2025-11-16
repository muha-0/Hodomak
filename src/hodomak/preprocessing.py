"""
Preprocessing pipeline for Hodomak.

Stage 1:
  - Use BART zero-shot classification to decide whether an item is
    topwear / bottomwear / full outfit.
  - Use Grounding DINO to crop the relevant clothing region.
  - Save cropped images to a folder and store the path in `cropped_image_path`.

Stage 2:
  - Use LLaVA (via Ollama) to generate a structured, one-line description
    for each cropped image.
  - Store the result in `generated_desc`.
"""

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import torch
import pandas as pd
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    pipeline,
)
import ollama  # requires local Ollama with llava:7b-v1.6


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

# This file lives at: hodomak/src/hodomak/preprocessing.py
# parents[0] = .../src/hodomak
# parents[1] = .../src
# parents[2] = .../hodomak (project root)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped_images"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = RAW_DIR / "hedomak_products.csv"


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def clean_filename(title: str) -> str:
    """Clean a string so it can be safely used as a filename."""
    return re.sub(r'[<>:"/\\|?*]', "", title).replace(" ", "_")


# -------------------------------------------------------------------
# Grounding DINO + BART zero-shot classification
# -------------------------------------------------------------------

# Grounding DINO
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # force CPU for now if GPU isn't available

processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(
    device
)

# Zero-shot classifier (BART)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",
    device=-1,  # CPU
)

contextualized_labels = {
    "topwear": (
        "Clothing worn on the upper body, such as shirts, jackets, "
        "sweaters, hoodies."
    ),
    "bottomwear": (
        "Clothing worn on the lower body, such as jeans, trousers, "
        "shorts, pants, leggings, denim."
    ),
    "full outfit": (
        "A full set of clothing, including both topwear and bottomwear."
    ),
}


def _get_cropping_label(description: str, title: str) -> str:
    """
    Decide whether this item is topwear / bottomwear / full outfit
    based on title and description using BART zero-shot classification.
    """
    labels = list(contextualized_labels.values())

    # 1) Try title first
    result = classifier(title, labels)
    predicted_value = result["labels"][0]
    predicted_category = [
        key for key, value in contextualized_labels.items() if value == predicted_value
    ][0]
    score = result["scores"][0]
    if score > 0.85:
        return predicted_category

    # 2) Fall back to description
    if pd.isna(description) or not isinstance(description, str) or description.strip() == "":
        return "full outfit"

    result = classifier(description, labels)
    predicted_value = result["labels"][0]
    predicted_category = [
        key for key, value in contextualized_labels.items() if value == predicted_value
    ][0]
    score = result["scores"][0]
    if score > 0.6:
        return predicted_category

    return "full outfit"


def _get_text_labels_for_grounding(cropping_label: str) -> Optional[str]:
    """Map the high-level category to a Grounding DINO text prompt."""
    if cropping_label == "topwear":
        return "top wear, t-shirt, hoodie, jacket, sweater, shirt"
    elif cropping_label == "bottomwear":
        return "bottom wear, pants, jeans, trousers, shorts, leggings"
    else:
        return None  # full outfit → use whole image


def _download_image(image_url: str) -> Optional[Image.Image]:
    """Download image from a URL and return a PIL Image, or None on failure."""
    try:
        response = requests.get(image_url, stream=True, timeout=20)
        if response.status_code == 200:
            return Image.open(response.raw).convert("RGB")
        print(f"Failed to download {image_url} (status {response.status_code})")
        return None
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return None


def process_image(
    image_url: str,
    product_title: str,
    description: str,
    output_dir: Path = CROPPED_DIR,
) -> Optional[Path]:
    """
    Process a single product image:
      - Download original image from URL.
      - Decide cropping label (topwear / bottomwear / full outfit).
      - If topwear/bottomwear: run Grounding DINO and crop the box.
      - Else: save full image.
      - Return path to cropped (or original) saved image, or None on failure.
    """
    try:
        image = _download_image(image_url)
        if image is None:
            return None

        original_image = image.copy()
        cropping_label = _get_cropping_label(description, product_title)

        text_labels = _get_text_labels_for_grounding(cropping_label)

        # If full outfit → save whole image
        filename = f"{clean_filename(product_title)}.jpg"
        output_path = output_dir / filename

        if text_labels is None:
            print(f"[Preprocessing] Taking whole fit for {product_title}")
            original_image.save(output_path)
            return output_path

        # Grounding DINO inference
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = grounding_dino(**inputs)

        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]],
        )

        # If detection found
        if len(results[0]["boxes"]) > 0:
            max_index = results[0]["scores"].argmax()
            box = [int(round(x)) for x in results[0]["boxes"][max_index].tolist()]

            # Crop the detected object
            cropped_image = original_image.crop((box[0], box[1], box[2], box[3]))
            cropped_image.save(output_path)
            print(f"[Preprocessing] Cropped image saved for {product_title}")
            return output_path
        else:
            print(f"[Preprocessing] No objects detected for {product_title}, saving full.")
            original_image.save(output_path)
            return output_path

    except Exception as e:
        print(f"[Preprocessing] Error processing {product_title}: {e}")
        return None


def crop_images_and_update_csv(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_dir: Path | str = CROPPED_DIR,
) -> Path:
    """
    Iterate over all products in the CSV, crop their images, and save
    the path to `cropped_image_path`.

    Overwrites the CSV in place.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        image_url = row.get("image_url", "")
        product_title = row.get("title", "")
        description = row.get("description", "")

        if not isinstance(image_url, str) or not image_url.strip():
            continue

        new_image_path = process_image(
            image_url=image_url,
            product_title=product_title,
            description=description,
            output_dir=output_dir,
        )

        if new_image_path:
            df.at[index, "cropped_image_path"] = str(new_image_path)

    df.to_csv(csv_path, index=False)
    print(f"[Preprocessing] Cropped images saved & CSV updated at {csv_path}")
    return csv_path


# -------------------------------------------------------------------
# LLaVA via Ollama – generated_desc
# -------------------------------------------------------------------

def generate_description(image_path: str | Path) -> Optional[str]:
    """
    Read an image from local path and generate a detailed description
    using LLaVA via Ollama.
    """
    try:
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")

        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        prompt = (
            "You are describing an apparel item for a fashion search index. "
            "Return ONE LINE under 30 words, exactly in this order: "
            "type; fit; length; material; primary color; texture/fabric feel; pattern/print; "
            "sleeve type; neckline/collar; rise/waistline; closure type; notable details; "
            "seasonality; style/occasion tags.\n"
            "If you are not sure about any attribute, write 'unknown' instead of leaving it blank. "
            "Do not add extra commentary.\n"
            "Examples:\n"
            "jeans; relaxed; full-length; rigid denim; black; matte; solid; no sleeves; n/a; "
            "high-rise; button fly; clean hem; all-season; casual,minimal\n"
            "hoodie; oversized; hip-length; cotton blend; heather grey; soft fleece; solid; long sleeves; "
            "hood; n/a; pullover; kangaroo pocket; winter; streetwear,casual\n"
            "Now describe the item in the image:"
        )

        full_response = ""
        for response in ollama.generate(
            model="llava:7b-v1.6",
            prompt=prompt,
            images=[image_bytes],
            stream=True,
            options={
                "num_gpu": 0,          # CPU-only if you need it
                "temperature": 0.2,    # concise & consistent
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "num_predict": 80,     # single-line cap
                "stop": ["\n"],        # cut at first newline
                "keep_alive": "10m",
            },
        ):
            full_response += response["response"]

        return full_response.strip()

    except Exception as e:
        print(f"[LLaVA] Error processing {image_path}: {e}")
        return None


def generate_llava_descriptions_for_csv(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    image_col: str = "cropped_image_path",
    desc_col: str = "generated_desc",
    save_every: int = 5,
    reset_existing: bool = False,
) -> Path:
    """
    For each row in the CSV, if `desc_col` is empty or NaN, call LLaVA via Ollama
    on `image_col` and fill in a generated description.

    Parameters
    ----------
    csv_path : Path or str
        Path to the CSV file.
    image_col : str
        Column containing paths to images (e.g., 'cropped_image_path').
    desc_col : str
        Column to store generated descriptions (e.g., 'generated_desc').
    save_every : int
        Save the CSV after this many updates for robustness.
    reset_existing : bool
        If True, overwrite existing values in `desc_col`.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if desc_col not in df.columns:
        df[desc_col] = ""

    updated_count = 0

    for idx, row in df.iterrows():
        existing = str(row.get(desc_col, "") or "").strip()
        if existing and not reset_existing:
            continue

        img_path = row.get(image_col, "")
        if not isinstance(img_path, str) or not img_path.strip():
            continue

        desc = generate_description(img_path)
        if desc is None:
            continue

        print(f"[LLaVA] {desc}\n{'-' * 50}")
        df.at[idx, desc_col] = desc
        updated_count += 1

        if updated_count % save_every == 0:
            df.to_csv(csv_path, index=False)
            print(f"[LLaVA] Intermediate save after {updated_count} updates.")

    df.to_csv(csv_path, index=False)
    print(f"[LLaVA] Completed. Updated descriptions written to {csv_path}")
    return csv_path


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def run_full_preprocessing(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    do_cropping: bool = True,
    do_llava: bool = True,
) -> Path:
    """
    Convenience function to run both preprocessing stages:
    1) cropping with Grounding DINO
    2) description generation with LLaVA
    """
    csv_path = Path(csv_path)

    if do_cropping:
        crop_images_and_update_csv(csv_path=csv_path, output_dir=CROPPED_DIR)

    if do_llava:
        generate_llava_descriptions_for_csv(csv_path=csv_path)

    return csv_path


if __name__ == "__main__":
    run_full_preprocessing()
