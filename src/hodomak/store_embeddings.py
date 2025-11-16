"""
Store embeddings for Hodomak products into Qdrant.

Pipeline:
  - Load product metadata from hedohmak_products.csv
  - Compute:
      * E5 text embeddings for raw + generated descriptions
      * SigLIP image embeddings for cropped local image + original URL
  - Upsert into a Qdrant collection with 4 named vector fields:
      * text_raw_desc
      * text_gen_desc
      * image_cropped
      * image_original
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import requests
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageOps
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import SiglipProcessor, SiglipModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


# -------------------------------------------------------------------
# Paths & env
# -------------------------------------------------------------------

# store_embeddings.py is at: hodomak/src/hodomak/store_embeddings.py
# parents[2] -> project root (hodomak/)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = RAW_DIR / "hedomak_products.csv"
COLLECTION_NAME = "products"

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")


# -------------------------------------------------------------------
# E5 text model
# -------------------------------------------------------------------

e5_model = SentenceTransformer("intfloat/e5-large-v2")


def e5_text_embeddings(text: str) -> List[float]:
    """
    Generate E5 embedding for a single text string.

    Uses 'query:' prefix as recommended for E5 queries.
    Returns a normalized vector (list of floats).
    """
    text = str(text).strip()
    if not text:
        return [0.0] * e5_model.get_sentence_embedding_dimension()

    text = f"query: {text}"
    emb = e5_model.encode(text, normalize_embeddings=True)
    return emb.tolist()


# -------------------------------------------------------------------
# SigLIP image model
# -------------------------------------------------------------------

def pad_to_square(img: Image.Image, fill_color=(255, 255, 255)) -> Image.Image:
    """Pad an image to a square with a given background color."""
    w, h = img.size
    size = max(w, h)
    return ImageOps.pad(
        img,
        (size, size),
        color=fill_color,
        centering=(0.5, 0.5),
    )


_siglip_name = "google/siglip-so400m-patch14-384"
_siglip_device = "cpu"  # change to "cuda" if you want GPU and it's available

_siglip_processor = SiglipProcessor.from_pretrained(_siglip_name)
_siglip_model = SiglipModel.from_pretrained(_siglip_name).to(_siglip_device).eval()

# Infer embedding dimension with a dummy forward pass
with torch.no_grad():
    dummy = Image.new("RGB", (384, 384), color="white")
    dummy_inputs = _siglip_processor(
        images=dummy, text="dummy", return_tensors="pt"
    ).to(_siglip_device)
    dummy_output = _siglip_model(**dummy_inputs)
    _siglip_dim = dummy_output.image_embeds.shape[-1]


def siglip_image_embedding(image_path: str) -> List[float]:
    """
    Compute a SigLIP image embedding from a local image path.
    Returns an L2-normalized vector (list[float]).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = pad_to_square(img)
        inputs = _siglip_processor(images=img, return_tensors="pt").to(_siglip_device)

        with torch.no_grad():
            vec = _siglip_model.get_image_features(**inputs)
            vec = F.normalize(vec, p=2, dim=-1)

        return vec.squeeze(0).cpu().tolist()
    except Exception as e:
        print(f"[SigLIP local] Error for {image_path}: {e}")
        return [0.0] * _siglip_dim


def siglip_image_from_url(url: str) -> List[float]:
    """
    Compute a SigLIP image embedding from an image URL.
    Returns an L2-normalized vector (list[float]).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = pad_to_square(img)
        inputs = _siglip_processor(images=img, return_tensors="pt").to(_siglip_device)

        with torch.no_grad():
            vec = _siglip_model.get_image_features(**inputs)
            vec = F.normalize(vec, p=2, dim=-1)

        return vec.squeeze(0).cpu().tolist()
    except Exception as e:
        print(f"[SigLIP URL] Error for {url}: {e}")
        return [0.0] * _siglip_dim


# -------------------------------------------------------------------
# Qdrant
# -------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client using env variables."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL not set in environment.")
    if not api_key:
        raise ValueError("QDRANT_API_KEY not set in environment.")

    return QdrantClient(url=url, api_key=api_key)


def recreate_products_collection(qdrant: QdrantClient) -> None:
    """
    Recreate the 'products' collection with the correct vector configuration.

    WARNING: This will drop existing data in the collection.
    """
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image_cropped": VectorParams(size=_siglip_dim, distance=Distance.COSINE),
            "image_original": VectorParams(size=_siglip_dim, distance=Distance.COSINE),
            "text_gen_desc": VectorParams(
                size=e5_model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
            "text_raw_desc": VectorParams(
                size=e5_model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        },
    )
    print(f"[Qdrant] Recreated collection '{COLLECTION_NAME}'.")


# -------------------------------------------------------------------
# Building points from CSV
# -------------------------------------------------------------------

def build_points_from_csv(csv_path: str | Path = DEFAULT_CSV_PATH) -> list[PointStruct]:
    """
    Load the product CSV and construct Qdrant points with all four embeddings.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    points: list[PointStruct] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building points"):
        point_id = str(uuid.uuid4())

        brand = row.get("brand", "")
        title = row.get("title", "")
        product_url = row.get("product_url", "")
        regular_price = row.get("regular_price", "")
        sale_price = row.get("sale_price", "")
        raw_desc = row.get("description", "")
        sizes = row.get("sizes", "")
        available = row.get("available", "")
        gen_desc = row.get("generated_desc", "")
        img_local = row.get("cropped_image_path", "")
        img_url = row.get("image_url", "")

        # Embeddings
        vec_text_raw = e5_text_embeddings(raw_desc)
        vec_text_gen = e5_text_embeddings(gen_desc)
        vec_image_cropped = siglip_image_embedding(img_local) if img_local else [0.0] * _siglip_dim
        vec_image_original = siglip_image_from_url(img_url) if img_url else [0.0] * _siglip_dim

        # Sanitize payload (replace NaNs with None)
        payload = {
            "brand": brand,
            "title": title,
            "product_url": product_url,
            "regular_price": regular_price,
            "sale_price": sale_price,
            "description": raw_desc,
            "sizes": sizes,
            "available": available,
            "image_url": img_url,
        }
        payload = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in payload.items()}

        point = PointStruct(
            id=point_id,
            vector={
                "text_raw_desc": vec_text_raw,
                "text_gen_desc": vec_text_gen,
                "image_cropped": vec_image_cropped,
                "image_original": vec_image_original,
            },
            payload=payload,
        )

        points.append(point)

    print(f"[Embeddings] Built {len(points)} points.")
    return points


def upsert_points(
    qdrant: QdrantClient,
    points: list[PointStruct],
    batch_size: int = 64,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Upsert points into Qdrant in batches.
    """
    total = len(points)
    print(f"[Qdrant] Uploading {total} points in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        try:
            qdrant.upsert(
                collection_name=collection_name,
                points=batch,
            )
            print(f"[Qdrant] Upserted batch {i // batch_size + 1}")
        except Exception as e:
            print(f"[Qdrant] Error upserting batch {i // batch_size + 1}: {e}")


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def run_store_embeddings(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    recreate_collection: bool = True,
) -> None:
    """
    High-level function to:
      1) Connect to Qdrant
      2) (Optionally) recreate the 'products' collection
      3) Build points from CSV
      4) Upsert points in batches
    """
    csv_path = Path(csv_path)

    qdrant = get_qdrant_client()
    if recreate_collection:
        recreate_products_collection(qdrant)

    points = build_points_from_csv(csv_path=csv_path)
    upsert_points(qdrant, points)


if __name__ == "__main__":
    run_store_embeddings()
