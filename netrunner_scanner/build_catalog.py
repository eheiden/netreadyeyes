from pathlib import Path

import numpy as np
from PIL import Image
import collector_vision as cvg


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def load_embedder():
    """
    Load CollectorVision's embedder.

    We currently borrow the embedder from CollectorVision's Scryfall catalog.
    The resulting embeddings are then saved against our own Netrunner IDs.
    """

    print("Loading CollectorVision embedder...")
    base_catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
    return base_catalog.embedder


def iter_image_files(image_dir):
    image_dir = Path(image_dir)

    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_catalog(image_dir="cards", output_path="netrunner-catalog.npz"):
    """
    Build a local Netrunner catalog.

    Each image filename stem becomes the card ID.

    Example:
        cards/buzzsaw_30005.jpg
        -> id: buzzsaw_30005

    This keeps the scanner compatible with the existing OBS bridge:
        http://localhost/cardmatch/buzzsaw_30005
        -> http://localhost/cards/buzzsaw_30005.jpg
    """

    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_files = list(iter_image_files(image_dir))

    if not image_files:
        raise RuntimeError(f"No supported images found in: {image_dir}")

    embedder = load_embedder()

    ids = []
    embeddings = []

    print(f"Found {len(image_files)} image(s).")
    print(f"Building catalog from: {image_dir}")
    print(f"Output file: {output_path}")

    for index, image_path in enumerate(image_files, start=1):
        card_id = image_path.stem

        print(f"[{index}/{len(image_files)}] Embedding {image_path.name}")

        try:
            image = Image.open(image_path).convert("RGB")
            embedding = embedder.embed(image)

            ids.append(card_id)
            embeddings.append(embedding)

        except Exception as e:
            print(f"FAILED: {image_path}")
            print(e)

    if not embeddings:
        raise RuntimeError("No embeddings were created.")

    ids_array = np.array(ids)
    embeddings_array = np.array(embeddings)

    np.savez_compressed(
        output_path,
        ids=ids_array,
        embeddings=embeddings_array,
    )

    print()
    print(f"Saved catalog: {output_path}")
    print(f"Cards embedded: {len(ids_array)}")
