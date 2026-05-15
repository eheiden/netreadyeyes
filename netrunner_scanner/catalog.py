import difflib
import re

import numpy as np
import collector_vision as cvg

from .config import TOP_K


def normalize_card_id(card_id):
    text = str(card_id).lower()
    text = re.sub(r"_alt_\d+$", "", text)
    text = re.sub(r"_\d{5}$", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CardCatalog:
    def __init__(self, catalog_path):
        print("\nLoading embedder...")
        base_catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
        self.embedder = base_catalog.embedder

        print("Loading Net Ready Eyes catalog...")
        data = np.load(catalog_path, allow_pickle=True)

        self.ids = data["ids"]
        self.embeddings = data["embeddings"]

        self.catalog_embs = self.embeddings / np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

    def search_image(self, pil_image):
        query_emb = self.embedder.embed(pil_image)
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores = self.catalog_embs @ query_emb
        best_indexes = np.argsort(scores)[::-1][:TOP_K]

        return [
            {
                "id": str(self.ids[idx]),
                "score": float(scores[idx]),
            }
            for idx in best_indexes
        ]

    def search_text(self, query, limit=5):
        query = str(query or "").strip()

        if not query:
            return []

        normalized_query = normalize_card_id(query)

        candidates = []
        seen = set()

        for card_id in self.ids:
            card_id = str(card_id)
            if card_id in seen:
                continue

            seen.add(card_id)
            display_name = normalize_card_id(card_id)

            if normalized_query in display_name:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, normalized_query, display_name).ratio()

            candidates.append({
                "id": card_id,
                "score": float(score),
                "rotation": "text",
                "display_name": display_name,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]
