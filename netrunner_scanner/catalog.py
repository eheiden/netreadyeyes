import numpy as np
import collector_vision as cvg

from .config import TOP_K


class CardCatalog:
    def __init__(self, catalog_path):
        print("\nLoading embedder...")
        base_catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
        self.embedder = base_catalog.embedder

        print("Loading Netrunner catalog...")
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
