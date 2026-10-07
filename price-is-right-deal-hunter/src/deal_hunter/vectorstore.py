"""Chroma vector store of priced products, used by the FrontierAgent for RAG and by the UI plot.

Populate it with ``deal-hunter build-vectorstore`` from the HF items dataset produced by
../price-dataset-and-baselines. Each document is the item ``summary``; metadata holds category and price.
"""

from __future__ import annotations

import logging
from itertools import islice
from typing import Iterable, Iterator, List

log = logging.getLogger(__name__)

COLLECTION_NAME = "products"
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]
UNKNOWN_COLOR = "grey"


def open_collection(path: str, name: str = COLLECTION_NAME):
    import chromadb

    return chromadb.PersistentClient(path=path).get_or_create_collection(name)


def reset_collection(path: str, name: str = COLLECTION_NAME):
    import chromadb

    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 - did not exist
        pass
    return client.create_collection(name)


def load_items(dataset_name: str, split: str = "train", token: str | None = None):
    """Rows of the HF items dataset (needs the ``build`` extra)."""
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, token=token)


def _batches(rows: Iterable[dict], size: int) -> Iterator[List[dict]]:
    it = iter(rows)
    while batch := list(islice(it, size)):
        yield batch


def build_vectorstore(rows: Iterable[dict], collection, encoder, batch_size: int = 1000,
                      limit: int | None = None, progress: bool = False) -> int:
    """Embed ``summary`` of each row and add to ``collection``. Returns the number of documents added.

    Ids are ``doc_<n>`` in input order, so re-running over the same data is deterministic.
    Rows without a summary are skipped.
    """
    if limit is not None:
        rows = islice(rows, limit)
    iterator = _batches(rows, batch_size)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="embedding batches")
    added = 0
    for batch in iterator:
        batch = [r for r in batch if r.get("summary")]
        if not batch:
            continue
        documents = [r["summary"] for r in batch]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": r.get("category", ""), "price": float(r["price"])} for r in batch]
        ids = [f"doc_{added + i}" for i in range(len(batch))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        added += len(batch)
    log.info("Vector store now holds %d new documents", added)
    return added


def color_for(category: str) -> str:
    return COLORS[CATEGORIES.index(category)] if category in CATEGORIES else UNKNOWN_COLOR


def get_plot_data(collection, max_datapoints: int = 2000):
    """(documents, 3-D t-SNE coordinates, colours) for the UI scatter plot. Needs the ``ui`` extra."""
    import numpy as np

    result = collection.get(include=["embeddings", "documents", "metadatas"], limit=max_datapoints)
    vectors = np.array(result["embeddings"])
    if len(vectors) < 5:  # t-SNE needs more samples than its perplexity
        return [], np.zeros((0, 3)), []
    documents = result["documents"]
    colors = [color_for(m.get("category", "")) for m in result["metadatas"]]
    from sklearn.manifold import TSNE

    perplexity = min(30.0, len(vectors) - 1.0)  # must be < n_samples
    reduced = TSNE(n_components=3, random_state=42, n_jobs=-1, perplexity=perplexity).fit_transform(vectors)
    return documents, reduced, colors
