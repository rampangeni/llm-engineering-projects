"""Load one Amazon-Reviews-2023 category in parallel and parse it into Items."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from tqdm import tqdm

from price_dataset.parser import parse

CHUNK_SIZE = 1000
WORKERS = max((os.cpu_count() or 2) - 1, 1)
SOURCE_DATASET = "McAuley-Lab/Amazon-Reviews-2023"


class ItemLoader:
    def __init__(self, category: str):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint):
        """Return an Item, or None if this datapoint shouldn't be included."""
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers: int):
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers: int = WORKERS):
        """Download + parse this category. Heavy: Electronics alone is several GB."""
        from datasets import load_dataset

        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        # The dataset ships a loading script; this needs `datasets<4` (or pass trust_remote_code).
        self.dataset = load_dataset(
            SOURCE_DATASET, f"raw_meta_{self.category}", split="full", trust_remote_code=True
        )
        results = self.load_in_parallel(workers)
        mins = (datetime.now() - start).total_seconds() / 60
        print(f"Completed {self.category} with {len(results):,} datapoints in {mins:.1f} mins", flush=True)
        return results
