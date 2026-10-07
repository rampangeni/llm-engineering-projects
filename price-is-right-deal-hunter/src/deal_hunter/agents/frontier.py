"""FrontierAgent: RAG. Retrieve 5 similar priced products from Chroma, ask a frontier LLM to price."""

from __future__ import annotations

import re
from typing import Dict, List

from ..config import Settings
from .base import Agent

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection, settings: Settings | None = None, client=None, encoder=None):
        """
        :param collection: Chroma collection of products (see ``deal-hunter build-vectorstore``)
        :param client: OpenAI-compatible client; :param encoder: object with ``encode(list[str])``
        """
        self.settings = settings or Settings()
        self.log("Initializing Frontier Agent")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.MODEL = self.settings.frontier_model
        self.collection = collection
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(ENCODER_NAME)
        self.model = encoder
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    @staticmethod
    def get_price(s: str) -> float:
        s = (s or "").replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including {len(documents)} similar products")
        kwargs = dict(model=self.MODEL, messages=self.messages_for(description, documents, prices), seed=42)
        if self.settings.frontier_reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.frontier_reasoning_effort
        response = self.client.chat.completions.create(**kwargs)
        result = self.get_price(response.choices[0].message.content)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
