import logging

import torch
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

import settings


class Retriever:
    def __init__(self, vector_db_uri: str):
        self.client = QdrantClient(url=vector_db_uri)
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.logger = logging.getLogger('data_pipeline')

    def encode_text(self, text_query):
        inputs = self.processor(text=text_query, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

    def search_by_text(self, text_query, top_k=5):
        text_vector = self.encode_text(text_query)[0]

        search_result = self.client.search(
            collection_name=settings.CLOTHES_COLLECTION_NAME,
            query_vector=text_vector,
            limit=top_k
        )

        return search_result
