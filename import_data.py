import json
import logging
import time
import uuid
from typing import List

import requests
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from transformers import CLIPProcessor, CLIPModel

import settings

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    )


class DataPipeline:
    BATCH_SIZE = 100

    def __init__(self, products_data_file_path: str, vector_db_uri: str):
        self.products_data_file_path = products_data_file_path
        self.products_info = {}
        self.images_info = {}
        self.client = QdrantClient(url=vector_db_uri)
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.logger = logging.getLogger('data_pipeline')

    def encode_images(self, urls: List[str]):
        images = [
            Image.open(requests.get(url, stream=True).raw)
            for url in urls]

        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def create_collection(self):
        if not self.client.collection_exists(settings.CLOTHES_COLLECTION_NAME):
            self.client.create_collection(settings.CLOTHES_COLLECTION_NAME,
                                          {'size': settings.CLOTHES_COLLECTION_EMBEDDING_SIZE,
                                           'distance': models.Distance.COSINE})

    def run(self):
        self.create_collection()
        self.logger.info('Starting to import data to vector db ...')

        batch_gen = self.batch_data_generator()
        for batch_idx, batch in batch_gen:
            self.logger.info(f'Inserting batch_idx: {batch_idx}')

            batch_img_urls = list(batch.keys())
            batch_encoded_img = self.encode_images(batch_img_urls)
            payloads = list(batch.values())

            points = [
                PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                for vector, payload in zip(batch_encoded_img, payloads)
            ]
            self.client.upsert(collection_name=settings.CLOTHES_COLLECTION_NAME, points=points)

            self.logger.info(f'Batch with batch_idx: {batch_idx}, batch_size:{str(len(points))} inserted to vector db.')
            time.sleep(0.1)

    def batch_data_generator(self):
        with open(self.products_data_file_path) as products_json_file:
            items = json.load(products_json_file)
            batch = {}
            batch_idx = 0

            for item in items:
                for image_url in item['images']:
                    batch[image_url] = {key: value for key, value in item.items() if key != 'images'}

                    if len(batch) == self.BATCH_SIZE:
                        yield batch_idx, batch
                        batch_idx += 1
                        batch = {}

            if len(batch) != 0:
                yield batch


def main():
    pipeline = DataPipeline(products_data_file_path=settings.PRODUCTS_DATA_FILE_PATH,
                            vector_db_uri=settings.VECTOR_DB_URI)
    pipeline.run()


if __name__ == '__main__':
    main()
