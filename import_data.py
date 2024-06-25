import json
from typing import List

import numpy as np
import torch
from PIL import Image
from datasets import Dataset, Image
from qdrant_client import QdrantClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


class DataPipeline:
    BATCH_SIZE = 50
    IMAGES_DATA_FILE_PATH = './images_data.json'
    INDEXED_PRODUCTS_DATA_FILE_PATH = './indexed_products_data.json'
    CLOTHES_COLLECTION_NAME = 'clothes-images'

    def __init__(self, products_data_file_path: str, vector_db_uri: str):
        self.products_data_file_path = products_data_file_path
        self.products_info = {}
        self.images_info = {}
        self.client = QdrantClient(url=vector_db_uri)
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def index_data(self):
        self.products_info = {}
        self.images_info = {}

        with open(self.products_data_file_path) as products_json_file, \
                open(self.IMAGES_DATA_FILE_PATH, 'w+') as images_json_file, \
                open(self.INDEXED_PRODUCTS_DATA_FILE_PATH, 'w+') as indexed_products_json_file:

            items = json.load(products_json_file)
            self.products_info = {item["id"]: item for item in items}

            for item in items:
                for image_url in item['images']:
                    self.images_info[image_url] = item['id']

            json.dump(self.products_info, indexed_products_json_file, indent=4)
            json.dump(self.images_info, images_json_file, indent=4)

    def import_data(self):
        with open(self.IMAGES_DATA_FILE_PATH) as images_json_file, \
                open(self.INDEXED_PRODUCTS_DATA_FILE_PATH) as indexed_products_json_file:
            self.products_info = json.load(indexed_products_json_file)
            self.images_info = json.load(images_json_file)

    def encode_images(self, images: List[str]):
        def transform_fn(el):
            return self.preprocess(images=[Image().decode_example(_) for _ in el['image']], return_tensors='pt')

        dataset = Dataset.from_dict({'image': images})
        dataset = dataset.cast_column('image', Image(decode=False)) if isinstance(images[0], str) else dataset
        dataset.set_format('torch')
        dataset.set_transform(transform_fn)

        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE)
        image_embeddings = []
        pbar = tqdm(total=len(images) // self.BATCH_SIZE, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                image_embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()

        return np.stack(image_embeddings)

    def create_collection(self):
        collections = self.client.get_collections()
        collection_exists = any(collection['name'] == self.CLOTHES_COLLECTION_NAME for collection in collections)

        if not collection_exists:
            self.client.create_collection(self.CLOTHES_COLLECTION_NAME)

    def run(self):
        self.index_data()

        sample_images = list(self.images_info.keys())[:2]
        encodings = self.encode_images(sample_images)

        print(len(encodings), type(encodings[0]), encodings.shape)


def main():
    pipeline = DataPipeline(products_data_file_path='./products.json',
                            vector_db_uri='http://localhost:6333')
    pipeline.run()


if __name__ == '__main__':
    main()
