import os

PRODUCTS_DATA_FILE_PATH = os.getenv("PRODUCTS_DATA_FILE_PATH", './products.json')
CLOTHES_COLLECTION_NAME = os.getenv("CLOTHES_COLLECTION_NAME", 'clothes-images')
CLOTHES_COLLECTION_EMBEDDING_SIZE = int(os.getenv("CLOTHES_COLLECTION_EMBEDDING_SIZE", '512'))
VECTOR_DB_URI = os.getenv("VECTOR_DB_URI", 'http://localhost:6333')
RECOMMENDER_TOP_K = int(os.getenv("RECOMMENDER_TOP_K", '4'))
