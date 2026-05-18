import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models/phobert_model")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "models/phobert_tokenizer")
LABEL_PATH = os.path.join(ROOT_DIR, "models/label_encoder.joblib")
INDEX_PATH = os.path.join(ROOT_DIR, "data/faiss.index")
DATA_PATH = os.path.join(ROOT_DIR, "data/data.pkl")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = 3600  # 1 giờ cho cache embedding

OLLAMA_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "keepitreal/vietnamese-sbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"