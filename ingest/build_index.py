"""Encode normalized records and upsert them into Qdrant."""
from __future__ import annotations
import json, os, uuid, sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, PayloadSchemaType
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.text import embedding_text
load_dotenv()
DATA = Path(__file__).resolve().parent.parent / "data" / "raw_items.jsonl"
COLLECTION = os.getenv("QDRANT_COLLECTION", "titles")

def main():
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
        for field, schema in [("title", PayloadSchemaType.KEYWORD), ("media_type", PayloadSchemaType.KEYWORD), ("vote_average", PayloadSchemaType.FLOAT), ("vote_count", PayloadSchemaType.INTEGER)]:
            client.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    records = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    for start in range(0, len(records), 64):
        batch = records[start:start + 64]
        vectors = model.encode([embedding_text(r["title"], r["overview"], r["genres"]) for r in batch], batch_size=64, normalize_embeddings=True).tolist()
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["id"])), vector=v, payload=r) for r, v in zip(batch, vectors)]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"Upserted {min(start + len(batch), len(records))}/{len(records)}", flush=True)

if __name__ == "__main__": main()
