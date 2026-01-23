import json

from lib.semantic_search import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer


def verify_image_embedding(image_path):
    very = MultimodalSearch()
    embedding = very.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def _mov_title_doc() -> list[str]:
    data = []
    with open("data/movies.json", "r") as f:
        documents = json.load(f)["movies"]
        for doc in documents:
            m = f"{doc['title']}: {doc['description']}"
            data.append(m)

    return data


def image_search_command(image_path):
    search = MultimodalSearch()
    return search.search_with_image(image_path)


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.texts = _mov_title_doc()
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path):
        image = Image.open(img_path)
        embeddings = self.model.encode(image)
        return embeddings

    def search_with_image(self, image_path):
        result = []
        img_embedding = self.embed_image(image_path)

        for i, (embedding, text) in enumerate(zip(self.text_embeddings, self.texts)):
            data = text.split(":")
            title = data[0].strip()
            desc = data[1].strip()
            cos_sim = cosine_similarity(embedding, img_embedding)
            d = {
                "title": title,
                "description": desc,
                "similarity_score": cos_sim,
            }
            result.append(d)

        result.sort(key=lambda item: item["similarity_score"], reverse=True)
        return result[:5]
