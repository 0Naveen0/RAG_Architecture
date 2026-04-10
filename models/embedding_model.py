
# from sentence_transformers import SentenceTransformer
# from config.config import EMBEDDING_MODEL_NAME

# class EmbeddingModel:
# 	def __init__(self):
# 		self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
		
# 	def embed(self,texts):
# 		return self.model.encode(texts,convert_to_tensor = False,normalize_embeddings=True)



# For Lazy Loady and using singleton Model load

from config.model_loader import get_embedder

class EmbeddingModel:
	def __init__(self):
		self.model = get_embedder()

	def embed(self,texts):
		return self.model.encode(texts,convert_to_tensor = False,normalize_embeddings=True)



