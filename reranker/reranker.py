from sentence_transformers import CrossEncoder # cross-encoder/ms-marco-MiniLM-L6-v2
from config.config import RERANKER_MODEL_NAME,RERANKER_TOP_K #3	#top_k = 10

class Reranker:
	def __init__(self):
		self.model = None
	
	def get_model(self):
		if self.model is None:
			self.model = CrossEncoder(RERANKER_MODEL_NAME)
		return self.model
		
	# @staticmethod	
	def rerank(self,query:str,chunks:list,top_n:int =RERANKER_TOP_K )->list:
		if not chunks:
			return chunks		
		model = self.get_model()
		pairs = [(query,chunk['text']) for chunk in chunks]
		scores = model.predict(pairs)
		print(scores)
		i=0
		for score,chunk in zip(scores,chunks):
			print(f"Score:{score}\t distance:{chunk['similarity']}Chunk:{chunk}")
			chunk['reranker_score'] = float(score)	
			i+=1		 
		reranked_chunks = sorted(chunks,key=lambda x:x['reranker_score'],reverse = True)		
		return reranked_chunks[:top_n]

    #  # Pair scores with original chunks and sort
    #     scored_chunks = []
    #     for i, chunk in enumerate(chunks):
    #         chunk_copy = chunk.copy()  # Avoid modifying original dict
    #         chunk_copy['reranker_score'] = float(scores[i]) # Ensure score is a float
    #         scored_chunks.append(chunk_copy)