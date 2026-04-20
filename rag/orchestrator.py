from rag.guard import Guard
from config.config import SIMILARITY_THRESHOLD,MAX_ALLOWED_CHUNKS,LOG_PATH,REFUSAL_MESSAGE
from rag.retriever import Retriever
from rag.generator import Generator
from models.embedding_model import EmbeddingModel
# from config.model_loader import get_llm # For Tinyllama
# from models.llm_model import LLMModel # For Tinyllama
from rewriter.query_rewriter import QueryRewriter
from reranker.reranker import Reranker,select_chunks,remove_duplicate_chunks
import time
from observability.logger import Logger
from observability.anomaly_detector import AnomalyDetector
from models.groq_model import GroqGenerator
from math import exp

#REFUSAL_MESSAGE = "I do not have sufficient information in the knowledge base to answer this question."

# from math import exp
def sigmoid(x):
  x_float = float(x)
  result = 1 / (1 + exp(-x_float))
  #return f"{result:.17f}"
  return result

class RAGOrchestrator :
	# def __init__(self,embedding_model,retriever,generator):
	# 	self.embedding_model = embedding_model
	# 	self.retriever = retriever
	# 	self.generator = generator

	def __init__(self):
		self.embedding_model = None
		self.retriever = None
		self.generator = None
		self.llm_model = None

	def run(self,query):
		self.embedding_model = EmbeddingModel()
		self.retriever = Retriever()
		self.llm_model,tokenizer = get_llm()
		llm = LLMModel()
		self.generator = Generator(llm)
		qrewriter = QueryRewriter(llm)
		rewrite_triggered = False
		rewrite_success = False
    # Embed query
		query_embeddings = self.embedding_model.embed([query])[0]
    
    # Retrieve Chunks
		results = self.retriever.retrieve(query_embeddings)
		final_result = ({"answer":REFUSAL_MESSAGE,"confidence":"low","source":None,"chunk_id":None,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		if(not results or len(results['documents'][0])==0):
			return final_result
    
    # Rerank Chunks
		raw_chunks = []
		reranker = Reranker()
		for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
				raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
		reranked_chunks = 	reranker.rerank(query=query,chunks=raw_chunks)
		reranked_results = {
            'documents': [[chunk['text'] for chunk in reranked_chunks]],
	          'distances': [[1-chunk['similarity'] for chunk in reranked_chunks]],
	          'metadatas': [[chunk['metadata'] for chunk in reranked_chunks]]
          }
    
    # Guard Chunks using confidence
		# guard_output = Guard.filter_results(results)
		guard_output = Guard.filter_results(reranked_results)
		# {'chunks':filtered_chunks,'retrieval_status':retrieval_status,'top_score':highest_similarity,'gap':diff}
		retrieval_status = guard_output['retrieval_status']
		filtered_chunks = guard_output['chunks']

		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			return final_result
		elif(retrieval_status=="GAP_ZONE"):	
				#query rewrite
			rewritten_query = qrewriter.rewrite(query)    
			rewritten_query_embeddings = self.embedding_model.embed([rewritten_query])[0]
			rewritten_results =  self.retriever.retrieve(rewritten_query_embeddings)
			rewrite_triggered =True
			if rewritten_results and len(rewritten_results['documents'][0])>0:
				rw_guard_output = Guard.filter_results(rewritten_results)
				rw_status = rw_guard_output['retrieval_status'] 
				rw_chunks = rw_guard_output['chunks']

				if rw_status == "HIGH":
					filtered_chunks = rw_chunks
					rewrite_success=True
				elif rw_status == "GAP_ZONE":
					filtered_chunks = filtered_chunks + rw_chunks
					filtered_chunks.sort(key=lambda x:x['similarity'],reverse=True)
		# 	answer = self.generator.generate(query,filtered_chunks)
		# else:
		# 	answer = self.generator.generate(query,filtered_chunks)			
		answer = self.generator.generate(query,filtered_chunks)
		top_meta = filtered_chunks[0]["metadata"]
		confidence= filtered_chunks[0]['confidence']
		if(retrieval_status=="GAP_ZONE"):
			confidence= 'LOW'
		# chunk_ids = [c['metadata']['chunk_id'] for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_id":top_meta["chunk_id"],'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		# final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_ids":chunk_ids,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		return final_result

######################################################################    
############ New version of run adding observability##################
######################################################################

	def run_v1(self,query):
		self.embedding_model = EmbeddingModel()
		self.retriever = Retriever()
		self.llm_model,tokenizer = get_llm()
		llm = LLMModel()
		self.generator = Generator(llm)
		qrewriter = QueryRewriter(llm)
		total_start = time.time()
		latency={"total":0.0,"retrieval":0.0,"reranking":0.0,"rewrite":0.0,"generation":0.0}
		rewrite_triggered = False
		rewrite_success = False
		rewritten_query = None
		retrieval_scores = []
		reranker_scores = []
		chunk_ids = []
		# Embed query
		query_embeddings = self.embedding_model.embed([query])[0]
		
		# Retrieve Chunks
		t0_retrieval = time.time()
		results = self.retriever.retrieve(query_embeddings)
		latency['retrieval'] = round(time.time()-t0_retrieval,3)
		final_result = ({"answer":REFUSAL_MESSAGE,"confidence":"low","source":None,"chunk_ids":None,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		if(not results or len(results['documents'][0])==0):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,[],[],[],False,None)
			return final_result
		
		# Rerank Chunks
		raw_chunks = []
		reranker = Reranker()
		for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
				raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
		t0_reranking = time.time()
		retrieval_scores = [chunk['similarity'] for chunk in raw_chunks]
		reranked_chunks = 	reranker.rerank(query=query,chunks=raw_chunks)
		selected_chunks = select_chunks(reranked_chunks,max_chunks=MAX_ALLOWED_CHUNKS)
		reranker_scores = [chunk['reranker_score'] for chunk in selected_chunks]
		reranked_results = {
						'documents': [[chunk['text'] for chunk in selected_chunks]],
						'distances': [[1-chunk['similarity'] for chunk in selected_chunks]],
						'metadatas': [[chunk['metadata'] for chunk in selected_chunks]]
						}    
		# reranked_results = {
		# 				'documents': [[chunk['text'] for chunk in reranked_chunks]],
		# 				'distances': [[1-chunk['similarity'] for chunk in reranked_chunks]],
		# 				'metadatas': [[chunk['metadata'] for chunk in reranked_chunks]]
		# 				}
		# reranker_scores = [chunk['reranker_score'] for chunk in reranked_chunks]
		latency['reranking'] = round(time.time()-t0_reranking,3)

    # Guard Chunks using confidence
		# guard_output = Guard.filter_results(results)
		guard_output = Guard.filter_results(reranked_results)
		# {'chunks':filtered_chunks,'retrieval_status':retrieval_status,'top_score':highest_similarity,'gap':diff}
		retrieval_status = guard_output['retrieval_status']
		filtered_chunks = guard_output['chunks']

		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			return final_result
		elif(retrieval_status=="GAP_ZONE"):	
				#query rewrite
			t0_rewrite = time.time()  
			rewritten_query = qrewriter.rewrite(query)    
			rewritten_query_embeddings = self.embedding_model.embed([rewritten_query])[0]
			rewritten_results =  self.retriever.retrieve(rewritten_query_embeddings)
			rewrite_triggered =True
			latency['rewrite'] = round(time.time()-t0_rewrite,3)
			if rewritten_results and len(rewritten_results['documents'][0])>0:
				rw_guard_output = Guard.filter_results(rewritten_results)
				rw_status = rw_guard_output['retrieval_status'] 
				rw_chunks = rw_guard_output['chunks']

				if rw_status == "HIGH":
					filtered_chunks = rw_chunks
					rewrite_success=True
				elif rw_status == "GAP_ZONE":
					filtered_chunks = filtered_chunks + rw_chunks
					filtered_chunks.sort(key=lambda x:x['similarity'],reverse=True)
		# 	answer = self.generator.generate(query,filtered_chunks)
		# else:
		# 	answer = self.generator.generate(query,filtered_chunks)
		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			return final_result

		t0_generation = time.time()			
		answer = self.generator.generate(query,filtered_chunks)
		latency['generation'] = round(time.time()-t0_generation,3)
		top_meta = filtered_chunks[0]["metadata"]
		confidence= filtered_chunks[0]['confidence']
		if(retrieval_status=="GAP_ZONE"):
			# latency['total'] = round(time.time()-total_start,3)
			# self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			confidence= 'LOW'
		# chunk_ids = [c['metadata']['chunk_id'] for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		chunk_ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_id']}" for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		latency['total'] = round(time.time()-total_start,3)
		final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_id":top_meta["chunk_id"],'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		# final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_ids":chunk_ids,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		self._log(query,final_result,latency,retrieval_scores,reranker_scores,chunk_ids,rewrite_triggered,rewritten_query)

		return final_result


############################################################################
####################Use Groq################################################
############################################################################

	def run_groq(self,query):
		self.embedding_model = EmbeddingModel()
		self.retriever = Retriever()
		# self.llm_model,tokenizer = get_llm()#For Tinyllama
		# llm = LLMModel() #For Tinyllama
		groqq = GroqGenerator()
		self.generator = Generator(groqq)
		# qrewriter = QueryRewriter(llm)#For Tinyllama
		qrewriter = QueryRewriter(groqq)
		total_start = time.time()
		latency={"total":0.0,"retrieval":0.0,"reranking":0.0,"rewrite":0.0,"generation":0.0}
		rewrite_triggered = False
		rewrite_success = False
		rewritten_query = None
		retrieval_scores = []
		reranker_scores = []
		chunk_ids = []
		# Embed query
		query_embeddings = self.embedding_model.embed([query])[0]
		
		# Retrieve Chunks
		t0_retrieval = time.time()
		results = self.retriever.retrieve(query_embeddings)

		latency['retrieval'] = round(time.time()-t0_retrieval,3)
		final_result = ({"answer":REFUSAL_MESSAGE,"guard_scores":[],"retrieval_scores":[(1-distance) for distance in results['distances'][0]],"reranker_scores_before":[],"reranker_scores_after":[],"confidence":"low","source":None,"chunk_ids":None,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		if(not results or len(results['documents'][0])==0):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,[],[],[],False,None)
			return final_result
		
		# Rerank Chunks
		raw_chunks = []
		reranker = Reranker()
		for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
				raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
		t0_reranking = time.time()
		retrieval_scores = [chunk['similarity'] for chunk in raw_chunks]
		# print(f"retrieval_scores->{retrieval_scores}")
		reranked_chunks = 	reranker.rerank(query=query,chunks=raw_chunks)
		reranker_scores_ = [f"(Reranker->{sigmoid(chunk['reranker_score'])},Similarity->{chunk['similarity']})" for chunk in reranked_chunks]
		# print(f"Reranker Score Before->{reranker_scores}")
		selected_chunks = select_chunks(reranked_chunks,max_chunks=MAX_ALLOWED_CHUNKS)
		reranker_scores = [sigmoid(chunk['reranker_score']) for chunk in selected_chunks]
		# reranker_scores = [f"(Reranker->{sigmoid(chunk['reranker_score'])},Similarity->{chunk['similarity']})" for chunk in selected_chunks]
		# print(f"Reranker Score After Select Chunks->{reranker_scores}")
		reranked_results = {
						'documents': [[chunk['text'] for chunk in selected_chunks]],
						'distances': [[1-chunk['similarity'] for chunk in selected_chunks]],
						'metadatas': [[chunk['metadata'] for chunk in selected_chunks]],
            'reranker_score':[[sigmoid(chunk['reranker_score']) for chunk in selected_chunks]]
						}    
		# reranked_results = {
		# 				'documents': [[chunk['text'] for chunk in reranked_chunks]],
		# 				'distances': [[1-chunk['similarity'] for chunk in reranked_chunks]],
		# 				'metadatas': [[chunk['metadata'] for chunk in reranked_chunks]]
		# 				}
		# reranker_scores = [chunk['reranker_score'] for chunk in reranked_chunks]
		latency['reranking'] = round(time.time()-t0_reranking,3)

    # Guard Chunks using confidence
		# guard_output = Guard.filter_results(results)
		guard_output = Guard.filter_results_v1(reranked_results)
		# {'chunks':filtered_chunks,'retrieval_status':retrieval_status,'top_score':highest_similarity,'gap':diff}
		retrieval_status = guard_output['retrieval_status']
		filtered_chunks = guard_output['chunks']
		guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in filtered_chunks]
		# print(f"Guard Scores->{guard_scores}")
		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			# guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in filtered_chunks]		
			return final_result
		elif(retrieval_status=="GAP_ZONE"):	
				#query rewrite
			t0_rewrite = time.time()  
			# rewritten_query = qrewriter.rewrite(query) #For Tinyllma   
			rewritten_query = qrewriter.rewrite_with_groq(query)    
			rewritten_query_embeddings = self.embedding_model.embed([rewritten_query])[0]
			rewritten_results =  self.retriever.retrieve(rewritten_query_embeddings)
			rewrite_triggered =True
			latency['rewrite'] = round(time.time()-t0_rewrite,3)
			if rewritten_results and len(rewritten_results['documents'][0])>0:
				raw_chunks=[]
				for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
					raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
				rw_reranked_chunks = 	reranker.rerank(query=rewritten_query,chunks=raw_chunks)
				rw_selected_chunks = select_chunks(reranked_chunks,max_chunks=MAX_ALLOWED_CHUNKS)
				rw_reranked_results = {
				'documents': [[chunk['text'] for chunk in rw_selected_chunks]],
				'distances': [[1-chunk['similarity'] for chunk in rw_selected_chunks]],
				'metadatas': [[chunk['metadata'] for chunk in rw_selected_chunks]],
        'reranker_score':[[sigmoid(chunk['reranker_score']) for chunk in rw_selected_chunks]]
				}
				rw_guard_output = Guard.filter_results_v1(rw_reranked_results)

				rw_status = rw_guard_output['retrieval_status'] 
				rw_chunks = rw_guard_output['chunks']
				guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in rw_chunks]
				if rw_status == "HIGH":
					filtered_chunks = rw_chunks
					rewrite_success=True
				elif rw_status == "GAP_ZONE":
					filtered_chunks = filtered_chunks + rw_chunks
					# filtered_chunks=remove_duplicate_chunks(filtered_chunks)
					filtered_chunks.sort(key=lambda x:x['reranker_score'],reverse=True)
		# 	answer = self.generator.generate(query,filtered_chunks)
		# else:
		# 	answer = self.generator.generate(query,filtered_chunks)
		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			return final_result

		t0_generation = time.time()		
		filtered_chunks=remove_duplicate_chunks(filtered_chunks)	
		# answer = self.generator.generate(query,filtered_chunks[:MAX_ALLOWED_CHUNKS])

		answer = self.generator.generate_with_groq(query,filtered_chunks[:MAX_ALLOWED_CHUNKS],groqq)
		# answer ="From LLM"
		latency['generation'] = round(time.time()-t0_generation,3)
		top_meta = filtered_chunks[0]["metadata"]
		confidence= filtered_chunks[0]['confidence']
		if(retrieval_status=="GAP_ZONE"):
			# latency['total'] = round(time.time()-total_start,3)
			# self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			confidence= 'LOW'
		# chunk_ids = [c['metadata']['chunk_id'] for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		chunk_ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_id']}" for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		latency['total'] = round(time.time()-total_start,3)
		final_result = ({"answer":answer,"guard_scores":guard_scores,"retrieval_scores":retrieval_scores,"reranker_scores_before":reranker_scores_,"reranker_scores_after":reranker_scores,"confidence":confidence,"source":top_meta["source"],"chunk_id":top_meta["chunk_id"],'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		# final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_ids":chunk_ids,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		self._log(query,final_result,latency,retrieval_scores,reranker_scores,chunk_ids,rewrite_triggered,rewritten_query)

		return final_result

	def _log(self,query,final_result,latency,retrieval_scores,reranker_scores,chunk_ids,rewrite_triggered,rewritten_query):
		logger = Logger.get_instance()
		temp_log = {
					"latency":latency,
					"confidence":final_result.get('confidence','low'),
					"chunk_ids":chunk_ids,
					"rewrite_triggered":rewrite_triggered
					}
		anomalies = AnomalyDetector.detect(temp_log)
		logger.log_request(query=query,answer=final_result.get('answer',''),confidence=final_result.get('confidence','low'),source=final_result.get('source',None),chunk_ids=chunk_ids,retrieval_scores=retrieval_scores,reranker_scores=reranker_scores,rewrite_triggered=rewrite_triggered,rewritten_query=rewritten_query,latency=latency,anomalies=anomalies)

############################################################################
##############Test Retrieval################################################
############################################################################

	def run_test(self,query):
		self.embedding_model = EmbeddingModel()
		self.retriever = Retriever()
		self.llm_model,tokenizer = get_llm()
		llm = LLMModel()
		self.generator = Generator(llm)
		qrewriter = QueryRewriter(llm)
		total_start = time.time()
		latency={"total":0.0,"retrieval":0.0,"reranking":0.0,"rewrite":0.0,"generation":0.0}
		rewrite_triggered = False
		rewrite_success = False
		rewritten_query = None
		retrieval_scores = []
		reranker_scores = []
		chunk_ids = []
		# Embed query
		query_embeddings = self.embedding_model.embed([query])[0]
		
		# Retrieve Chunks
		t0_retrieval = time.time()
		results = self.retriever.retrieve(query_embeddings)

		latency['retrieval'] = round(time.time()-t0_retrieval,3)
		final_result = ({"answer":REFUSAL_MESSAGE,"guard_scores":[],"retrieval_scores":[(1-distance) for distance in results['distances'][0]],"reranker_scores_before":[],"reranker_scores_after":[],"confidence":"low","source":None,"chunk_ids":None,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		if(not results or len(results['documents'][0])==0):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,[],[],[],False,None)
			return final_result
		
		# Rerank Chunks
		raw_chunks = []
		reranker = Reranker()
		for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
				raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
		t0_reranking = time.time()
		retrieval_scores = [chunk['similarity'] for chunk in raw_chunks]
		# print(f"retrieval_scores->{retrieval_scores}")
		reranked_chunks = 	reranker.rerank(query=query,chunks=raw_chunks)
		reranker_scores_ = [f"(Reranker->{sigmoid(chunk['reranker_score'])},Similarity->{chunk['similarity']})" for chunk in reranked_chunks]
		# print(f"Reranker Score Before->{reranker_scores}")
		selected_chunks = select_chunks(reranked_chunks,max_chunks=MAX_ALLOWED_CHUNKS)
		reranker_scores = [sigmoid(chunk['reranker_score']) for chunk in selected_chunks]
		# reranker_scores = [f"(Reranker->{sigmoid(chunk['reranker_score'])},Similarity->{chunk['similarity']})" for chunk in selected_chunks]
		# print(f"Reranker Score After Select Chunks->{reranker_scores}")
		reranked_results = {
						'documents': [[chunk['text'] for chunk in selected_chunks]],
						'distances': [[1-chunk['similarity'] for chunk in selected_chunks]],
						'metadatas': [[chunk['metadata'] for chunk in selected_chunks]],
            'reranker_score':[[sigmoid(chunk['reranker_score']) for chunk in selected_chunks]]
						}    
		# reranked_results = {
		# 				'documents': [[chunk['text'] for chunk in reranked_chunks]],
		# 				'distances': [[1-chunk['similarity'] for chunk in reranked_chunks]],
		# 				'metadatas': [[chunk['metadata'] for chunk in reranked_chunks]]
		# 				}
		# reranker_scores = [chunk['reranker_score'] for chunk in reranked_chunks]
		latency['reranking'] = round(time.time()-t0_reranking,3)

    # Guard Chunks using confidence
		# guard_output = Guard.filter_results(results)
		guard_output = Guard.filter_results_v1(reranked_results)
		# {'chunks':filtered_chunks,'retrieval_status':retrieval_status,'top_score':highest_similarity,'gap':diff}
		retrieval_status = guard_output['retrieval_status']
		filtered_chunks = guard_output['chunks']
		guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in filtered_chunks]
		# print(f"Guard Scores->{guard_scores}")
		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			# guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in filtered_chunks]		
			return final_result
		elif(retrieval_status=="GAP_ZONE"):	
				#query rewrite
			t0_rewrite = time.time()  
			rewritten_query = qrewriter.rewrite(query)    
			rewritten_query_embeddings = self.embedding_model.embed([rewritten_query])[0]
			rewritten_results =  self.retriever.retrieve(rewritten_query_embeddings)
			rewrite_triggered =True
			latency['rewrite'] = round(time.time()-t0_rewrite,3)
			if rewritten_results and len(rewritten_results['documents'][0])>0:
				raw_chunks=[]
				for doc,distance,metadata in zip(results['documents'][0],results['distances'][0],results['metadatas'][0]):
					raw_chunks.append({'text':doc,'similarity':1-distance,'metadata':metadata})
				rw_reranked_chunks = 	reranker.rerank(query=rewritten_query,chunks=raw_chunks)
				rw_selected_chunks = select_chunks(reranked_chunks,max_chunks=MAX_ALLOWED_CHUNKS)
				rw_reranked_results = {
				'documents': [[chunk['text'] for chunk in rw_selected_chunks]],
				'distances': [[1-chunk['similarity'] for chunk in rw_selected_chunks]],
				'metadatas': [[chunk['metadata'] for chunk in rw_selected_chunks]],
        'reranker_score':[[sigmoid(chunk['reranker_score']) for chunk in rw_selected_chunks]]
				}
				rw_guard_output = Guard.filter_results_v1(rw_reranked_results)

				rw_status = rw_guard_output['retrieval_status'] 
				rw_chunks = rw_guard_output['chunks']
				guard_scores = [f"(Guard->{chunk['reranker_score']},Similarity->{chunk['similarity']},Confidence->{chunk['confidence']})" for chunk in rw_chunks]
				if rw_status == "HIGH":
					filtered_chunks = rw_chunks
					rewrite_success=True
				elif rw_status == "GAP_ZONE":
					filtered_chunks = filtered_chunks + rw_chunks
					filtered_chunks.sort(key=lambda x:x['reranker_score'],reverse=True)
		# 	answer = self.generator.generate(query,filtered_chunks)
		# else:
		# 	answer = self.generator.generate(query,filtered_chunks)
		if(retrieval_status=="REFUSE" or (not filtered_chunks)):
			latency['total'] = round(time.time()-total_start,3)
			self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			return final_result

		t0_generation = time.time()			
		answer = self.generator.generate(query,filtered_chunks[:MAX_ALLOWED_CHUNKS])
		# answer ="From LLM"
		latency['generation'] = round(time.time()-t0_generation,3)
		top_meta = filtered_chunks[0]["metadata"]
		confidence= filtered_chunks[0]['confidence']
		if(retrieval_status=="GAP_ZONE"):
			# latency['total'] = round(time.time()-total_start,3)
			# self._log(query,final_result,latency,retrieval_scores,reranker_scores,[],rewrite_triggered,rewritten_query)
			confidence= 'LOW'
		# chunk_ids = [c['metadata']['chunk_id'] for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		chunk_ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_id']}" for c in filtered_chunks[:MAX_ALLOWED_CHUNKS] ]
		latency['total'] = round(time.time()-total_start,3)
		final_result = ({"answer":answer,"guard_scores":guard_scores,"retrieval_scores":retrieval_scores,"reranker_scores_before":reranker_scores_,"reranker_scores_after":reranker_scores,"confidence":confidence,"source":top_meta["source"],"chunk_id":top_meta["chunk_id"],'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		# final_result = ({"answer":answer,"confidence":confidence,"source":top_meta["source"],"chunk_ids":chunk_ids,'rewrite_triggered':rewrite_triggered,'rewrite_success':rewrite_success})
		self._log(query,final_result,latency,retrieval_scores,reranker_scores,chunk_ids,rewrite_triggered,rewritten_query)

		return final_result

	def _log(self,query,final_result,latency,retrieval_scores,reranker_scores,chunk_ids,rewrite_triggered,rewritten_query):
		logger = Logger.get_instance()
		temp_log = {
					"latency":latency,
					"confidence":final_result.get('confidence','low'),
					"chunk_ids":chunk_ids,
					"rewrite_triggered":rewrite_triggered
					}
		anomalies = AnomalyDetector.detect(temp_log)
		logger.log_request(query=query,answer=final_result.get('answer',''),confidence=final_result.get('confidence','low'),source=final_result.get('source',None),chunk_ids=chunk_ids,retrieval_scores=retrieval_scores,reranker_scores=reranker_scores,rewrite_triggered=rewrite_triggered,rewritten_query=rewritten_query,latency=latency,anomalies=anomalies)
