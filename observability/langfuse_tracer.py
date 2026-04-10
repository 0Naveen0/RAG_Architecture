import os
from langfuse import Langfuse
# from langfuse.otel import Langfuse

class LangfuseTracer:
	_client = None
	
	@classmethod
	def get_client(cls):		
		cls._client = Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),secret_key=os.getenv("LANGFUSE_SECRET_KEY"),host=os.getenv("LANGFUSE_BASE_URL"),)
		return cls._client

	@staticmethod	
	def trace(log:dict):
		client = LangfuseTracer.get_client()
		if client is None:
			print("Langfuse client not initialized. Skipping trace.")
			return
		with trace(
         name="eka_request",
         input={"query": log['query']},
         output={"answer": log["answer"]},
         metadata=log
     ):
		 pass
		# client.trace(name="eka_request",input={"query":log['query']},output = {"answer":log["answer"]},metadata=log)