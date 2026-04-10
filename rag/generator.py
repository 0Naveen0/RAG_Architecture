from config.config import MAX_CONTEXT_TOKENS,MAX_ALLOWED_CHUNKS,REFUSAL_MESSAGE


class Generator:
	def __init__(self,llm_model):
		self.llm = llm_model

	# def get_answer_from_text(self, text):
	# 	answer =text
	# 	answer_start_tag = "Answer:"
	# 	mylist = text.split("###")
	# 	for item in mylist:
	# 		# print(item)
	# 		if answer_start_tag in item:
	# 			answer = item
	# 	return answer

				
	def get_answer_from_text(self,text):
		answer_tag = "Response:"
		if answer_tag in text:
			answer = text.split(answer_tag)[-1].strip()
			if "###" in answer or "<|" in answer:
				answer = answer.split("###")[0].split("<|")[0].strip()
			return answer if len(answer)>5 else REFUSAL_MESSAGE
  
		return text

	# def get_answer_from_text(self,text):
	# 	answer_tag = "### Answer:"
	# 	if answer_tag in text:
	# 		answer = text.split(answer_tag)[-1].strip()
	# 		if "###" in answer:
	# 			answer = answer.split("###")[0].strip()
	# 		return answer if answer else REFUSAL_MESSAGE
	
	# 	return REFUSAL_MESSAGE

	def build_context(self,filtered_chunks):
		context = ""
		i=0
		for chunk in filtered_chunks:
			if(i<MAX_ALLOWED_CHUNKS):
				context+= f"Context {i+1}:\n"+chunk["text"]+"\nSource: "+f"{chunk["metadata"]["source"]}"+f"\nChunk Id: {chunk["metadata"]["chunk_id"]}"+"\n\n"
				i+=1
		return context[:MAX_CONTEXT_TOKENS*4]

	def generate(self,query,filtered_chunks):
		context = self.build_context(filtered_chunks)
		# PROMPT_TEMPLATE = """
		# 			    ### System :
		# 			    You are a Laravel Expert Assistant.

		# 			    ### Rules:
		# 					- Use only the provided context to answer the question.
		# 					- If the answer is not explicitely contained in the context,
		# 						say I can not find this information in the provided context.
		# 					- Cite the chunk id when answering.
		# 					- Do not repeat instructions.
		# 					- Do not use prior knowledge.
		# 					- Do not repeat Knowledge_Base section.
		# 					- Answer concisely.
		# 					- Only output the final answer.

		# 			    ### Knowledge_Base:
		# 			    {context}

		# 	        ### Question:
		# 	        {query}

		# 	        ### Answer:

		# 		  """
		PROMPT_TEMPLATE = """
				<|System|> You are a laravel Expert Assistant.
				Answer using only the provided context.
				If the answer is not in the context,say: {REFUSAL_MESSAGE}</s>
				<|User|>
				Context:
				{context}
				
				Question: {query}
				</s>
				<|assistance|>
				"""
		prompt  = PROMPT_TEMPLATE.format(context=context,query=query,REFUSAL_MESSAGE=REFUSAL_MESSAGE)
		print(f"Filtered Chunks->{filtered_chunks}")
		print(f"Context->{context}")
		print(f"Prompt->{prompt}")
		print(f"LLM -> {self.llm}")
		raw_output = self.llm.generate(prompt)
		# answer = raw_output[len(prompt):].strip()
		print(f"Raw_output -> {raw_output}")
		answer = self.get_answer_from_text(raw_output)
		# return raw_output
		print(f"Answer -> {answer}")
		return answer