# Take prompt tokenize it send to llm get generated text and return
# import torch
# from transformers import AutoTokenizer,AutoModelForCausalLM
# from config.config import (LLM_MODEL_NAME,MAX_NEW_TOKENS,TEMPERATURE,DO_SAMPLE,REPETITION_PENALTY)

# class LLMModel:
# 	def __init__(self):
# 		self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# 		self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME,dtype=torch.float32)
# 		self.model.to("cpu")
# 		self.model.eval()

# 	def generate(self,prompt):	
# 		inputs = self.tokenizer(prompt,return_tensors="pt",truncation=True)
# 		with torch.no_grad():
# 			outputs = self.model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,temperature=TEMPERATURE,do_sample=DO_SAMPLE,repetition_penalty=REPETITION_PENALTY)
# 		generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
# 		# generated_tokens = outputs[0]
# 		generated_text = self.tokenizer.decode(generated_tokens,skip_special_tokens=True)	
# 		return generated_text


# Take prompt tokenize it send to llm get generated text and return
# Singleton implemented
import torch
from config.model_loader import get_llm
from config.config import (MAX_NEW_TOKENS,TEMPERATURE,DO_SAMPLE,REPETITION_PENALTY,MAX_TIME)

class LLMModel:
	def __init__(self):
		model,tokenizer = get_llm()
		self.tokenizer = tokenizer
		self.model = model
		self.model.to("cpu")
		self.model.eval()

	def generate(self,prompt):	
		inputs = self.tokenizer(prompt,return_tensors="pt",truncation=True)
		with torch.no_grad():
			outputs = self.model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,temperature=TEMPERATURE,do_sample=DO_SAMPLE,repetition_penalty=REPETITION_PENALTY,max_time=MAX_TIME)
		generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
		# generated_tokens = outputs[0]
		generated_text = self.tokenizer.decode(generated_tokens,skip_special_tokens=True)	
		return generated_text