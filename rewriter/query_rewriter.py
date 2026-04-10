# import torch

class QueryRewriter:
    def __init__(self,llm_model) -> None:
        self.llm = llm_model
        self.REWRITE_TEMPLATE = """
      ### System:
      You are a search query optimizer.

      ### Rules:
      - Rephrase the question using formal, document-style language.
      - Use technical vocabulary that would appear in documentation.
      - Do not add new information or assumptions.
      - Do not answer the question.
      - Output only the rephrased question. Nothing else.
      
      ### Original Question:
      {query}

      ### Rephrased Question:   
    
    """

    def get_answer_from_text(self, text,start_tag,splitter):
        answer =text
        answer_start_tag = start_tag
        mylist = text.split(splitter)
        for item in mylist:
            # print(item)
            if answer_start_tag in item:
                answer = item
        return answer
  
    # @staticmethod
    def rewrite(self, query_text: str):
        prompt = self.REWRITE_TEMPLATE.format(query=query_text)
        result = self.llm.generate(prompt)
        answer = self.get_answer_from_text(result,"Rephrased Question:","###")
        return answer


        