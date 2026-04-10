#Collect top_k chunk results from retriever and filter those chunks on the basis SIMILARITY_THRESHOLD
from config.config import SIMILARITY_THRESHOLD,GUARD_ACTIVE,GAP,SIMILARITY_THRESHOLD_ACCEPTED
class Guard:
    @staticmethod
    def filter_results(results):
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        filtered_chunks = []
        # confidence = "LOW"
        highest_similarity=0.0
        second_highest    =0.0
        scored = []
        diff = 0.0
        # Calculate Similarity and gap
        for doc,distance,metadata in zip(documents,distances,metadatas):
            similarity = 1-distance
            scored.append({'text' : doc,'similarity': similarity,'metadata' : metadata})
            if(highest_similarity<similarity):
              second_highest = highest_similarity
              highest_similarity=similarity
            elif (second_highest<similarity):
              second_highest = similarity              

        diff = highest_similarity-second_highest
        # Chunk Classification and filtering 
        for chunk in scored:
          if (chunk['similarity']>=SIMILARITY_THRESHOLD_ACCEPTED):
            chunk['confidence']="HIGH"
            filtered_chunks.append(chunk)
          elif(chunk['similarity']>=SIMILARITY_THRESHOLD and diff>GAP):
            chunk['confidence']="MEDIUM"
            filtered_chunks.append(chunk)
          # else :
          #   chunk['cofidence']="LOW"
        filtered_chunks.sort(key=lambda x:x['similarity'],reverse =True) #Highest similarity first
        if not filtered_chunks:
          retrieval_status = "REFUSE"
        elif filtered_chunks[0]['confidence'] == "HIGH":
          retrieval_status = "HIGH"
        else:
          retrieval_status = "GAP_ZONE"


        return {'chunks':filtered_chunks,'retrieval_status':retrieval_status,'top_score':highest_similarity,'gap':diff}
			