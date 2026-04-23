from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
import importlib
import os
import psutil
# from rag.orchestrator import RAGOrchestrator
from rag.retriever import Retriever
from rag.generator import Generator
from models.groq_model import GroqGenerator
from models.hf_embedding import HFEmbeddingModel
from rag.pipeline import Pipeline
from utils.rate_limiter import RateLimiter
from config.validate_query import validate_query
from config.config import LIMITER_MAX_REQUESTS,LIMITER_WINDOW_SECOND

print("STARTING APP...")
print("PORT:", os.environ.get("PORT"))


app= Flask(__name__)
IS_RENDER = os.environ.get('RENDER') == 'true'
CORS(app)

def log_chroma_files():
  total =0
  for root,dirs,files in os.walk("/tmp/chroma_db"):
    for f in files:
      fp = os.path.join(root,f)
      size_mb = os.path.getsize(fp)/1024/1024
      total+=size_mb
      print(f"[Chromadb]{fp}:{size_mb:.3f}MB")
  print(f"[Chromadb]Total Size:{total:.3f}MB")
# orchestrator = RAGOrchestrator()
retriever = Retriever(create_if_missing=False)
log_chroma_files()
chroma_count =0
try:
  chroma_count = retriever.collection.count()
  print(f"Chroma initialized->{retriever.collection.count()}")
except Exception as e:
  print("Chroma init failed",e)
groqq = GroqGenerator()
gene = Generator(groqq)
hf_embedding_model = HFEmbeddingModel()
orchestrator = Pipeline(retriever=retriever,generator=gene,embedding_model=hf_embedding_model,groqq=groqq)
limiter = RateLimiter(max_requests=LIMITER_MAX_REQUESTS,window_seconds=LIMITER_WINDOW_SECOND)

@app.route("/",methods=["GET"])
def index():
  return render_template('index.html')

@app.route("/ask",methods=["POST"])
def ask():
    client_ip = request.remote_addr
    if not limiter.is_allowed(client_ip):
      return jsonify({"answer":"Too many requests.Please wait before trying again","cofidence":"low","source":None,"chunk_id":None}),429
    data = request.json
    query= data.get("query")
    is_valid,result = validate_query(query)
    if not is_valid:
        return jsonify({"error":result}),400
    query =result
   
    # query_embeddings = embeddingModel.embed(query)
    
    # orchestrator = RAGOrchestrator()
    # result = orchestrator.run_v1(query)
    # result = orchestrator.run_groq(query)
    result = orchestrator.run_production(query)
    return result

@app.route("/health",methods=["GET"])
def health():
  process = psutil.Process(os.getpid())
  mem_mb = process.memory_info().rss/1024/1024
  return jsonify({"message":"Server Running","Chroma":f"Init({chroma_count})","port":os.environ.get("PORT"),"host_render":IS_RENDER,"memory_mb":mem_mb})


if __name__ == "__main__":
    if IS_RENDER :
       app.run(debug=True)# In production
    else:
       app.run(host="0.0.0.0",port=5001) #In colab 
