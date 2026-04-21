from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
import importlib
import os
from rag.orchestrator import RAGOrchestrator
from utils.rate_limiter import RateLimiter
from config.validate_query import validate_query
from config.config import LIMITER_MAX_REQUESTS,LIMITER_WINDOW_SECOND

print("STARTING APP...")
print("PORT:", os.environ.get("PORT"))


app= Flask(__name__)
CORS(app)
orchestrator = RAGOrchestrator()
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
    result = orchestrator.run_groq(query)
    return result

if __name__ == "__main__":
    app.run(debug=True) # In production
    # app.run(host="0.0.0.0",port=5001) #In colab 