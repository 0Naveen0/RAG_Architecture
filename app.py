from flask import Flask,request,jsonify,render_template

from rag.orchestrator import RAGOrchestrator

from config.validate_query import validate_query



app= Flask(__name__)

@app.route("/",methods=["GET"])
def index():
  return render_template('index.html')

@app.route("/ask",methods=["POST"])
def ask():
    data = request.json
    query= data.get("query")
    is_valid,result = validate_query(query)
    if not is_valid:
        return jsonify({"error":result}),400
    query =result
   
    # query_embeddings = embeddingModel.embed(query)
    
    orchestrator = RAGOrchestrator()
    result = orchestrator.run(query)
    return result

if __name__ == "__main__":
    # app.run(debug=True) # In production
    app.run(host="0.0.0.0",port=5001) #In colab use this to expose server to ngrok