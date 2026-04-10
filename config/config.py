#Configuration File

EMBEDDING_MODEL_NAME  ="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# CHROMA_DB_PATH        = "/db/chroma_db"
# CHROMA_DB_PATH        = "/content/drive/MyDrive/Colab Notebooks/rag-docs/vectordb/chroma_db"
CHROMA_DB_PATH         = "/content/drive/MyDrive/ColabNotebooks/EKA_RAG_Project_v2/db/chroma_db"
TOP_K				           = 7	
RERANKER_MODEL_NAME    = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_TOP_K         = 3
SIMILARITY_THRESHOLD   = 0.12  #0.08
SIMILARITY_THRESHOLD_ACCEPTED = 0.45

GAP                    =0.05   #0.02
MAX_CONTEXT_TOKENS	   = 1500	
MAX_NEW_TOKENS 		     = 350
TEMPERATURE			       = 0.2
DO_SAMPLE			         = False
TOP_P                  = 0.9
REPETITION_PENALTY    = 1.1
MAX_TIME              = 30.0
GUARD_ACTIVE          = True
CHUNK_SIZE            = 400
CHUNK_OVERLAP         = 50
MAX_ALLOWED_CHUNKS    = 2
RAW_DOCS_PATH         = "/content/drive/MyDrive/ColabNotebooks/EKA_RAG_Project/data/raw_docs/text"
# RAW_DOCS_PATH         = "/data/raw_docs/text"
# RAW_DOCS_PATH         = "/content/drive/MyDrive/Colab Notebooks/rag-docs"
REFUSAL_MESSAGE = "I do not have sufficient information in the knowledge base to answer this question."
MAX_TOTAL_LATENCY = 45.0
MAX_GENERATION_LATENCY = 30.0
LOG_PATH                = "/content/drive/MyDrive/ColabNotebooks/EKA_RAG_Project_v2/observability/logs/requests.json"