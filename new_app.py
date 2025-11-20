from flask import Flask, render_template, request
from research.new_setup import (
    initialize_embeddings,
    load_existing_vector_store,
    create_rag_chain,
    query_rag_chain,
)
from dotenv import load_dotenv
import os
import torch

app = Flask(__name__)

# =======================================
# LOAD ENVIRONMENT VARIABLES
# =======================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

index_name = "medical-chatbot"

# =======================================
# GPU CHECK
# =======================================
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# =======================================
# LOAD EMBEDDINGS (ONCE)
# =======================================
print("\nInitializing embeddings...")
embeddings = initialize_embeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=torch.cuda.is_available()
)

# =======================================
# LOAD VECTOR STORE (ONCE)
# =======================================
print("\nLoading vector store...")
vector_store = load_existing_vector_store(index_name, embeddings)
print("Vector store loaded successfully")

# =======================================
# CREATE RAG CHAIN (ONCE)
# =======================================
print("\nCreating RAG chain...")
rag_chain = create_rag_chain(vector_store)
print("RAG chain ready!\n")

# =======================================
# ROUTES
# =======================================
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]

        print("\n============================================")
        print("User:", msg)
        print("============================================")

        # Query global RAG chain
        response = query_rag_chain(rag_chain, msg)

    
        print("Bot:", response)
        print("============================================")

        return response

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Sorry, something went wrong."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
