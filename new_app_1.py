from flask import Flask, render_template, jsonify, request
from research.setup import initialize_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import torch

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# # Initialize embeddings
print("\nInitializing embeddings...")
embeddings = initialize_embeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=torch.cuda.is_available()
)

# Load existing vector store
print("\nLoading vector store from Pinecone...")
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
print(f"Vector store loaded: {index_name}")


index_name = "medical-chatbot"
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"\n{'='*60}")
        print(f"User Question: {msg}")
        print(f"{'='*60}")

        vector_store = load_existing_vector_store(index_name, embeddings)
        rag_chian = create_rag_chain(vector_store)
        # Generate answer
        response = query_rag_chain(rag_chian ,msg)
        
        print(f"Bot Response: {response}")
        print(f"{'='*60}\n")
        
        return str(response)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"Sorry, an error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)