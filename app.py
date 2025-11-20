from flask import Flask, render_template, jsonify, request
from transformers import (
    AutoTokenizer,AutoModelForCausalLM
)
from research.setup import initialize_embeddings ,create_rag_chain ,query_rag_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = initialize_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


def load_pretraied_quantized_model():
    loaded_model = AutoModelForCausalLM.from_pretrained(
    "./llama2-7b-4bit-quantized",
    device_map="auto",
    trust_remote_code=True
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-4bit-quantized")
    return loaded_model , loaded_tokenizer



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = load_pretraied_quantized_model()
model_name, model_tokenizer = create_rag_chain()

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = query_rag_chain(model_name, model_tokenizer,input)
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



