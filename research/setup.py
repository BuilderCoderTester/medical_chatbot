import os
import torch
from typing import List
from dotenv import load_dotenv
import langchain
print(langchain.__version__)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
print("prone")
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,

)


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def check_gpu_availability():
    """Check and print GPU availability."""
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("No GPU available, using CPU")
        return False


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    os.environ["PINECONE_API_KEY"] = pinecone_key
    os.environ["GOOGLE_API_KEY"] = google_key
    
    return pinecone_key, google_key


# ============================================================================
# DOCUMENT LOADING AND PREPROCESSING
# ============================================================================

def load_pdf_files(data_path: str) -> List[Document]:
    """
    Load PDF files from a directory.
    
    Args:
        data_path: Path to directory containing PDF files
        
    Returns:
        List of Document objects extracted from PDFs
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from PDFs")
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to contain only essential metadata.
    
    Args:
        docs: List of Document objects with full metadata
        
    Returns:
        List of Document objects with only 'source' in metadata
    """
    minimal_docs = []
    for doc in docs:
        source = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": source}
            )
        )
    return minimal_docs


def split_documents(documents: List[Document], chunk_size: int = 500, 
                    chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from documents")
    return chunks


# ============================================================================
# EMBEDDINGS (GPU-ACCELERATED)
# ============================================================================

def initialize_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                         use_gpu: bool = True):
    """
    Initialize HuggingFace embeddings model with GPU support.
    
    Args:
        model_name: Name of the HuggingFace model to use
        use_gpu: Whether to use GPU for embeddings
        
    Returns:
        HuggingFaceEmbeddings object
    """
    # Configure device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    # Model kwargs for GPU acceleration
    model_kwargs = {
        'device': device
    }
    
    # Encode kwargs for better performance
    encode_kwargs = {
        'normalize_embeddings': True,
        'batch_size': 32  # Adjust based on your GPU memory
    }
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"Initialized embeddings model: {model_name} on {device.upper()}")
    return embeddings


# ============================================================================
# PINECONE VECTOR STORE
# ============================================================================

def initialize_pinecone(api_key: str, index_name: str, dimension: int = 384):
    """
    Initialize Pinecone index.
    
    Args:
        api_key: Pinecone API key
        index_name: Name of the index to create/use
        dimension: Dimension of the embeddings
        
    Returns:
        Pinecone client and index
    """
    pc = Pinecone(api_key=api_key)
    
    if not pc.has_index(index_name):
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")
    
    index = pc.Index(index_name)
    return pc, index


def create_vector_store(documents: List[Document], embeddings, index_name: str, batch_size: int = 100):
    """
    Create vector store from documents and upload to Pinecone.
    GPU acceleration is applied through the embeddings model.
    
    Args:
        documents: List of Document objects to embed
        embeddings: Embeddings model (GPU-accelerated)
        index_name: Name of Pinecone index
        batch_size: Batch size for processing documents
        
    Returns:
        PineconeVectorStore object
    """
    print("Creating vector store and uploading documents to Pinecone...")
    print(f"Processing {len(documents)} documents in batches on GPU...")
    
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        batch_size=batch_size  # Process in batches for better GPU utilization
    )
    print("Vector store created successfully")
    return vector_store


def load_existing_vector_store(index_name: str, embeddings):
    """
    Load existing Pinecone vector store.
    
    Args:
        index_name: Name of existing Pinecone index
        embeddings: Embeddings model
        
    Returns:
        PineconeVectorStore object
    """
    print(f"Loading existing vector store: {index_name}")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return vector_store


def add_custom_documents(vector_store, documents: List[Document]):
    """
    Add custom documents to existing vector store.
    
    Args:
        vector_store: PineconeVectorStore object
        documents: List of Document objects to add
    """
    vector_store.add_documents(documents=documents)
    print(f"Added {len(documents)} custom documents to vector store")


# ============================================================================
# RAG CHAIN SETUP
# ============================================================================

def load_pretraied_quantized_model():
    loaded_model = AutoModelForCausalLM.from_pretrained(
    "./llama2-7b-4bit-quantized",
    device_map="auto",
    trust_remote_code=True
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-4bit-quantized")
    return loaded_model , loaded_tokenizer

def create_rag_chain(vector_store,  search_k: int = 3):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    
    Args:
        vector_store: PineconeVectorStore object
        model_name: Google Gemini model to use
        search_k: Number of documents to retrieve
        
    Returns:
        RAG chain for question answering
    """
    #initialize the model
    model_name , model_tokenizer = load_pretraied_quantized_model()
    # Initialize retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": search_k}
    )
    
    # Initialize chat model (Google Gemini)
    # chat_model = ChatGoogleGenerativeAI(model=model_name)
    
    # Create prompt template
    system_prompt = (
        "You are a Medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create chains
    question_answer_chain = create_stuff_documents_chain(model_name, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("RAG chain created successfully")
    return rag_chain , model_name, model_tokenizer


def query_rag_chain(rag_chain, model,tokenizer ,question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    print(inputs)
    print("do the agsugasuiydgaiuysuiaygduaygduy")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("the problem is here")
    # response = rag_chain.invoke({"input": question})
    return answer


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with GPU support."""
    
    # Check GPU availability
    print("=" * 80)
    print("GPU CHECK")
    print("=" * 80)
    gpu_available = check_gpu_availability()
    
    # 1. Load environment variables
    print("\n" + "=" * 80)
    print("STEP 1: Loading Environment Variables")
    print("=" * 80)
    pinecone_key, google_key = load_environment_variables()
    print(f"Pinecone Key: {pinecone_key[:10]}..." if pinecone_key else "Not found")
    print(f"Google Key: {google_key[:10]}..." if google_key else "Not found")
    
    # 2. Load and preprocess documents
    print("\n" + "=" * 80)
    print("STEP 2: Loading and Preprocessing Documents")
    print("=" * 80)
    raw_documents = load_pdf_files("/home/anu/Desktop/anurag/python/medicalChatbot/medical_chatbot/data")
    minimal_documents = filter_to_minimal_docs(raw_documents)
    chunked_documents = split_documents(minimal_documents)
    
    # 3. Initialize embeddings with GPU
    print("\n" + "=" * 80)
    print("STEP 3: Initializing Embeddings (GPU-Accelerated)")
    print("=" * 80)
    embeddings = initialize_embeddings(
        "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=gpu_available
    )
    
    # 4. Initialize Pinecone
    print("\n" + "=" * 80)
    print("STEP 4: Initializing Pinecone")
    print("=" * 80)
    index_name = "medical-chatbot"
    pc, index = initialize_pinecone(pinecone_key, index_name, dimension=384)
    
    # 5. Create or load vector store
    print("\n" + "=" * 80)
    print("STEP 5: Creating/Loading Vector Store (GPU-Accelerated)")
    print("=" * 80)
    
    # Uncomment to create new vector store:
    vector_store = create_vector_store(
        chunked_documents, 
        embeddings, 
        index_name,
        batch_size=100  # Larger batches for GPU
    )
    
    # Load existing vector store:
    vector_store = load_existing_vector_store(index_name, embeddings)
    
    # Optional: Add custom documents
    # custom_doc = Document(
    #     page_content="dswithbappy is a youtube channel that provides tutorials on various topics.",
    #     metadata={"source": "Youtube"}
    # )
    # add_custom_documents(vector_store, [custom_doc])
    
    # 6. Create RAG chain
    print("\n" + "=" * 80)
    print("STEP 6: Creating RAG Chain")
    print("=" * 80)
    rag_chain,model_name ,model_tokenizer = create_rag_chain(vector_store)
    
    # 7. Test queries
    print("\n" + "=" * 80)
    print("STEP 7: Testing Queries")
    print("=" * 80)
    
    test_questions = [
        "What is Acromegaly and gigantism?",
        "What is Acne?",
        "What is the Treatment of Acne?",
        "What is dswithbappy?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = query_rag_chain(rag_chain,model_name,model_tokenizer, question)
        print(f"Answer: {answer}")
        print("-" * 80)
    
    # Print GPU memory usage if available
    if gpu_available:
        print("\n" + "=" * 80)
        print("GPU MEMORY USAGE")
        print("=" * 80)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()