from langchain_core.document_loader import PyPDfLoader , DirectoryLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import HuggingFaceEmbeddings
from typing import List
from langchian_core.schemas import Document


# get the pdf 
def load_pdf(data):
    loader = DirectoryLoader(data, glob ="*.pdf",loader_cls=PyPDfLoader)
    documetns = loader.load()
    return documetns

# filter to minimal document s
def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

