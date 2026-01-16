from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings 
from chunk_documents import chunk_documents 
from pathlib import Path 

INDEX_DIR=Path('vectorstore')
INDEX_DIR.mkdir(exist_ok=True)

def build_index():
    print('Loading & chunking documents...')
    chunks=chunk_documents('data/raw')

    print(f'Embedding:{len(chunks)}:chunks')

    embeddings=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLm-L6-v2'
    )

    vectorstore=FAISS.from_documents(chunks,
                                     embeddings)
    vectorstore.save_local(INDEX_DIR)

    print('FAISS:index created successfully😎')
if __name__=='__main__':
    build_index()
    