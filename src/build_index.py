from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from pathlib import Path 
import pickle 

INDEX_DIR=Path('vectorstore')
INDEX_DIR.mkdir(exist_ok=True)

with open('data/chunks.pkl','rb')as f:
    chunks=pickle.load(f)

print(f'EMBADDING {len(chunks)}chunks...')

embeddings=HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

db=FAISS.from_documents(chunks,embeddings)

db.save_local(str(INDEX_DIR))

print('FAISS index built and saved successfully!')
import pickle 
Path('data').mkdir(exist_ok=True)

with open('data/chunks.pkl','wb')as f:
    pickle.dump(chunks,f)