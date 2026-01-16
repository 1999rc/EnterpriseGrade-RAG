from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from pathlib import Path 
import pickle
from langchain_community.document_loaders import DirectoryLoader
RAW_DIR='data/raw'
CHUNKS_PATH='data/chunks.pkl'

Path('data').mkdir(exist_ok=True)

print('\n--- 1.Loading PDFs from data/raw ---')

loader=DirectoryLoader(
    RAW_DIR,
    glob='./*.pdf',
    loader_cls=PyPDFLoader
)
try:
    docs=loader.load()
    print(f'Successfully loaded {len(docs)}:pages:')
except Exception as e:
    print(f'Error loading PDFs:{e}')
    docs=[]
if docs:
    print('---2. Splitting Documnets into Chunks ----')
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n","\n","",""]
    )
    chunks=splitter.split_documents(docs)
    print(f'Created {len(chunks)}:text chunk.')

    with open(CHUNKS_PATH,'wb')as f:
        pickle.dump(chunks,f)
    print(f'Saved chunks to {CHUNKS_PATH}')
else:
    print('No documents found😁 to process!')
'''RAW_DIR=Path('data/raw')

docs=[]

for pdf in RAW_DIR.glob('*.pdf'):
    loader=PyPDFLoader(str(pdf))
    docs.extend(loader.load())

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks=splitter.split_documents(docs)

Path('data').mkdir(exist_ok=True)
with open('data/chunks.pkl','wb')as f:
    pickle.dump(chunks,f)
print('Chunks saved to data/chunks.pkl')
print(f"Loaded PDFs:{len(list(RAW_DIR.glob('*.pdf')))}")
print(f"Total pages:{len(docs)}")
print(f"Total chunks:{len(chunks)}")'''