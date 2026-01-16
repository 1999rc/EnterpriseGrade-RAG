from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader 

def load_pdfs(pdf_dir:str):
    documents=[]
    pdf_path=Path(pdf_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f'{pdf_dir}:not found👀')
    for pdf in pdf_path.glob('*.pdf'):
        loader=PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents 

if __name__=='__main__':
    docs=load_pdfs('data/raw')
    print(f'Loaded👏{len(docs)}:pages')