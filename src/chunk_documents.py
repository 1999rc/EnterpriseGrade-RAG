from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_pdfs import load_pdfs 

def chunk_documents(pdf_dir:str,chunk_size=500,overlap=100):
    docs=load_pdfs(pdf_dir)
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks=splitter.split_documents(docs)
    return chunks

if __name__=='__main__':
    chunks=chunk_documents('data/raw')
    print(f'Created:{len(chunks)}:chunks👍')