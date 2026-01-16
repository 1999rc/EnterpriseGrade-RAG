from sentence_transformers import SentenceTransformer 
def main():
    model=SentenceTransformer('sentence-transformers/all-MiniLm-L6-v2')
    print("Embedding model Loadded successfully!")
    
if __name__=='__main__':
    main()