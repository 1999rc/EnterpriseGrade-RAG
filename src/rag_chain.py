from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

from langchain_community.llms import Ollama 
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate 

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLm-L6-v2')
vectorstore=FAISS.load_local("vectorstore",embeddings,allow_dangerous_deserialization=True)

template="""Use the following pieces of 
            context to answer the question at the end.
            if you don't know the answer,just say that you 
            don't know,don't try to make up an answer.
            {context}
            
            Question:{question}
            Helpful Answer:"""
QA_CHAIN_PROMPT=PromptTemplate.from_template(template)

llm=Ollama(model="llama3") #you change this to mistral or phi3

qa_chain=RetrievalQA.from_chain_type(
 llm=llm,
 chain_type="stuff",
 retriever=vectorstore.as_retriever(), #(search_keywords={"k":3}),
 chain_type_kwargs={
     "prompt":QA_CHAIN_PROMPT}
 
)

if __name__=='__main__':
    print('😁\n... PDF ChainBot active ----✌')
    while True :
        user_input=input('😎\nASk a question (or type "exit"):')
        if user_input.lower()=="exit":
            break 
        response=qa_chain.invoke(user_input)
        print(f'👏\nAI: {response["result"]}')
