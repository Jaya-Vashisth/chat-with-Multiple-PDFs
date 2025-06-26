import streamlit as st  # type: ignore
from dotenv import load_dotenv # type: ignore
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import CharacterTextSplitter  # type: ignore
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings# type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
# langchain_community.embeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langhain.chains import ConversationalRetrievalChain # type: ignore
# from langchain.llms import OpenAI # type: ignore
from langchain.chatModels import ChatOpenAI # type: ignore
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks



#generate embeddings and create a vector store
def get_vector_store(text_chunks):
    # 1. Add instruction prefix to each chunk
    chunks_with_prefix = [
        f"Represent this passage for retrieval: {chunk}"
        for chunk in text_chunks
    ]

    # 2. Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # 3. Create vector store using prefixed chunks
    vectorstore = FAISS.from_texts(texts=chunks_with_prefix, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    converation_chain =  ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

    return converation_chain

def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
    )
    
    st.header("chat with multiple PDFs :books:")
    st.text_input("Ask a question about your PDFs", key="question")

    with st.sidebar:
      st.subheader("Your documents")
      pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process", accept_multiple_files=True, type=["pdf"], key="pdf_uploader")
      
      if st.button("Process", key="process_button"):
            with st.spinner("Processing your PDFs..."):
                 
                  #get the pdf text
                  raw_text = get_pdf_text(pdf_docs)

                  #get the text chunks
                  text_chunks = get_text_chunks(raw_text)
                  #st.write(text_chunks)
                  
                  #create the vector store
                  vector_store = get_vector_store(text_chunks)
                  #st.write("Vector store created successfully!")

                  ## create conversation chain
                  converstation = get_conversation_chain(vector_store)


  
        



           




        

if __name__ == "__main__":
    main()
