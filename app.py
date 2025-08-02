import streamlit as st  # type: ignore
# from dotenv import load_dotenv # type: ignore
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import CharacterTextSplitter  # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain.chains import ConversationalRetrievalChain # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore
from htmlTemplate import css,bot_template,user_template
from langchain.embeddings import OpenAIEmbeddings # type: ignore           
import os

api_key = st.secrets.get("OPENAI_API_KEY") 
os.environ["OPENAI_API_KEY"] = api_key

#Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text

#Function to split the text into chunks
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
    
    # 1. Load embedding model
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")


    # 2. Create vector store using prefixed chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


#Function to create a conversation chain using the vector store
def get_conversation_chain(vector_store):
    


    llm = ChatOpenAI()  # type: ignore


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    conversation_chain =  ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

    return conversation_chain



#Function to handle user input and display the response
def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:   
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():

  

    # load_dotenv()
    st.set_page_config(page_title = "chat with multiple PDFs", page_icon = ":books:")  # Set wide layout for the app
    st.write(css, unsafe_allow_html=True) 

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your PDFs", key="question")

    if(user_question):
        handle_user_input(user_question)

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
                  st.session_state.conversation = get_conversation_chain(vector_store)
      

if __name__ == "__main__":
    main()
