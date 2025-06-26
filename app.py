import streamlit as st  # type: ignore
from dotenv import load_dotenv # type: ignore
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import CharacterTextSplitter  # type: ignore


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
                  text_chunks = get_text_chunks(raw_text)
                  st.write(text_chunks) 



           #get the text chunks


           #create the vector store


        

if __name__ == "__main__":
    main()
