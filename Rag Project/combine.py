
# import os
# import streamlit as st
# from qdrant_client import QdrantClient
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# import git

# # Load environment variables
# load_dotenv()

# # Set up Streamlit page
# st.set_page_config(page_title="Document & Repository Query Assistant", page_icon=":books:")
# st.title("Chat with PDF Documents or GitHub Repositories :books:")

# # Define model and embeddings
# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# # Qdrant client configuration from .env
# qdrant_url_pdf = os.getenv('QDRANT_URL_pdfchat')
# qdrant_api_key_pdf = os.getenv('QDRANT_API_KEY_pdfchat')
# qdrant_url_repo = os.getenv('QDRANT_URL_repochat')
# qdrant_api_key_repo = os.getenv('QDRANT_API_KEY_repochat')

# # Initialize memory for conversation history
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Sidebar for user selection
# option = st.sidebar.selectbox("Choose an option:", ["Chat with PDF", "Chat with GitHub Repository"])

# # Function to process PDF and extract text
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to ingest the PDF text into Qdrant
# def ingest_pdf_to_qdrant(pdf_docs):
#     try:
#         documents = []
#         for pdf in pdf_docs:
#             pdf_reader = PdfReader(pdf)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()

#             documents.append(Document(page_content=text, metadata={"source": pdf.name}))

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = text_splitter.split_documents(documents)

#         # Connect to the specific collection on the Qdrant cluster
#         qdrant = Qdrant.from_documents(
#             texts,
#             embeddings,
#             url=qdrant_url_pdf,
#             api_key=qdrant_api_key_pdf,
#             prefer_grpc=False,
#             collection_name="chat-with-pd"  # Collection name for PDFs
#         )

#         st.success("PDFs ingested successfully!")
#         return "\n".join([doc.page_content for doc in documents])

#     except Exception as e:
#         st.error(f"Error ingesting PDFs: {e}")
#         return None

# # Function to create Conversational Retrieval Chain for PDF
# def get_conversational_chain_pdf():
#     client_pdf = QdrantClient(url=qdrant_url_pdf, api_key=qdrant_api_key_pdf, prefer_grpc=False)
#     db = Qdrant(client=client_pdf, embeddings=embeddings, collection_name="chat-with-pd")
    
#     llm = ChatGroq(
#         groq_api_key=os.getenv('GROQ_API_KEY'),
#         model_name='mixtral-8x7b-32768'
#     )

#     # Create conversational retrieval chain
#     conversational_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=db.as_retriever(),
#         memory=memory
#     )
#     return conversational_chain

# # GitHub repository handling
# def clone_repo(repo_url, target_folder):
#     try:
#         git.Repo.clone_from(repo_url, target_folder)
#         st.success(f"Repository cloned successfully to {target_folder}")
#     except Exception as e:
#         st.error(f"Error cloning repository: {e}")

# # Function to extract text from GitHub repository
# def extract_text_from_repo(repo_folder):
#     text = ""
#     for root, _, files in os.walk(repo_folder):
#         for file in files:
#             if file.endswith(('.md', '.py', '.txt')):  # You can add more file types as needed
#                 file_path = os.path.join(root, file)
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     text += f.read() + "\n"  # Add a newline to separate contents
#     return text

# # Function to ingest GitHub repository content into Qdrant
# def ingest_repo_to_qdrant(repo_folder):
#     try:
#         text = extract_text_from_repo(repo_folder)
        
#         # Create a Document object for the repository content
#         documents = [Document(page_content=text, metadata={"source": repo_folder})]

#         # Split the documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = text_splitter.split_documents(documents)

#         # Ingest the documents into Qdrant vector DB
#         qdrant = Qdrant.from_documents(
#             texts,
#             embeddings,
#             url=qdrant_url_repo,
#             api_key=qdrant_api_key_repo,
#             prefer_grpc=False,
#             collection_name="repochat"  # Use the 'repochat' collection for GitHub repos
#         )

#         st.success("Repository content ingested successfully!")
#         return text

#     except Exception as e:
#         st.error(f"Error ingesting repository content: {e}")
#         return None

# # Function to create Conversational Retrieval Chain for GitHub Repos
# def get_conversational_chain_repo():
#     client_repo = QdrantClient(url=qdrant_url_repo, api_key=qdrant_api_key_repo, prefer_grpc=False)
#     db = Qdrant(client=client_repo, embeddings=embeddings, collection_name="repochat")
    
#     llm = ChatGroq(
#         groq_api_key=os.getenv('GROQ_API_KEY'),
#         model_name='mixtral-8x7b-32768'
#     )

#     # Create conversational retrieval chain
#     conversational_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=db.as_retriever(),
#         memory=memory
#     )
#     return conversational_chain

# # Main logic for PDF interaction
# if option == "Chat with PDF":
#     with st.sidebar:
#         st.subheader("Upload your PDF documents")
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

#         if st.button("Process PDFs"):
#             with st.spinner("Processing..."):
#                 try:
#                     raw_text = ingest_pdf_to_qdrant(pdf_docs)
#                     st.session_state.raw_text = raw_text
#                     st.session_state.conversational_chain = get_conversational_chain_pdf()
#                 except Exception as e:
#                     st.error(f"Error processing PDFs: {e}")

#     # Query input and processing
#     if "raw_text" in st.session_state:
#         user_query = st.text_input("Ask a question about your documents:")

#         if user_query:
#             with st.spinner("Generating response..."):
#                 try:
#                     response = st.session_state.conversational_chain(
#                         {"question": user_query}
#                     )

#                     # Display response
#                     st.write("### Assistant Response:")
#                     st.write(response['answer'])

#                 except Exception as e:
#                     st.error(f"Error during response generation: {e}")
#     else:
#         st.info("Please upload and process your PDFs first.")

# # Main logic for GitHub repository interaction
# elif option == "Chat with GitHub Repository":
#     with st.sidebar:
#         st.subheader("Input your GitHub repository URL")
#         repo_url = st.text_input("Enter GitHub Repository URL")

#         if st.button("Clone and Process Repository"):
#             if repo_url:
#                 with st.spinner("Cloning repository..."):
#                     repo_folder = "cloned_repo"
#                     clone_repo(repo_url, repo_folder)
#                     with st.spinner("Ingesting content..."):
#                         raw_text = ingest_repo_to_qdrant(repo_folder)
#                         st.session_state.raw_text = raw_text
#                         st.session_state.conversational_chain = get_conversational_chain_repo()
#             else:
#                 st.error("Please enter a valid GitHub repository URL.")

#     # Query input and processing
#     if "raw_text" in st.session_state:
#         user_query = st.text_input("Ask a question about your repository:")

#         if user_query:
#             with st.spinner("Generating response..."):
#                 try:
#                     response = st.session_state.conversational_chain(
#                         {"question": user_query}
#                     )

#                     # Display response
#                     st.write("### Assistant Response:")
#                     st.write(response['answer'])

#                 except Exception as e:
#                     st.error(f"Error during response generation: {e}")
#     else:
#         st.info("Please enter a GitHub repository URL and process it first.")


import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import git

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Document & Repository Query Assistant", page_icon=":books:")
st.title("Chat with PDF Documents or GitHub Repositories :books:")

# Define model and embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Qdrant client configuration from .env
qdrant_url_pdf = os.getenv('QDRANT_URL_pdfchat')
qdrant_api_key_pdf = os.getenv('QDRANT_API_KEY_pdfchat')
qdrant_url_repo = os.getenv('QDRANT_URL_repochat')
qdrant_api_key_repo = os.getenv('QDRANT_API_KEY_repochat')

# Initialize memory for conversation history
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Sidebar for user selection
option = st.sidebar.selectbox("Choose an option:", ["Chat with PDF", "Chat with GitHub Repository"])

# Function to process PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to ingest the PDF text into Qdrant
def ingest_pdf_to_qdrant(pdf_docs):
    try:
        documents = []
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            documents.append(Document(page_content=text, metadata={"source": pdf.name}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Connect to the specific collection on the Qdrant cluster
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url_pdf,
            api_key=qdrant_api_key_pdf,
            prefer_grpc=False,
            collection_name="chat-with-pd"  # Collection name for PDFs
        )

        st.success("PDFs ingested successfully!")
        return "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        st.error(f"Error ingesting PDFs: {e}")
        return None

# Function to create Conversational Retrieval Chain for PDF
def get_conversational_chain_pdf():
    client_pdf = QdrantClient(url=qdrant_url_pdf, api_key=qdrant_api_key_pdf, prefer_grpc=False)
    db = Qdrant(client=client_pdf, embeddings=embeddings, collection_name="chat-with-pd")
    
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name='mixtral-8x7b-32768'
    )

    # Create conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=st.session_state.memory
    )
    return conversational_chain

# GitHub repository handling
def clone_repo(repo_url, target_folder):
    try:
        git.Repo.clone_from(repo_url, target_folder)
        st.success(f"Repository cloned successfully to {target_folder}")
    except Exception as e:
        st.error(f"Error cloning repository: {e}")

# Function to extract text from GitHub repository
def extract_text_from_repo(repo_folder):
    text = ""
    for root, _, files in os.walk(repo_folder):
        for file in files:
            if file.endswith(('.md', '.py', '.txt')):  # You can add more file types as needed
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text += f.read() + "\n"  # Add a newline to separate contents
    return text

# Function to ingest GitHub repository content into Qdrant
def ingest_repo_to_qdrant(repo_folder):
    try:
        text = extract_text_from_repo(repo_folder)
        
        # Create a Document object for the repository content
        documents = [Document(page_content=text, metadata={"source": repo_folder})]

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Ingest the documents into Qdrant vector DB
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url_repo,
            api_key=qdrant_api_key_repo,
            prefer_grpc=False,
            collection_name="repochat"  # Use the 'repochat' collection for GitHub repos
        )

        st.success("Repository content ingested successfully!")
        return text

    except Exception as e:
        st.error(f"Error ingesting repository content: {e}")
        return None

# Function to create Conversational Retrieval Chain for GitHub Repos
def get_conversational_chain_repo():
    client_repo = QdrantClient(url=qdrant_url_repo, api_key=qdrant_api_key_repo, prefer_grpc=False)
    db = Qdrant(client=client_repo, embeddings=embeddings, collection_name="repochat")
    
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name='mixtral-8x7b-32768'
    )

    # Create conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=st.session_state.memory
    )
    return conversational_chain

# Main logic for PDF interaction
if option == "Chat with PDF":
    with st.sidebar:
        st.subheader("Upload your PDF documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                try:
                    raw_text = ingest_pdf_to_qdrant(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.session_state.conversational_chain = get_conversational_chain_pdf()
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Query input and processing
    if "raw_text" in st.session_state:
        user_query = st.text_input("Ask a question about your documents:")

        if user_query:
            with st.spinner("Generating response..."):
                try:
                    # Append user query to memory
                    st.session_state.memory.add_user_message(user_query)
                    
                    response = st.session_state.conversational_chain(
                        {"question": user_query}
                    )

                    # Append assistant response to memory
                    st.session_state.memory.add_assistant_message(response['answer'])

                    # Display response
                    st.write("### Assistant Response:")
                    st.write(response['answer'])

                    # Display previous messages
                    st.write("### Chat History:")
                    for msg in st.session_state.memory.messages:
                        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

                except Exception as e:
                    st.error(f"Error during response generation: {e}")
    else:
        st.info("Please upload and process your PDFs first.")

# Main logic for GitHub repository interaction
elif option == "Chat with GitHub Repository":
    with st.sidebar:
        st.subheader("Input your GitHub repository URL")
        repo_url = st.text_input("Enter GitHub Repository URL")

        if st.button("Clone and Process Repository"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    repo_folder = "cloned_repo"
                    clone_repo(repo_url, repo_folder)
                    with st.spinner("Ingesting content..."):
                        raw_text = ingest_repo_to_qdrant(repo_folder)
                        st.session_state.raw_text = raw_text
                        st.session_state.conversational_chain = get_conversational_chain_repo()
            else:
                st.error("Please enter a valid GitHub repository URL.")

    # Initialize memory for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query input and processing
if "raw_text" in st.session_state:
    user_query = st.text_input("Ask a question about your documents:")

    if user_query:
        with st.spinner("Generating response..."):
            try:
                # Generate response using the conversational chain
                response = st.session_state.conversational_chain(
                    {"question": user_query}
                )

                # Save user query and assistant response to session state for chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

                # Display response
                st.write("### Assistant Response:")
                st.write(response['answer'])

                # Display previous messages
                st.write("### Chat History:")
                for msg in st.session_state.chat_history:
                    st.write(f"{msg['role'].capitalize()}: {msg['content']}")

            except Exception as e:
                st.error(f"Error during response generation: {e}")
