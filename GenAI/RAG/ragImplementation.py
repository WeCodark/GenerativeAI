# To run below libraries
# run "!pip install -qU langchain-google-genai langchain-community langchain chromadb"

from langchain_community.document_loaders import TextLoader # pypdfloader for PDF input
from langchain_text_splitters import RecursiveCharacterTextSplitter # This will help us to split the docs into smaller chuncks
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # This will help us to create embeddings for the chunks
from langchain_community.vectorstores import Chroma # This will help us to create a vector store
from langchain_core.prompts import ChatPromptTemplate # This will help us to create a prompt template
from langchain_core.output_parsers import StrOutputParser # This will help us to parse the output of the model
from langchain_core.runnables import RunnablePassthrough # This will help us to create a runnable chain
import os # This will help us to get the environment variables
import getpass # This will help us to get the password
import sys # This will help us to exit the program
import gradio as gr

# We are using Google Gemini as our LLM model
GOOGLE_API_KEY = "AIzaSyA8RQH5Z4x2KEJ530lTs5-X5iLrrZncT1E" # Please be careful as we should not expose API KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def setup_env():
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")


# We are loading the data from the text file and splitting it into chunks
def load_and_split(filepath):
    # Checking if the file exists
    if not os.path.exists(filepath):
        print(f'Error: File not found at {filepath}')
        sys.exit(1)  

    # Loading the data from the text file
    print(f'Loading Data')
    loader = TextLoader(filepath)
    # For pdf
    # loader = PyPDFLoader(filepath)
    docs = loader.load()

    # Splitting the data into chunks
    print(f'Splitting Data into chunks')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f'Split {len(splits)} chunks')
    return splits


# We are creating a vector store and creating a RAG chain
def create_rag_chain(splits):
    print(f'Intialize Vector Store and Create Rag Chain')
    # We need to embbed the data into vectors
    # taskType -> "retrieval_document" for embedding documents
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",task_type="retrieval_document")
    # We will now pass embedding to my vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever() # This is my context

    # Since our Retriever is ready, We will create our LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)

    template = """Answer the question based only on the following context: {context}
        Question: {question}

        Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # We are creating Doc content pages
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG CHAIN
    chain = (
        {"context": retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
    


setup_env()

file_path = 'harrypotter.txt'
splits = load_and_split(file_path)
rag_chain = create_rag_chain(splits)

# message = input('Ask Question: ')
# output = rag_chain.invoke(message)
# print(output)

# Gradio Setup
# Alothough we are not using history param, still i am defining it because chatInterface accepts 2 arguments in function
def respond(message, history):
    try:
        return rag_chain.invoke(message)
    except Exception as e:
        return f'An error occured {e}'


demo = gr.ChatInterface(
    fn = respond,
    title = "Harry Potter RAG Explorer",
    description="You can ask anything about Harry potter based on provided context",
    textbox=gr.Textbox(placeholder='Ask a question regarding Harry Potter', container=False, scale = 7)
)

demo.launch(share=True)