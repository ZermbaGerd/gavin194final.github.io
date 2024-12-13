from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


articles = ["https://www.mdpi.com/2504-2289/8/11/146", "https://link.springer.com/article/10.1007/s10676-024-09792-4", "https://dl.acm.org/doi/pdf/10.1145/3442188.3445922"]

print("You will soon be speaking with the LLM. YOu can stop the conversation any time by typing <Done>, with the carrots included.")
input("Press Enter to continue: ")
print("In order to speak with the LLM, you will need to provide your Replicate API token. If you don't know what it is, check the README.txt")
print("Input it below:")

print("Loading the model...")

# Load the model - replace this with Ollama
llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")

print("Model loaded!")
print("Loading our background documents")
# load our documents
loader = WebBaseLoader(articles)
docs = loader.load()

# store those documents in a way that the model can use

# Split the document into chunks with a specified chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

# Store the document into a vector store with a specific embedding model
vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# setup conversation history
chat_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

print("Documents loaded!")

chat_history = []

print("You are now speaking with Llama 3.1 8b")
print("======================================")

while(True):
    currentQuestion = input("Input: ")

    if(currentQuestion == "<Done>"):
        break

    currentAnswer = chat_chain({"question": currentQuestion, "chat_history": chat_history})

    print("Output:", currentAnswer['answer'])

    chat_history.append((currentQuestion, currentAnswer['answer']))

print("==========================")
print("Conversation ended.")