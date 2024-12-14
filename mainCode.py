from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# This is the list of articles which the chatbot will be able to access. Having more makes it load slower, but expands its knowledge base

#============== ARTICLES ==================#
articles = ["https://www.mdpi.com/2504-2289/8/11/146", "https://link.springer.com/article/10.1007/s10676-024-09792-4", 
            "https://dl.acm.org/doi/pdf/10.1145/3442188.3445922"]

articleNames = ["Leveraging Large Language Models for Enhancing Literature-Based Discovery", 
                "Easy-read and large language models: on the ethical dimensions of LLM-based text simplification", 
                "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"]
#============== ARTICLES ==================#

# Load the articles from their URLs
loader = WebBaseLoader(articles)
print("Loading documents, this may take a while...")
docs = loader.load()

# Split the document into chunks with a specified chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

# Store the document into a vector store with a specific embedding model
vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
print("Documents loaded!")


#================== LLM =================#
print("Loading the model...")
llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
#================== LLM =================#
print("Model loaded!")

# setup conversation history with documents as filtering step
chat_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)


chat_history = []

print("You are now speaking with Llama 3.1 8b.")
print("======================================")

while(True):
    currentQuestion = input("Input: ")

    if(currentQuestion.lower() in ["<Done>", "done", "no", "exit", "quit", "n", "q", "stop"]):
        break
    
    promptPrefix = """You are a conversational chatbot interacting with a curious user. You have access to a selection of academic papers and articles from newspapers, 
    which largely cover the topic of ethics in AI. The sources' names are as such: {}. Answer the user's following question respectfully, in a way that invites conversation and questioning.
    If you don't know the answer to the question, or there are debates about what the correct answer is, explicitly recognize that and give examples of how a user could navigate that uncertainty.
    The user's question is: """.format(articleNames)

    currentPrompt = promptPrefix + currentQuestion
    print("thinking...")
    currentAnswer = chat_chain({"question": currentQuestion, "chat_history": chat_history})

    print("Output:", currentAnswer['answer'])

    chat_history.append((currentQuestion, currentAnswer['answer']))

print("==========================")
print("Conversation ended.")