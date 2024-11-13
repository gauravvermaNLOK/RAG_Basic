from langchain_community.document_loaders import PyPDFDirectoryLoader

# Specify the directory where your PDFs are stored
pdf_directory = 'InputData'

# Initialize the PyPDFDirectoryLoader to load all PDFs present in the directory
loader = PyPDFDirectoryLoader(pdf_directory)

# Load the documents from the PDFs
listOfDocuments = loader.load()
print("number of documents loaded: ", len(listOfDocuments))  #output: 150

#Output: first 20 charcters of first page i.e. ECONOMIC SURVEY 2023 from IndianEconomySurvey2023-24.pdf
print("content of first document is: ", listOfDocuments[0].page_content[:20])

from langchain.text_splitter import RecursiveCharacterTextSplitter

#Split the document in chunk where each chunk has 500 tokens
# and 300 tokens are overlapping in adjacent chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                               chunk_overlap=100)
listOfSplittedDocList = text_splitter.split_documents(listOfDocuments)


print("number of documents after splitting is ", len(listOfSplittedDocList))

print("Content of first chunk: ", listOfSplittedDocList[0].page_content)
print("#######################################################")
print("Content of second chunk: ", listOfSplittedDocList[1].page_content)

from dotenv import load_dotenv

load_dotenv()  #load the api key

from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Create object of "GoogleGenerativeAIEmbeddings"
# using embedding model "embedding-001" provided by Google

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Create chroma vector db from list of documents
from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(listOfSplittedDocList, embedding_model)

#Initialize the VectorStoreRetriever to retrieve the data from Vector DB
#search_kwargs={"k": 3} - Amount of documents to return. Default is 4
vector_store_retriever = vectordb.as_retriever(search_kwargs={"k": 3}, search_type="similarity")

#to get the relevant information -
# you can call invoke function of VectorStoreRetriever by passing your query as parameter
listOfRelevantDocs = vector_store_retriever.invoke("what is gdp")
#print(listOfRelevantDocs)  #Print the information
# This will return information along with metadata in below format
"""
[Document(metadata={'page': 1, 'source': 'InputData\\PressReleaseApr_to_Jun2024-25.pdf'}, page_content='This Press Release is embargoed against publication, telecast or circulation on internet till 5.30 pm on 30th August,
Document(metadata={'page': 7, 'source': 'InputData\\PressReleaseApr_to_Jun2024-25.pdf'}, page_content='3.1 Trade, Hotels, Transport, Communication,
Document(metadata={'page': 1, 'source': 'InputData\\PressReleaseApr_to_Jun2024-25.pdf'}, page_content='its expenditure components both at Constant (2011 -12) ]
"""
#This content is not human friendly,
# so we will instruct our model to generate user-friendly response
#To provide the instruction to system crate a template

#Prompt template must contain input variable "context",
# which will be used for passing in the formatted documents.
prompt_template = ("strictly Use only the given context to answer the question. "
                   "If you don't find the answer in given {context}, only say one sentence that I don't have "
                   "information about this ."
                   "if you find the the answer in given {context} then Use five sentence maximum and keep the answer "
                   "concise."
                   )

#Create structured template for  LLM
from langchain_core.prompts import ChatPromptTemplate

chatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

#Define LLM model
from dotenv import load_dotenv
import os

load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=apikey)

#Use 'create_stuff_documents_chain' to create a chain that stuffs (concatenates)
#the docs into a prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(model, chatPromptTemplate)


#Generate output from Custom Dataset (which are PDFs in this case)
from langchain.chains.retrieval import create_retrieval_chain

rag_chain = create_retrieval_chain(vector_store_retriever, question_answer_chain)
user_query = "What is GDP of India in 2024"
response = rag_chain.invoke({"input": user_query})
print("#######################################################")
print("Response is ", response['answer'])