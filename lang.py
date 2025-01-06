import os

import faiss
import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    raise ValueError("No API key found.")

genai.configure(api_key=api_key)

loader = PyPDFLoader("Employee-Handbook (1).pdf")
pages = loader.load_and_split()
pages = pages[4:]  
text = "\n".join([doc.page_content for doc in pages])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False, #treated as a string instead of a regex function
)
docs = text_splitter.create_documents([text]) 

for i, d in enumerate(docs):
   d.metadata = {"doc_id": i} #assigns a unique metadata id for each doc/chunck

#creating the embeddings on google's pre trained model 
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="retrieval_document")
    return embedding['embedding']

content_list = [doc.page_content for doc in docs]
#stores it in a numpy array 
embeddings = np.array([get_embeddings(content) for content in content_list])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)   #euclidean distance for nearest neighbor search
index.add(embeddings) 

def get_relevant_docs(user_query):
    query_embeddings = np.array([get_embeddings(user_query)])
    distances, indices = index.search(query_embeddings, k=3) #top 3 most relevant chunks/docs
    relevant_docs = [docs[i].page_content for i in indices[0]]
    return relevant_docs

def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. Every question asked can be found in the reference passage. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer
 
answer = generate_answer(query="Is there leave for blood donation?")
print(answer)


















'''import os
import google.generativeai as genai
import pandas as pd
import clickhouse_connect
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please set the GOOGLE_API_KEY environment variable.")

# Configure the API key for Google Generative AI
genai.configure(api_key=api_key)


loader = PyPDFLoader("Employee-Handbook (1).pdf")
pages = loader.load_and_split()
pages = pages[4:]  # Skip the first few pages as they are not required
text = "\n".join([doc.page_content for doc in pages])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.create_documents([text])
for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}
    
os.environ["GEMINI_API_KEY"] = "AIzaSyAQFDd6KerMZ0GTWENGeYciAhOcqoKpM0c"

# This function takes a a sentence as an arugument and return it's embeddings
def get_embeddings(text):
    # Define the embedding model
    model = 'models/embedding-001'
    # Get the embeddings
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="retrieval_document")
    return embedding['embedding']
# Get the page_content from the documents and create a new list
content_list = [doc.page_content for doc in docs]
# Send one page_content at a time
embeddings = [get_embeddings(content) for content in content_list]

# Create a dataframe to ingest it to the database
dataframe = pd.DataFrame({
    'page_content': content_list,
    'embeddings': embeddings
})

client = clickhouse_connect.get_client(
    host='msc-4222c802.us-east-1.aws.myscale.com',
    port=443,
    username='smartwater_org_default',
    password='passwd_Qndu6ohvZ6jwLm'
)

# Create a table with the name 'handbook'
client.command("""
    CREATE TABLE default.handbook (
        id Int64,
        page_content String,
        embeddings Array(Float32),
        CONSTRAINT check_data_length CHECK length(embeddings) = 768
    ) ENGINE = MergeTree()
    ORDER BY id
""")

# The CONSTRAINT will ensure that the length of each embedding vector is 768

# Insert the data in batches
batch_size = 10
num_batches = len(dataframe) // batch_size
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_data = dataframe[start_idx:end_idx]
    # Insert the data
    client.insert("default.handbook", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
    print(f"Batch {i+1}/{num_batches} inserted.")
# Create a vector index for a quick retrieval of data
client.command("""
ALTER TABLE default.handbook
    ADD VECTOR INDEX vector_index embeddings
    TYPE MSTG
""")

def get_relevant_docs(user_query):
    # Call the get_embeddings function again to convert user query into vector embeddngs
    query_embeddings = get_embeddings(user_query)
    # Make the query
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.handbook ORDER BY dist LIMIT 3
    """)
    relevant_docs = []
    for row in results.named_results():
        relevant_docs.append(row['page_content'])
    return relevant_docs

def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer
answer = generate_answer(query="what is the Work Dress Code?")
print(answer)
'''

'''from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os


load_dotenv()  

#0.5-0.8 
llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings()

pdf_loader = PyPDFLoader('/Users/arushipradhan/chatbot/BeeMovie.pdf') 
docs = pdf_loader.load() 

vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=retriever
)

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = rag_chain.run(query)
    print(f"Bot: {response}")
    '''
