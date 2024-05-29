# import the required libraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import tempfile
import whisper
from pytube import YouTube


# load the environment variables
load_dotenv()

openai_key = os.getenv("OPEN_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

youtube_video_url = "https://www.youtube.com/watch?v=BrsocJb-fAo"

# create a model
model = ChatOpenAI(openai_api_key=openai_key, model="gpt-4o")

# create output parser: this will help to parse the output of the model to plain text/string
parser = StrOutputParser()

# create a prompt
template = """
    Answer the question based on the context below. If you can't answer, just say "I don't know".
    
    Context: {context}
    
    Question: {question}
"""

# create a prompt object, this will pass the context and question to the template
promt = ChatPromptTemplate.from_template(template)

# let's create a transcript from the youtube video
if not os.path.exists("transcripts.txt"):
    youtube = YouTube(youtube_video_url)
    audio = youtube.streams.filter(only_audio=True).first()

    # lets load the model
    whisper_model = whisper.load_model('base') # load the base model - this is not the best model, but let's use it for now
    with tempfile.TemporaryDirectory() as tmpdirname:
        file = audio.download(output_path=tmpdirname)
        transcript = whisper_model.transcribe(file, fp16=False)['text'].strip()
        with open("transcripts.txt", "w") as f:
            f.write(transcript)

# read the transcript
with open("transcripts.txt", "r") as f:
    transcript = f.read()


# Split the transcript: there is a limit to the number of tokens that can be passed to the model so we need to split the transcript, let's say transcript can be in gb so we need to split it into smaller chunks
# 1. Convert text to text documents

loader = TextLoader('transcripts.txt')
text_documents = loader.load()

# 2. Split the text documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text_documents_chunks = text_splitter.split_documents(text_documents)


# Generate embeddings for the text text_documents_chunks: this will help to generate embeddings for the text chunks and then we can use the embeddings to find the most relevant text chunk to the question

# Cohere playground is a good place to play arounds documents and see embeddings

embeddings = OpenAIEmbeddings(api_key=openai_key)

# Setup the pipeline vector database 
index_name = "youtube-rag-index"

pinecone = PineconeVectorStore.from_documents(documents=text_documents_chunks, embedding=embeddings, pinecone_api_key=pinecone_key, index_name=index_name)

chain = (
    {
        "context": pinecone.as_retriever(),
        "question": RunnablePassthrough()
    }
    | promt 
    | model 
    | parser
)


print(chain.invoke('What is RAG Application?'))
