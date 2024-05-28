# import the required libraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import tempfile
import whisper
from pytube import YouTube


# load the environment variables
load_dotenv()

openai_key = os.getenv("OPEN_API_KEY")

youtube_video_url = "https://www.youtube.com/watch?v=cdiD-9MMpb0"

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

# create a chain
chain = promt | model | parser

# invoke the chain: this will invoke the model and then parse the output: output of first model will be input to the second one
print(chain.invoke({"context": "Mery's sister is susan", "question": "Who is Mary's sister?"}))


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

print(transcript[: 100])