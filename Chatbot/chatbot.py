from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from tools import initial_tools
from config import OPENAI_API_KEY
from semantic_router import SemanticRouter, Route
from sample_query import productsSample, chitchatSample
from STT.EchoAds.Text2Speech.tts import generate_text_to_speech
from STT.EchoAds.Speech2text.stt import transcribe_audio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import os
from pydantic import BaseModel


llm = OpenAI(model="gpt-4o-mini", temperature=0)

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

productsRoute = Route("products", productsSample)
chitchatRoute = Route("chitchat", chitchatSample)

# Create a SemanticRouter with the defined routes
semantic_router = SemanticRouter([productsRoute, chitchatRoute])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    text: str

@app.post("/voice-query/")
async def voice_query(audio: UploadFile = File(...)):
    temp_audio_path = f"temp_{audio.filename}"
    with open(temp_audio_path, "wb") as buffer:
        buffer.write(await audio.read())

    text_query = transcribe_audio(temp_audio_path, "vi")
    os.remove(temp_audio_path)  
    print(text_query, type(text_query))
    best_route = semantic_router.guide(text_query)
    response = agent.query(text_query)

    return {
        "transcribed_text": text_query,
        "best_matching_route": best_route[1],
        "score": best_route[0],
        "response": str(response),
    }

@app.post("/speak/")
async def speak(request: SpeechRequest):
    if not request.text:
        return {"error": "Text field is required"}
    else: 
        audio_file_path = generate_text_to_speech(request.text, "Vietnamese")
        if audio_file_path:
            return {"audio_file": audio_file_path}
        else:
            return {"error": "Cannot generate audio file"}
