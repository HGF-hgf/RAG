from typing import List, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from openai import OpenAI
app = FastAPI()
from dotenv import load_dotenv
import os
from core import Reflection
from chatbot import get_response

load_dotenv()
class Message(BaseModel):
    message: str
    sender: str
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reflection = Reflection(client)
messages: List[Message] =  []
clients: List[WebSocket] = []


def process_message(message: str) -> str:
    # Replace with your chatbot logic
    reflectedQ = reflection(messages)
    # res = aiResponse(reflectedQ)
    res = get_response(reflectedQ)
    messages.append(Message(message=res, sender="Bot"))
    return f"Echo: {message}"



app.add_middleware(
    CORSMiddleware,
    # allow_origins=["chrome-extension:/jlpalndmgiboeonamcfpcpnabddlapoa"],
    allow_origins=["*"],  # Replace with your extension ID
    allow_methods=["*", "OPTIONS"],
    allow_headers=["*"],
)

def clearMessages():
    global messages
    messages = []



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    
    try:
        await websocket.send_text(json.dumps([message.dict() for message in messages]))
        while True:
            message = await websocket.receive_text()
            
            # Wait for new messages
            if message == "refresh":
                clearMessages()
            else: 
                messages.append(Message(message=message, sender="You"))
                # send back the message so the frontend can render
                for client in clients:
                    await client.send_text(json.dumps([message.dict() for message in messages]))
                # now we actually start processing the message
                process_message(message)
                      
            # Broadcast the message to all connected clients
            for client in clients:
                await client.send_text(json.dumps([message.dict() for message in messages]))
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {websocket.client}")
        clients.remove(websocket)



def aiResponse(message) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": message
            }
        ]
    )
    content = completion.choices[0].message.content
    return content if content is not None else ""