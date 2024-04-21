from fastapi import FastAPI, Request, Response
from kafka import KafkaProducer
import json
# Import all the packages and modules required for the FastAPI server.
from src.kafka_sdk import (
    TrafficProcessingSDK,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi import FastAPI, Request
from starlette.concurrency import iterate_in_threadpool


app = FastAPI()

producer = KafkaProducer(bootstrap_servers='localhost:9092')


import json

# Create an instance of the FastAPI class.
app = FastAPI()

# Define the origins for the CORS middleware.
origins = [
    "*",
]

# Add middlewares to the origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an instance of the TrafficProcessingSDK class and consume
kafka_bootstrap_servers = "localhost:9092"
group_id = "traffic-processing-group"
sdk = TrafficProcessingSDK(kafka_bootstrap_servers, group_id)

@app.middleware("http")
async def some_middleware(request: Request, call_next):
    # Read the request body
    body_bytes = await request.body()

    # Decode the bytes into a string
    body_str = body_bytes.decode()

    # Parse the string into a dictionary
    body_dict = json.loads(body_str)

    # Log the dictionary
    print("Received JSON:", body_dict)

    # Process the request
    sdk.process_request(request.url.path, request.method, body_dict)

    # Call the next middleware in the chain
    response = await call_next(request)

    return response


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/items")
async def create_item(item: dict):
    return item