# Import all the packages and modules required for the FastAPI server.
from fastapi import FastAPI, Request, Response
from kafka import KafkaProducer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from starlette.concurrency import iterate_in_threadpool
import json
from kafka_sdk import (
    TrafficProcessingSDK,
)


app = FastAPI()

producer = KafkaProducer(bootstrap_servers="localhost:9092")


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
    response = await call_next(request)

    # Decode the bytes into a string
    body_str = body_bytes.decode()

    # Parse the string into a dictionary
    body_dict = json.loads(body_str)

    # Log the dictionary

    # Log the dictionary only if the request path matches a specific route
    if request.url.path == "/items":
        print("Received JSON:", body_dict)

        # Call the next middleware in the chain
        # Process the request
        sdk.process_request(request.url.path, request.method, body_dict)

    return response


@app.post("/items")
async def create_item(item: dict):
    return {"The data saved successfully": item}
