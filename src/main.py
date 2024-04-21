from fastapi import FastAPI
from src.prediction import predict_output
from typing import List
from pydantic import BaseModel


class InputData(BaseModel):
    Demand: List[int]
    Cost: List[int]
    Recession: List[str]
    Economy: List[str]
    Competition: List[str]
    Market_Size: List[int]


app = FastAPI()


@app.post("/predict")
def predict(input_data: InputData):
    print(input_data.model_dump())

    result = predict_output(
        input_data.Cost,
        input_data.Demand,
        input_data.Recession,
        input_data.Economy,
        input_data.Competition,
        input_data.Market_Size,
    )

    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
