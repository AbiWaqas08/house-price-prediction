from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("model\house_price_model.pkl")

app = FastAPI(
    title="House Price Prediction",
    description="HOuse price prsdiction",
    version="1.0.0"
)

# INPUT SCHEMA
class HouseInput(BaseModel):
    Square_feet : float
    num_rooms : int
    age : int
    distance_to_city : float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #eef2f7;
                padding: 6px;
            }
            .container {
                max-width: 700px;
                margin: auto;
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 {
                color: #2c3e50;
            }
            p {
                font-size: 16px;
                color: #555;
                line-height: 1.6;
            }
            ul {
                text-align: left;
                margin-top: 20px;
            }
            li {
                margin-bottom: 10px;
            }
            .btn {
                display: inline-block;
                margin-top: 30px;
                padding: 12px 25px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-size: 16px;
            }
            .btn:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>House Price Prediction API</h1>

            <p>
                This project is a Machine Learning–powered API built using
                <strong>FastAPI</strong>. It predicts house prices based on
                key features such as house size, number of rooms, age,
                and distance from the city center.
            </p>

            <p>
                The model is trained using historical housing data and exposed
                as a REST API for easy integration into web or mobile applications.
            </p>

            <ul>
                <li>✔ Built with FastAPI</li>
                <li>✔ Machine Learning model loaded using Joblib</li>
                <li>✔ Supports real-time predictions</li>
                <li>✔ Interactive Swagger documentation</li>
            </ul>

            <a href="/docs" class="btn">Go to API Documentation</a>
        </div>
    </body>
    </html>
    """

# Prediction api end point
@app.post("/predict")
def predict(data: HouseInput):
    features = np.array([[
        data.Square_feet,
        data.num_rooms,
        data.age,
        data.distance_to_city,
    ]])

    prediction = model.predict(features)[0]

    return {
        "prediction_price": round(float(prediction), 2)
    }
