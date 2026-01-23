# Put the code for your API here.
"""
FastAPI application for Census Income Classification.

This API provides endpoints for:
- GET /: Welcome message
- POST /predict: Model inference for income prediction
"""

from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Import ML modules
from ml.data import process_data, get_categorical_features
from ml.model import inference, load_model, load_encoder


# # DVC pull on Heroku
# if "DYNO" in os.environ and os.path.isdir(".dvc"): # pragma: no cover
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")


# Global variables for model and encoders
model = None
encoder = None
lb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and encoders on startup."""
    global model, encoder, lb

    model_path = "model/model.pkl"
    encoder_path = "model/encoder.pkl"
    lb_path = "model/lb.pkl"

    model = load_model(model_path)
    encoder = load_encoder(encoder_path)
    lb = load_encoder(lb_path)

    yield
    # Cleanup (if needed)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Census Income Classification API",
    description="API for predicting whether income exceeds $50K/year "
                "based on census data.",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic model for input data with examples
# Using Field aliases to handle hyphenated column names
class CensusData(BaseModel):
    """Input data model for census income prediction."""

    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Literal["<=50K", ">50K"]


@app.get("/")
async def root():
    """
    Root endpoint returning a welcome message.

    Returns
    -------
    dict
        Welcome message.
    """
    return {
        "message": "Welcome to the Census Income Classification API!",
        "description": "Use POST /predict to get income predictions.",
        "docs": "Visit /docs for API documentation."
    }


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(data: CensusData):
#     """
#     Predict income class based on census data.
#
#     Parameters
#     ----------
#     data : CensusData
#         Census data for prediction.
#
#     Returns
#     -------
#     PredictionResponse
#         Predicted income class (<=50K or >50K).
#     """
#     # Convert input data to DataFrame with original column names (with hyphens)
#     input_dict = {
#         "age": data.age,
#         "workclass": data.workclass,
#         "fnlgt": data.fnlgt,
#         "education": data.education,
#         "education-num": data.education_num,
#         "marital-status": data.marital_status,
#         "occupation": data.occupation,
#         "relationship": data.relationship,
#         "race": data.race,
#         "sex": data.sex,
#         "capital-gain": data.capital_gain,
#         "capital-loss": data.capital_loss,
#         "hours-per-week": data.hours_per_week,
#         "native-country": data.native_country,
#     }
#
#     df = pd.DataFrame([input_dict])
#
#     # Get categorical features
#     cat_features = get_categorical_features()
#
#     # Process data
#     X, _, _, _ = process_data(
#         df,
#         categorical_features=cat_features,
#         label=None,
#         training=False,
#         encoder=encoder,
#         lb=lb
#     )
#
#     # Get prediction
#     pred = inference(model, X)
#
#     # Convert prediction back to label
#     prediction_label = lb.inverse_transform(pred)[0]
#
#     return PredictionResponse(prediction=prediction_label)

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CensusData):
    """
    Predict income class based on census data.

    Parameters
    ----------
    data : CensusData
        Census data for prediction.

    Returns
    -------
    PredictionResponse
        Predicted income class (<=50K or >50K).
    """
    # Optimized: Automatically convert Pydantic model to dict using the aliases
    # defined in CensusData (e.g., 'education_num' -> 'education-num')
    input_dict = data.model_dump(by_alias=True)

    df = pd.DataFrame([input_dict])

    # Get categorical features
    cat_features = get_categorical_features()

    # Process data
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Get prediction
    pred = inference(model, X)

    # Convert prediction back to label
    prediction_label = lb.inverse_transform(pred)[0]

    return PredictionResponse(prediction=prediction_label)


# Health check endpoint (optional but useful)
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
