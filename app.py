import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define request body
class ModelInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: ModelInput):
    input_dict = data.dict()
    
    # Rename to match training feature names
    df = pd.DataFrame([input_dict])
    df.rename(columns={
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)"
    }, inplace=True)
    
    prediction = model.predict(df)[0]
    #Mapping numerical prediction to class labels
    label_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    predicted_class = label_mapping.get(prediction, "Unknown")

    return {"prediction": predicted_class}
