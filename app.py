from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
import os

app = FastAPI()

# Load the model
model_path = "F:\TRIAL\my_fruit_quality_model.h5"
model = tf.keras.models.load_model(model_path)

# Static files configuration
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML template route
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return f.read()

# Route for predicting fruit quality
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image
    image_resized = cv2.resize(img, (150, 150))
    image_normalized = image_resized.astype('float32') / 255.0
    image_normalized = np.expand_dims(image_normalized, axis=0)

    # Make prediction
    prediction = model.predict(image_normalized)
    percentage_quality = prediction[0][0] * 100
    quality = "Good" if percentage_quality > 50 else "Bad"
    
    return {"quality": quality, "percentage": percentage_quality}

if __name__ == "__main__":
    # Run the server with: uvicorn app:app --reload
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4000)
