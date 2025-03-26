from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import shutil
import os

app = FastAPI()

# 設定模板與靜態資源
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 載入模型
model = load_model("model/cat_dog_cnn.h5")
IMG_SIZE = 128  # ← 如果你用 MobileNet，請改成 224

# 定義圖片預處理函數
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# /predict 路由：接收圖片，回傳預測
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    upload_path = f"static/uploaded/{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        image_array = preprocess_image(upload_path)
        prediction = model.predict(image_array)[0][0]
        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 2)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "label": label,
            "confidence": confidence,
            "img_url": "/" + upload_path
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "label": "錯誤",
            "confidence": "無法預測",
            "img_url": "",
            "error": str(e)
        })
