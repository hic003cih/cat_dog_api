import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 載入訓練好的模型
model = load_model("model/cat_dog_cnn.h5")

# 圖片大小要與訓練時一致
IMG_SIZE = 128


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # 正規化
    img_array = np.expand_dims(img_array, axis=0) # 多加一個維度

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat" # 接近1是狗,接近0是貓
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"📸 圖片：{img_path}")
    print(f"✅ 預測結果：{label}")
    print(f"🎯 信心度：{confidence:.2f}")
    

# 測試圖片路徑（請修改為你自己的圖片路徑）
predict_image("test_images/sample_dog.jpg")
