import tensorflow as tf
# import tensorflow_datasets as tfds
import os

print("🐱 貓圖數量:", len(os.listdir("cat_dog_dataset/train/cats")))
print("🐶 狗圖數量:", len(os.listdir("cat_dog_dataset/train/dogs")))

# 載入資料集
# (ds_train, ds_val), ds_info = tfds.load(
#     'cats_vs_dogs',
#     split=['train[:80%]', 'train[80%:]'],
#     with_info=True,
#     as_supervised=True,
# )

# 資料預處理
IMG_SIZE = 128
BATCH_SIZE = 32

train_dir = "cat_dog_dataset/train"
val_dir = "cat_dog_dataset/validation"

ds_train = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# 加入資料預處理
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("🔥 開始訓練模型")
model.fit(ds_train, validation_data=ds_val, epochs=5)
print("✅ 訓練完成")
# 儲存
model.save("model/cat_dog_cnn.h5")
