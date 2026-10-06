import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
from rembg import remove
import os
from style import apply_style, lede

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

apply_style("Demo Neural Network", ":material/back_hand:")

st.title("Demo Neural Network")
lede("อัปโหลดภาพมือ แล้วให้โมเดล MobileNetV2 ทายว่าเป็น ค้อน กระดาษ หรือ กรรไกร")

@st.cache_resource
def load_dataset():
    dataset_name = "rock_paper_scissors"
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset, info

dataset, info = load_dataset()

# แยก train และ test
train_data, test_data = dataset["train"], dataset["test"]

# กำหนดขนาดภาพและbatch
batch_size = 32
image_size = (128, 128)  

# ฟังก์ชันปรับแต่งข้อมูล(Preprocessing)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    image = tf.image.resize(image, image_size)  # Resize
    return image, label

# ฟังก์ชันเพิ่มข้อมูล(Augmentation)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    return image, label

# ชุดข้อมูลสำหรับการTrain
train_data = (
    train_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Preprocess ก่อน
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)    # Augment ทีหลัง
    .shuffle(1000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# ชุดข้อมูลสำหรับการTest
test_data = (
    test_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenetv2_model.keras')
    return model

def train_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
  
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}  
    model.fit(train_data, epochs=20, validation_data=test_data, class_weight=class_weights, callbacks=[early_stopping])

    model.save('mobilenetv2_model.keras')  # บันทึกโมเดลที่ฝึกเสร็จแล้ว
    return model

# ตรวจสอบว่ามีโมเดลที่บันทึกไว้หรือยัง
if os.path.exists('mobilenetv2_model.keras'):
    model = load_model()
else:
    with st.spinner("ยังไม่มีโมเดลที่บันทึกไว้ กำลังฝึกโมเดลใหม่…"):
        model = train_model()

st.subheader("ตัวอย่างภาพที่ใช้ทดสอบ")
col1, col2, col3 = st.columns(3)
image1 = Image.open("paper.png")
image2 = Image.open("rock.png")
image3 = Image.open("scrissor.png")
with col1:
    st.image(image1, caption="ภาพกระดาษ", width=150)
with col2:
    st.image(image2, caption="ภาพค้อน", width=150)
with col3:
    st.image(image3, caption="ภาพกรรไกร", width=150)

st.subheader("อัปโหลดภาพของคุณเพื่อทดสอบ")

uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)

    # ลบพื้นหลัง
    with st.spinner("กำลังลบพื้นหลัง..."):
        image_no_bg = remove(image)

    # แปลงภาพและทำนาย
    image_arr = np.array(image.convert("RGB"))
    image_arr = tf.image.resize(image_arr, image_size) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)

    with st.spinner("กำลังทำนาย..."):
        prediction = model.predict(image_arr)

    predicted_class = np.argmax(prediction, axis=1)[0]
    probabilities = np.squeeze(prediction)

    labels = {0: "ค้อน (Rock)", 1: "กระดาษ (Paper)", 2: "กรรไกร (Scissors)"}

    left, mid, right = st.columns([1, 1, 1.3], gap="large")
    with left:
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)
    with mid:
        st.image(image_no_bg, caption="ภาพหลังจากลบพื้นหลัง (แสดงเปรียบเทียบ การทำนายใช้ภาพต้นฉบับ)", use_container_width=True)
    with right:
        st.caption("คำทำนาย")
        st.markdown(f'<p class="verdict">{labels[predicted_class]}</p>', unsafe_allow_html=True)
        for i, name in enumerate(info.features["label"].names):
            st.progress(float(probabilities[i]), text=f"{labels[i]} · {probabilities[i]*100:.1f}%")
        st.caption(f"Label Mapping: {', '.join(f'{i} = {n}' for i, n in enumerate(info.features['label'].names))}")
