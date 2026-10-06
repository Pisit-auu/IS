import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from style import apply_style, lede

apply_style("Neural Network", ":material/neurology:")

st.title("Neural Network")
lede("จำแนกภาพมือ ค้อน กระดาษ กรรไกร ด้วย Convolutional Neural Network บน MobileNetV2")

st.header("การเตรียมข้อมูล")
st.markdown("Dataset rock_paper_scissors จาก TensorFlow และดึงมาใช้ โดยโหลดข้อมูลมาจาก tensorflow")
st.code("dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)", language="python")

@st.cache_resource
def load_dataset():
    dataset_name = "rock_paper_scissors"
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset, info

dataset, info = load_dataset()
train_data, test_data = dataset["train"], dataset["test"]
label_names = info.features["label"].names

c1, c2, c3 = st.columns(3)
c1.metric("รูปภาพทั้งหมด", "2,892")
c2.metric("ชุดฝึก (train)", "2,520")
c3.metric("ชุดทดสอบ (test)", "372")

st.subheader("ตัวอย่าง Dataset")
sample_data = train_data.take(10)

for row_idx in range(2):
    cols = st.columns(5)
    for col_idx, (image, label) in enumerate(sample_data.skip(row_idx * 5).take(5)):
        image_np = image.numpy()
        label_np = int(label.numpy())

        with cols[col_idx]:
            st.image(image_np, caption=f"Label {label_np}: {label_names[label_np]}", use_container_width=True)

st.markdown("""ข้อมูลนี้เป็นภาพมือที่เล่นเกมหิน กระดาษ กรรไกร มีรูปภาพทั้งหมด 2,892 รูปภาพ
- ชุดทดสอบ (test): 372 รูปภาพ
- ชุดฝึก (train): 2,520 รูปภาพ
- ในข้อมูลจะมีภาพของ 3 ประเภทคือ มือที่ทำรูป ค้อน กระดาษ กรรไกร
""")
st.caption("ข้อมูลจาก https://www.tensorflow.org/datasets/catalog/rock_paper_scissors")


st.header("Algorithm ที่ใช้")
st.markdown("**Convolutional Neural Network (CNN) และ MobileNetV2**")
a1, a2 = st.columns(2, gap="large")
with a1:
    st.markdown("""ใช้ Convolutional Neural Network (CNN) เพราะว่าจัดการเกี่ยวกับภาพได้ดี
- Convolutional Layer ช่วยค้นหาลักษณะต่างๆ
- ReLU Layer จะเปลี่ยนค่าลบเป็น 0 และค่าบวกจะยังคงเหมือนเดิม
- Pooling Layer ช่วยให้โมเดลมีความสามารถในการจับลักษณะภาพที่สำคัญ
- Fully Connected Layer ช่วยในการตัดสินใจสุดท้ายในการจำแนกภาพ
- Softmax Layer แปลงผลลัพธ์ที่ได้จาก Fully Connected Layer ให้เป็นค่าความน่าจะเป็น
""")
with a2:
    st.markdown("""โครงสร้างของ CNN โดยทั่วไป
- Input Layer: ข้อมูลภาพ
- Convolutional Layer: ค้นหาลักษณะเฉพาะของภาพ
- ReLU Activation: เพิ่มความไม่เชิงเส้น
- Pooling Layer: ลดขนาดภาพ
- Flatten: เปลี่ยนข้อมูลภาพให้เป็นเวกเตอร์ 1 มิติ
- Fully Connected Layer: ใช้ในการตัดสินใจทำนาย
- Output Layer: ผลลัพธ์การทำนาย เช่น การจำแนกประเภท
""")
st.markdown("และใช้ MobileNetV2 เป็นโมเดลของ CNN ที่ใช้ทรัพยากรน้อย แต่ยังจำแนกภาพได้ดี")
st.subheader("การทำนาย")
st.markdown("ทำนายว่าภาพที่ส่งไปเป็น ค้อน กระดาษ หรือ กรรไกร")

st.header("ขั้นตอนพัฒนา Convolutional Neural Network โดยใช้ MobileNetV2")

st.subheader("1. แยกชุดข้อมูลเป็น train และ test")
st.code("""dataset, info = load_dataset()
train_data, test_data = dataset["train"], dataset["test"]""", language="python")

st.subheader("2. กำหนด batch_size และ image_size")
st.code("""batch_size = 32
image_size = (128, 128)""", language="python")

st.subheader("3. ปรับค่าพิกเซลให้อยู่ในช่วง 0–1")
st.markdown("แปลงค่าพิกเซลในช่วง 0–255 เป็น float32 และหารด้วย 255")
st.code("""def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image, label""", language="python")

st.subheader("4. เพิ่มข้อมูล (Augmentation)")
st.markdown("หมุนภาพซ้ายขวา ปรับความสว่าง ปรับความชัด และปรับสี เพื่อให้ทำนายได้ดีขึ้น")
st.code("""def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    return image, label""", language="python")

st.subheader("5. เตรียมข้อมูลสำหรับการฝึก")
st.code("""train_data = (
    train_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)""", language="python")
st.markdown("""
- Map: ใช้ map() ในการแปลงข้อมูลทั้งชุด
- Shuffle: ใช้ shuffle() สุ่มลำดับของข้อมูล
- Batch: ใช้ batch() แบ่งข้อมูลเป็นชุดย่อยๆ
- Prefetch: ใช้ prefetch() ช่วยให้ TensorFlow เตรียมข้อมูลล่วงหน้าในระหว่างที่โมเดลกำลังฝึกฝน
  AUTOTUNE ช่วยให้ TensorFlow สามารถปรับจำนวนของเธรด""")

st.subheader("6. เตรียมชุดข้อมูลสำหรับการทดสอบ")
st.code("""test_data = (
    test_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)""", language="python")
st.markdown("ใช้ preprocess และแบ่งข้อมูลเป็น batch พร้อม prefetch")

st.subheader("7. การโหลดและฝึกโมเดล MobileNetV2")
st.code("""@st.cache_resource
def trainmodel():
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

    return model""", language="python")
st.markdown("""MobileNetV2 เป็นฐานในการสร้างโมเดล
- include_top=False หมายถึงตัดชั้นจำแนกประเภท (classification head) 1,000 คลาสของ ImageNet ออก แต่ยังใช้ชั้น convolution ที่ฝึกมาแล้ว (weights="imagenet") และตั้ง trainable=False เพื่อไม่ให้ค่าถูกเปลี่ยน
- ใช้ GlobalAveragePooling2D เพื่อแปลงผลลัพธ์ให้เป็นขนาดเล็กลงและเชื่อมต่อกับเลเยอร์ Dense สำหรับการจำแนก 3 คลาส
- โมเดลนี้ฝึกด้วย Adam optimizer เป็นหนึ่งในอัลกอริธึมที่ใช้สำหรับการปรับค่าพารามิเตอร์ของโมเดล
- และ Sparse Categorical Crossentropy loss ใช้สำหรับปัญหาการจำแนกประเภทที่มี Label เป็นตัวเลข (Integer) แทน One-Hot Encoding
- ใช้ EarlyStopping เพื่อหยุดการฝึกหากไม่เห็นการปรับปรุงใน validation loss
- ใช้ class_weights เพื่อ เน้นที่กระดาษกับกรรไกร
""")

st.subheader("8. การทำนายภาพที่อัปโหลด")
st.code("""uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", width=250)

    image_no_bg = remove(image)
    st.image(image_no_bg, caption="ภาพหลังจากลบพื้นหลัง", width=250)

    image = np.array(image.convert("RGB"))
    image = tf.image.resize(image, image_size) / 255.0
    image = np.expand_dims(image, axis=0)

    with st.spinner("กำลังทำนาย..."):
        prediction = model.predict(image)

    predicted_class = np.argmax(prediction, axis=1)[0]
    probabilities = np.squeeze(prediction)""", language="python")
st.markdown("""ใช้ st.file_uploader เพื่ออัปโหลดไฟล์ภาพ
- ลบพื้นหลังจากภาพด้วยฟังก์ชัน remove จากไลบรารี rembg เพื่อแสดงเปรียบเทียบ (การทำนายใช้ภาพต้นฉบับ)
- แปลงภาพเป็น RGB และปรับขนาดตามที่กำหนด
- ทำนายผลด้วย MobileNetV2 และแสดงผลลัพธ์ (คำทำนายและความน่าจะเป็นของแต่ละคลาส)
""")
st.page_link("pages/Demoneural.py", label="ลองอัปโหลดภาพในหน้า Demo Neural Network", icon=":material/arrow_forward:")
