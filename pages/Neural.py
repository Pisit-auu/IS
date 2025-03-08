import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


st.title("Neural Network")
st.subheader("การเตรียมข้อมูล")
st.write("-  Dataset rock_paper_scissors จาก TensorFlow และดึงมาใช้ โดยโหลดข้อมูลมาจาก tensorflow")
st.write("-      dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True) ")
st.subheader("ตัวอย่างDataset ")
def load_dataset():
    dataset_name = "rock_paper_scissors"
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset, info

dataset, info = load_dataset()
train_data, test_data = dataset["train"], dataset["test"]

sample_data = train_data.take(10)

cols = st.columns(5) 
for row_idx in range(2):  
    cols = st.columns(5)
    for col_idx, (image, label) in enumerate(sample_data.skip(row_idx * 5).take(5)):  
        image_np = image.numpy() 
        label_np = label.numpy() 

        with cols[col_idx]:
            st.image(image_np, caption=f"Label: {label_np}", width=150)


st.write("""ข้อมูลนี้เป็นภาพมือที่เล่นเกมหิน กระดาษ กรรไกร มีรูปภาพทั้งหมด 2,892 รูปภาพ
-  ชุดทดสอบ (test): 372 รูปภาพ
- ชุดฝึก (train): 2,520 รูปภาพ
- ในข้อมูลจะมีภาพของ 3 ประภทคือ มือที่ทำรูป ค้อน กระ ดาษ กรรไกร
"""  
)
st.write("""ข้อมูลจาก https://www.tensorflow.org/datasets/catalog/rock_paper_scissors""" )



st.subheader("""Algorithm ที่ใช้
-  Convolutional Neural Network (CNN) และ MobileNetV2 """) 
st.write(""" ใช้ Convolutional Neural Network (CNN)  เพราะว่าจัดการเกี่ยวกับภาพได้ดี
- Convolutional Layer ช่วยค้นหาลักษณะต่างๆ
- ReLU Layer จะเปลี่ยนค่าลบเป็น 0 และค่าบวกจะยังคงเหมือนเดิม
- Pooling Layer ช่วยให้โมเดลมีความสามารถในการจับลักษณะภาพที่สำคัญ
- Fully Connected Layer ช่วยในการตัดสินใจสุดท้ายในการจำแนกภาพ
- Softmax Layer แปลงผลลัพธ์ที่ได้จาก Fully Connected Layer ให้เป็นค่าความน่าจะเป็น  
         """ )
st.write(""" โครงสร้างของ CNN โดยทั่วไป
- Input Layer: ข้อมูลภาพ
- Convolutional Layer: ค้นหาลักษณะเฉพาะของภาพ
- ReLU Activation: เพิ่มความไม่เชิงเส้น
- Pooling Layer: ลดขนาดภาพ
- Flatten: เปลี่ยนข้อมูลภาพให้เป็นเวกเตอร์ 1 มิติ
- Fully Connected Layer: ใช้ในการตัดสินใจทำนาย
- Output Layer: ผลลัพธ์การทำนาย เช่น การจำแนกประเภท
         """ )
st.write("""และใช้ MobileNetV2 เป็นโมเดลของ CNN ที่ใช้ทรัพยากรน้อย แต่ยังจำแนกภาพได้ดี""" )
st.subheader("การทำนาย")
st.write("ทำนายว่าภาพที่ส่งไปเป็น ค้อน กระดาษหรือ กรรไกร ")

st.title("ขั้นตอนพัฒนา Convolutional Neural Network โดยใช้ MobileNetV2 ")

st.subheader("""  แยกชุดข้อมูลเป็น train และ test
      dataset, info = load_dataset()
        train_data, test_data = dataset["train"], dataset["test"]
             """ )


st.write("""กำหนด batch_size และ image_size
- batch_size = 32
- image_size = (128, 128)""" )


st.subheader("""  ปรับภาพค่าพิกเซลในช่วง 0-255 ไปเป็น float32 และ /255 เพื่อให้ภาพอยู่ในช่วง 0-1
      def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0 
        image = tf.image.resize(image, image_size)  
        return image, label""" )

st.subheader("""  เพิ่มข้อมูลด้วยการหมุนภาพซ้ายขวา ปรับความสว่าง ปรับความชัด และปรับสี เพื่อให้ทำนายได้ดีขึ้น
      def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.02)
        return image, label
             """ )

st.subheader("""  เตรียมข้อมูลสำหรับการฝึก
      train_data = (
        train_data
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
            )
             """ )
st.write("""
- Map: ใช้ map() ในการแปลงข้อมูลทั้งชุด 
- Shuffle: ใช้ shuffle() สุ่มลำดับของข้อมูล
- Batch: ใช้ batch() แบ่งข้อมูลเป็นชุดย่อยๆ
- Prefetch: ใช้ prefetch()  ช่วยให้ TensorFlow เตรียมข้อมูลล่วงหน้าในระหว่างที่โมเดลกำลังฝึกฝน
         AUTOTUNE ช่วยให้ TensorFlow สามารถปรับจำนวนของเธรด""" )


st.subheader("""  เตรียมชุดข้อมูลสำหรับการทดสอบ
      test_data = (
        test_data
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
            )
             """ )
st.write("""ใช้ preprocess และแบ่งข้อมูลเป็น batch พร้อม prefetch)""" )

st.subheader("""  การโหลดและฝึกโมเดล MobileNetV2
      @st.cache_resource
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

            return model
             """ )
st.write("""MobileNetV2 เป็นฐานในการสร้างโมเดล
- include_top=False หมายถึงไม่ใช้เลเยอร์ที่ฝึกมาแล้วจาก ImageNet
- ใช้ GlobalAveragePooling2D เพื่อแปลงผลลัพธ์ให้เป็นขนาดเล็กลงและเชื่อมต่อกับเลเยอร์ Dense สำหรับการจำแนก 3 คลาส
- โมเดลนี้ฝึกด้วย Adam optimizer เป็นหนึ่งในอัลกอริธึมที่ใช้สำหรับการปรับค่าพารามิเตอร์ของโมเดล 
- และ Sparse Categorical Crossentropy loss  ใช้สำหรับปัญหาการจำแนกประเภทที่มี Label เป็นตัวเลข (Integer) แทน One-Hot Encoding
- ใช้ EarlyStopping เพื่อหยุดการฝึกหากไม่เห็นการปรับปรุงใน validation loss
- ใช้ class_weights เพื่อ เน้นที่กระดาษกับกรรไกร
""" )

st.subheader("""  การทำนายภาพที่อัปโหลด
        uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["png", "jpg", "jpeg"])
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
            probabilities = np.squeeze(prediction)

            labels = {0: "✊ Rock", 1: "✋ Paper", 2: "✌️ Scissors"}
            st.write(f"คำทำนาย: **{labels[predicted_class]}**")
            st.write("Label Mapping:", info.features["label"].names)
            st.write("Prediction Probabilities:", probabilities)

             """ )
st.write("""ใช้ st.file_uploader เพื่ออัปโหลดไฟล์ภาพ
- ลบพื้นหลังจากภาพด้วยฟังก์ชัน remove จากไลบรารี rembg
- แปลงภาพเป็น RGB และปรับขนาดตามที่กำหนด
- ทำนายผลด้วย MobileNetV2 และแสดงผลลัพธ์ (คำทำนายและความน่าจะเป็นของแต่ละคลาส)
""" )
