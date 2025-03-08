import streamlit as st
import pandas as pd
import numpy as np
# สร้างโมเดล
st.title("Machine Learning")
st.subheader("การเตรียมข้อมูล")
st.write("- Dataset คือ Adult")
file_path = "pages/adult.data"

df = pd.read_csv(file_path, sep=",\s*", engine='python', na_values=["?"])

st.subheader("Dataset ดิบ")
st.dataframe(df, height=300)

st.write("""
- Donated on 4/30/1996
- ข้อมูลนี้เกี่ยวข้องกับการ ทำนายว่ารายได้ประจำปีของบุคคลจะเกิน 50,000 ดอลลาร์ต่อปีหรือไม่โดยอิงจากข้อมูลสำมะโนประชากร
- โดยแคมเปญการตลาดใช้การโทรศัพท์ติดต่อกับลูกค้า บ่อยครั้งต้องมีการติดต่อมากกว่าหนึ่งครั้งเพื่อประเมินว่าลูกค้าจะสมัครเงินฝากประจำของธนาคารหรือไม่ (ตอบว่า "yes" หรือ "no")
""" 
)
st.subheader("Features ของ Dataset")
st.write("""
- "age": อายุของบุคคล
- "workclass": สถานะการทำงาน 
- "fnlwgt": น้ำหนักทางสถิติของบุคคลในชุดข้อมูล
- "education": ระดับการศึกษาของบุคคล
- "education-num": จำนวนปีการศึกษาของบุคคล
- "marital-status": สถานะสมรส
- "occupation": อาชีพของบุคคล 	
- "relationship": ความสัมพันธ์ของบุคคล
- "race":เชื้อชาติของบุคคล
- "sex": เพศ
- "capital-gain": เงินได้จากการลงทุน
- "capital-loss": ขาดทุนจากการลงทุน
- "hours-per-week": ชั่วโมงการทำงานต่อสัปดาห์
- "native-country":ประเทศที่เกิด
- "income": รายได้
         """)


st.write("""ข้อมูลจาก https://archive.ics.uci.edu/dataset/2/adult""" )
st.subheader("""Algorithm ที่ใช้
- SVM (Support Vector Machine) เป็นอัลกอริธึมการเรียนรู้ของเครื่องที่ใช้หลักการของ Hyperplane เพื่อแยกประเภทของข้อมูลให้อยู่ในคลาสที่แตกต่างกัน โดยอาศัย Support Vectors ซึ่งเป็นจุดข้อมูลที่อยู่ใกล้ขอบเขตการแบ่งมากที่สุด
- KNN (K-Nearest Neighbors) เป็นอัลกอริธึมที่ใช้หลักการของ ระยะห่าง (Distance) ระหว่างจุดข้อมูลเพื่อจำแนกประเภท โดยใช้ข้อมูลรอบข้าง (Neighbors) เป็นเกณฑ์ในการตัดสิน""") 
st.subheader("การทำนาย")
st.write("กำหนดว่ารายได้ของบุคคลนั้นเกิน 50,000 ดอลลาร์ต่อปีหรือไม่ ")

st.title("ขั้นตอนการพัฒนา Support Vector Machine และ K-Nearest Neighbors")
st.subheader("""นำเข้าไฟล์
             columns = [
                 "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]
               df = pd.read_csv(file_path, sep=",\s*", engine='python', na_values=["?"], names=columns)
             """)
st.write("""-  ใช้ na_values แปลง "?" เป็นNaN""" )
st.subheader("จัดการ Missing Values")
st.write("""-  แทนค่า None ด้วย Unknown  df.fillna("Unknown", inplace=True)""" )
st.subheader("""  แปลงข้อมูล categorical เป็นตัวเลข ด้วย label_encoders
                  label_encoders = {}
                  for col in df.select_dtypes(include=["object"]).columns:
                     le = LabelEncoder()
                     df[col] = le.fit_transform(df[col].astype(str))
                     label_encoders[col] = le
             """)

st.subheader("""  แบ่งชุดข้อมูลเป็น Train/Test
                X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, 
                random_state=42, stratify=y  )
             """)


st.subheader("""  ปรับขนาดข้อมูล (Normalization) เพื่อให้ข้อมูลอยู่ในช่วงที่เหมาะสม
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
             """) 

st.subheader("""   เลือกโมเดลและตั้งค่าพารามิเตอร์
             """)
st.write("""svm_kernel = poly""" )
st.write("""knn_neighbors = 17 โดยหาจาก Cross-validatio""" )
st.subheader("""  Cross-calidatio
             k_values = list(range(1, 20, 2))  # ลองค่า k = 1, 3, 5, ..., 15
               best_k = k_values[0]
               best_score = 0

               # ใช้ Cross-validation หา k ที่ดีที่สุด
               for k in k_values:
               knn_temp = KNeighborsClassifier(n_neighbors=k)
               score = np.mean(cross_val_score(knn_temp, X_train_scaled, y_train, cv=5))
               
               if score > best_score:
                    best_score = score
                    best_k = k   (ได้เป็น 17)
             """)


st.subheader("""   เทรนโมเดล  SVM
             svm_model = SVC(kernel=svm_kernel)
        svm_model.fit(X_train_scaled, y_train)
         """)
st.subheader("""   เทรนโมเดล  KNN 
             knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
        knn_model.fit(X_train_scaled, y_train)
        
             """)



st.subheader("""  ทำนายผลลัพธ์ SVM
             y_pred_svm = svm_model.predict(X_test_scaled)
             """)

st.subheader("""  ทำนายผลลัพธ์ KNN
             y_pred_knn = knn_model.predict(X_test_scaled)
             """)

st.subheader("""  วัดผล ด้วย accuracy_score และ classification_report                     """)
st.write("""โดย accuracy_score 
- บอกเป็นเปอร์เซ็นต์ว่าโมเดลทำนายถูกต้องกี่ครั้ง ง่ายและตรงไปตรงมา แต่ อาจไม่เพียงพอ หากข้อมูลไม่สมดุล """ )
st.write("""classification_report
- Precision (ความแม่นยำของคลาส) → ทำนายว่าเป็น 1 แล้วถูกต้องกี่เปอร์เซ็นต์
- Recall (Sensitivity/Recall Score) → ความสามารถของโมเดลในการหาคลาส 1
- F1-score → ค่าเฉลี่ยระหว่าง Precision และ Recall
- Support → จำนวนตัวอย่างในแต่ละคลาส""" )

st.subheader("""  วัดผล SVM
             svm_acc = accuracy_score(y_test, y_pred_svm)
             svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
             """)
st.subheader("""  วัดผล KNN
                knn_acc = accuracy_score(y_test, y_pred_knn)
                knn_report = classification_report(y_test, y_pred_knn, output_dict=True)
             """)

st.subheader("""  แสดงผล KNN และ SVM
- st.subheader(" ผลลัพธ์ของโมเดล")
- st.write(f"**Accuracy SVM ({svm_kernel} kernel):** {svm_acc:.4f}")
- st.write(f"**Accuracy KNN ({knn_neighbors} neighbors):** {knn_acc:.4f}")
- st.subheader("📋 รายงานผล SVM")
- st.dataframe(pd.DataFrame(svm_report).transpose())
- st.subheader("📋 รายงานผล KNN")
- st.dataframe(pd.DataFrame(knn_report).transpose())

             """)