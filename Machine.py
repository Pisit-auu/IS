import streamlit as st
import pandas as pd
import numpy as np
from style import apply_style, lede

apply_style("Machine Learning", ":material/insights:")

# สร้างโมเดล
st.title("Machine Learning")
lede("ทำนายว่ารายได้ต่อปีของบุคคลเกิน 50,000 ดอลลาร์หรือไม่ ด้วย SVM และ KNN จากชุดข้อมูล Adult")

st.header("การเตรียมข้อมูล")
file_path = "pages/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

@st.cache_data
def load_data():
    return pd.read_csv(file_path, sep=r",\s*", engine='python', na_values=["?"], names=columns)

df = load_data()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset", "Adult")
c2.metric("จำนวนแถว", f"{len(df):,}")
c3.metric("Features", f"{df.shape[1] - 1}")

st.subheader("Dataset ดิบ")
st.dataframe(df, height=300)

st.markdown("""
- Donated on 4/30/1996
- ข้อมูลนี้เกี่ยวข้องกับการ ทำนายว่ารายได้ประจำปีของบุคคลจะเกิน 50,000 ดอลลาร์ต่อปีหรือไม่โดยอิงจากข้อมูลสำมะโนประชากร
- ข้อมูลสกัดมาจากฐานข้อมูลสำมะโนประชากรของสหรัฐอเมริกาปี 1994 โดยคลาสเป้าหมาย (income) มี 2 ค่า คือ ">50K" และ "<=50K"
""")

st.subheader("Features ของ Dataset")
features = pd.DataFrame([
    ("age", "อายุของบุคคล"),
    ("workclass", "สถานะการทำงาน"),
    ("fnlwgt", "น้ำหนักทางสถิติของบุคคลในชุดข้อมูล"),
    ("education", "ระดับการศึกษาของบุคคล"),
    ("education-num", "จำนวนปีการศึกษาของบุคคล"),
    ("marital-status", "สถานะสมรส"),
    ("occupation", "อาชีพของบุคคล"),
    ("relationship", "ความสัมพันธ์ของบุคคล"),
    ("race", "เชื้อชาติของบุคคล"),
    ("sex", "เพศ"),
    ("capital-gain", "เงินได้จากการลงทุน"),
    ("capital-loss", "ขาดทุนจากการลงทุน"),
    ("hours-per-week", "ชั่วโมงการทำงานต่อสัปดาห์"),
    ("native-country", "ประเทศที่เกิด"),
    ("income", "รายได้"),
], columns=["Feature", "ความหมาย"])
st.dataframe(features, hide_index=True, use_container_width=True)

st.caption("ข้อมูลจาก https://archive.ics.uci.edu/dataset/2/adult")

st.header("Algorithm ที่ใช้")
a1, a2 = st.columns(2, gap="large")
with a1:
    st.subheader("SVM (Support Vector Machine)")
    st.markdown("อัลกอริธึมการเรียนรู้ของเครื่องที่ใช้หลักการของ Hyperplane เพื่อแยกประเภทของข้อมูลให้อยู่ในคลาสที่แตกต่างกัน "
                "โดยอาศัย Support Vectors ซึ่งเป็นจุดข้อมูลที่อยู่ใกล้ขอบเขตการแบ่งมากที่สุด")
with a2:
    st.subheader("KNN (K-Nearest Neighbors)")
    st.markdown("อัลกอริธึมที่ใช้หลักการของ ระยะห่าง (Distance) ระหว่างจุดข้อมูลเพื่อจำแนกประเภท "
                "โดยใช้ข้อมูลรอบข้าง (Neighbors) เป็นเกณฑ์ในการตัดสิน")

st.subheader("การทำนาย")
st.markdown("กำหนดว่ารายได้ของบุคคลนั้นเกิน 50,000 ดอลลาร์ต่อปีหรือไม่")

st.header("ขั้นตอนการพัฒนา Support Vector Machine และ K-Nearest Neighbors")

st.subheader("1. นำเข้าไฟล์")
st.code(r"""columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"]
df = pd.read_csv(file_path, sep=",\s*", engine='python', na_values=["?"], names=columns)""", language="python")
st.markdown('- ใช้ `na_values` แปลง "?" เป็น NaN')

st.subheader("2. จัดการ Missing Values")
st.markdown('แทนค่าที่หายไป (NaN) ด้วย "Unknown"')
st.code('df.fillna("Unknown", inplace=True)', language="python")

st.subheader("3. แปลงข้อมูล categorical เป็นตัวเลข ด้วย label_encoders")
st.code("""label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le""", language="python")

st.subheader("4. แบ่งชุดข้อมูลเป็น Train/Test")
st.code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)""", language="python")

st.subheader("5. ปรับสเกลข้อมูล (Standardization)")
st.markdown("ใช้ StandardScaler ปรับแต่ละ feature ให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 เพื่อไม่ให้ feature ที่มีค่าใหญ่ (เช่น fnlwgt) ครอบงำระยะทางใน SVM และ KNN")
st.code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)""", language="python")

st.subheader("6. เลือกโมเดลและตั้งค่าพารามิเตอร์")
st.markdown("""
- `svm_kernel = poly`
- `knn_neighbors = 17` โดยหาจาก Cross-validation
""")

st.subheader("7. Cross-validation")
st.code("""k_values = list(range(1, 20, 2))  # ลองค่า k = 1, 3, 5, ..., 19
best_k = k_values[0]
best_score = 0

# ใช้ Cross-validation หา k ที่ดีที่สุด
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    score = np.mean(cross_val_score(knn_temp, X_train_scaled, y_train, cv=5))

    if score > best_score:
        best_score = score
        best_k = k   # ได้เป็น 17""", language="python")

st.subheader("8. เทรนโมเดล")
t1, t2 = st.columns(2)
with t1:
    st.markdown("**SVM**")
    st.code("""svm_model = SVC(kernel=svm_kernel)
svm_model.fit(X_train_scaled, y_train)""", language="python")
with t2:
    st.markdown("**KNN**")
    st.code("""knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
knn_model.fit(X_train_scaled, y_train)""", language="python")

st.subheader("9. ทำนายผลลัพธ์")
st.code("""y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)""", language="python")

st.subheader("10. วัดผล ด้วย accuracy_score และ classification_report")
st.markdown("""**accuracy_score**
- บอกเป็นเปอร์เซ็นต์ว่าโมเดลทำนายถูกต้องกี่ครั้ง ง่ายและตรงไปตรงมา แต่ อาจไม่เพียงพอ หากข้อมูลไม่สมดุล

**classification_report**
- Precision (ความแม่นยำของคลาส) → ทำนายว่าเป็น 1 แล้วถูกต้องกี่เปอร์เซ็นต์
- Recall (Sensitivity/Recall Score) → ความสามารถของโมเดลในการหาคลาส 1
- F1-score → ค่าเฉลี่ยระหว่าง Precision และ Recall
- Support → จำนวนตัวอย่างในแต่ละคลาส""")
st.code("""svm_acc = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)

knn_acc = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)""", language="python")

st.subheader("11. แสดงผล KNN และ SVM")
st.code("""st.subheader("ผลลัพธ์ของโมเดล")
st.write(f"**Accuracy SVM ({svm_kernel} kernel):** {svm_acc:.4f}")
st.write(f"**Accuracy KNN ({knn_neighbors} neighbors):** {knn_acc:.4f}")
st.subheader("รายงานผล SVM")
st.dataframe(pd.DataFrame(svm_report).transpose())
st.subheader("รายงานผล KNN")
st.dataframe(pd.DataFrame(knn_report).transpose())""", language="python")
st.page_link("pages/DemoMachine.py", label="ลองใช้งานโมเดลจริงในหน้า Demo SVM & KNN", icon=":material/arrow_forward:")
