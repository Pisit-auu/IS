import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from style import apply_style, lede

apply_style("Demo SVM & KNN", ":material/model_training:")

st.title("Demo SVM & KNN")
lede("เทรน SVM และ KNN บนชุดข้อมูล Adult แล้วลองสุ่มข้อมูลบุคคลเพื่อดูว่าแต่ละโมเดลทำนายรายได้อย่างไร")

# โหลดข้อมูล
file_path = "pages/adult.data"
# ใช้ na_values แปลง "?" เป็นNaN
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
@st.cache_data
def load_data():
    return pd.read_csv(file_path, sep=r",\s*", engine='python', na_values=["?"], names=columns)

df = load_data()
with st.expander("Features ของ Dataset"):
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


#ตัวอย่างข้อมูล 
st.subheader("Dataset ดิบ")
st.dataframe(df, height=300)

# แปลง missing เป็น  "Unknown"
df.fillna("Unknown", inplace=True)

# แปลงข้อมูลเป็นตัวเลข
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

st.subheader("Dataset หลังแปลง")
st.dataframe(df, height=300)

# กำหนด x y
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ปรับขนาดข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_kernel = "poly"
knn_neighbors = 17

# เทรนครั้งเดียวแล้ว cache ไว้ ไม่ต้องเทรนใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def train_models(X_train_scaled, y_train, X_test_scaled):
    #train SVM
    svm_model = SVC(kernel=svm_kernel)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)

    #train KNN
    knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    return svm_model, knn_model, y_pred_svm, y_pred_knn

with st.spinner("กำลังเทรน SVM และ KNN… (ครั้งแรกใช้เวลาประมาณ 20 วินาที)"):
    svm_model, knn_model, y_pred_svm, y_pred_knn = train_models(X_train_scaled, y_train, X_test_scaled)

# result
svm_acc = accuracy_score(y_test, y_pred_svm)
knn_acc = accuracy_score(y_test, y_pred_knn)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)

#  output
st.header("ผลลัพธ์ของโมเดล")
m1, m2 = st.columns(2)
m1.metric(f"Accuracy SVM ({svm_kernel})", f"{svm_acc:.4f}")
m2.metric(f"Accuracy KNN ({knn_neighbors} neighbors)", f"{knn_acc:.4f}")

r1, r2 = st.columns(2)
with r1:
    st.subheader("ผลลัพธ์ของ SVM")
    st.dataframe(pd.DataFrame(svm_report).transpose().style.format("{:.3f}"), use_container_width=True)
with r2:
    st.subheader("ผลลัพธ์ของ KNN")
    st.dataframe(pd.DataFrame(knn_report).transpose().style.format("{:.3f}"), use_container_width=True)


st.header("ทำนายจากข้อมูลที่สุ่ม")

#สร้างข้อมูลจากข้อมูลที่มี
if st.button("สุ่มข้อมูล", type="primary", icon=":material/casino:"):
    random_data = []
    for col in X.columns:
        if col in label_encoders:  
            random_value = np.random.choice(df[col].unique()) 
        else: 
            min_val, max_val = df[col].min(), df[col].max()
            random_value = np.random.uniform(min_val, max_val) 
        random_data.append(random_value)


    random_df = pd.DataFrame([random_data], columns=X.columns)
    random_data_scaled = scaler.transform(random_df)

    # ใช้โมเดลทำนายผล
    pred_svm = svm_model.predict(random_data_scaled)
    pred_knn = knn_model.predict(random_data_scaled)

    # แปลงค่าที่ทำนายกลับเป็นข้อความ
    income_classes = label_encoders[df.columns[-1]].inverse_transform([pred_svm[0], pred_knn[0]])

    # แสดงผลลัพธ์
    st.write("**ข้อมูลที่ใช้ทำนาย**")
    st.dataframe(random_df, hide_index=True)
    p1, p2 = st.columns(2)
    p1.metric("SVM คาดการณ์ว่า", income_classes[0])
    p2.metric("KNN คาดการณ์ว่า", income_classes[1])
else:
    st.caption("กดปุ่มเพื่อสุ่มข้อมูลหนึ่งแถวจากค่าที่มีในชุดข้อมูล แล้วดูผลทำนายของทั้งสองโมเดล")