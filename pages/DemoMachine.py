import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

st.title("Demo SVM & KNN ")

# โหลดข้อมูล
file_path = "pages/adult.data"
# ใช้ na_values แปลง "?" เป็นNaN
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
df = pd.read_csv(file_path, sep=",\s*", engine='python', na_values=["?"], names=columns)
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


#ตัวอย่างข้อมูล 
st.title("Dataset ดิบ")
st.dataframe(df, height=300)

# แปลง missing เป็น  "Unknown"
df.fillna("Unknown", inplace=True)

# แปลงข้อมูลเป็นตัวเลข
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

st.title("Dataset หลังแปลง")
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

#train SVM
svm_model = SVC(kernel="poly")
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

#train KNN
knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# result
svm_acc = accuracy_score(y_test, y_pred_svm)
knn_acc = accuracy_score(y_test, y_pred_knn)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)

#  output
st.subheader("ผลลัพธ์ของโมเดล")
st.write(f"**Accuracy SVM ({svm_kernel}):** {svm_acc:.4f}")
st.write(f"**Accuracy KNN ({knn_neighbors} neighbors):** {knn_acc:.4f}")

st.subheader("ผลลัพธ์ของ SVM")
st.dataframe(pd.DataFrame(svm_report).transpose())

st.subheader("ผลลัพธ์ของ KNN")
st.dataframe(pd.DataFrame(knn_report).transpose())


st.subheader("ทำนายจากข้อมูลที่สุ่ม")

#สร้างข้อมูลจากข้อมูลที่มี
if st.button("สุ่มข้อมูล"):
    random_data = []
    for col in X.columns:
        if col in label_encoders:  
            random_value = np.random.choice(df[col].unique()) 
        else: 
            min_val, max_val = df[col].min(), df[col].max()
            random_value = np.random.uniform(min_val, max_val) 
        random_data.append(random_value)


    random_data = np.array(random_data).reshape(1, -1)
    random_data_scaled = scaler.transform(random_data)

    # ใช้โมเดลทำนายผล
    pred_svm = svm_model.predict(random_data_scaled)
    pred_knn = knn_model.predict(random_data_scaled)

    # แปลงค่าที่ทำนายกลับเป็นข้อความ
    income_classes = label_encoders[df.columns[-1]].inverse_transform([pred_svm[0], pred_knn[0]])

    # แสดงผลลัพธ์
    st.write("**ข้อมูลที่ใช้ทำนาย**")
    random_df = pd.DataFrame([random_data[0]], columns=X.columns)
    st.dataframe(random_df)
    st.write(f"**SVM คาดการณ์ว่า:** {income_classes[0]}")
    st.write(f"**KNN คาดการณ์ว่า:** {income_classes[1]}")