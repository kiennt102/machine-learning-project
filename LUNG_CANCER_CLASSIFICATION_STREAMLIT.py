import streamlit as st
import pickle
import pandas as pd

# Load mô hình đã lưu
model_path = "LogisticRegression_model.pkl"
model = pickle.load(open(model_path, 'rb'))

# Tạo giao diện với Streamlit
st.title("Ứng dụng Dự đoán Phân loại")


type = st.selectbox("Phân loại theo", options=["Nhập thông tin để dự đoán phân loại", "Từ file csv"])
if type == "Từ file csv":
    st.write("Tải lên file CSV chứa bệnh nhân cần dự đoán.")

    # Upload file
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])

    if uploaded_file:
        try:
            # Đọc file CSV
            df = pd.read_csv(uploaded_file)

            # Kiểm tra các cột cần thiết
            required_features = model.feature_names_in_  # Hoặc danh sách các thuộc tính
            missing_features = [col for col in required_features if col not in df.columns]

            if missing_features:
                st.error(f"Các cột bị thiếu: {', '.join(missing_features)}")
            else:
                # Dự đoán
                X = df[required_features]
                predictions = model.predict(X)

                # Gắn kết quả vào DataFrame
                df['Prediction'] = predictions

                # Chuyển pred thành low, medium, high
                for index in range(len(df['Prediction'])):
                    if df['Prediction'][index] == 0:
                        df['Prediction'][index] = "Low"
                    elif df['Prediction'][index] == 1:
                        df['Prediction'][index] = "Medium"
                    elif df['Prediction'][index] == 2:
                        df['Prediction'][index] = "High"

                # Hiển thị kết quả
                st.write("Kết quả dự đoán:")
                st.dataframe(df)

                # Cho phép tải file kết quả
                output_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải file kết quả",
                    data=output_csv,
                    file_name="output_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")


if type == "Nhập thông tin để dự đoán phân loại":
    st.write("Nhập thông tin để dự đoán phân loại bệnh nhân")

    # Tạo các trường nhập liệu
    age = st.number_input("Tuổi", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Giới tính", options=["Nam", "Nữ"], index=0)
    air_pollution = st.slider("Ô nhiễm không khí (1-8)", min_value=1, max_value=8, value=5)
    alcohol_use = st.slider("Mức độ sử dụng rượu (1-8)", min_value=1, max_value=8, value=5)
    dust_allergy = st.slider("Dị ứng bụi (1-8)", min_value=1, max_value=8, value=5)
    occuPational_hazards = st.slider("Nguy hiểm nghề nghiệp (1-8)", min_value=1, max_value=8, value=5)
    genetic_risk = st.slider("Bệnh di truyền (1-7)", min_value=1, max_value=7, value=5)
    chronic_lung_disease = st.slider("Bệnh phổi mãn tính (1-7)", min_value=1, max_value=7, value=5)
    balanced_diet = st.slider("Chế độ ăn uống cân bằng (1-7)", min_value=1, max_value=7, value=5)
    obesity = st.slider("Béo phì (1-7)", min_value=1, max_value=7, value=5)
    smoking = st.slider("Hút thuốc (1-8)", min_value=1, max_value=8, value=5)
    passive_smoker = st.slider("Hút thuốc thụ động (1-8)", min_value=1, max_value=8, value=5)
    chest_pain = st.slider("Đau ngực (1-9)", min_value=1, max_value=9, value=5)
    coughing_of_blood = st.slider("Ho ra máu (1-9)", min_value=1, max_value=9, value=5)
    fatigue = st.slider("Mệt mỏi (1-9)", min_value=1, max_value=9, value=5)
    weight_loss = st.slider("Giảm cân (1-8)", min_value=1, max_value=8, value=5)
    shortness_of_breath = st.slider("Khó thở (1-8)", min_value=1, max_value=8, value=5)
    wheezing = st.slider("Khò khè (1-8)", min_value=1, max_value=8, value=5)
    swallowing_difficulty = st.slider("Khó nuốt (1-8)", min_value=1, max_value=8, value=5)
    clubbing_of_finger_nails = st.slider("Đầu ngón tay dùi trống (1-9)", min_value=1, max_value=9, value=5)
    frequent_cold = st.slider("Tần suất cảm lạnh (1-7)", min_value=1, max_value=7, value=5)
    dry_cough = st.slider("Ho khan (1-7)", min_value=1, max_value=7, value=5)
    snoring = st.slider("Ngủ ngáy (1-7)", min_value=1, max_value=7, value=5)


    # Chuyển đổi giới tính thành số
    gender_encoded = 1 if gender == "Nam" else 2

    # Tập hợp đầu vào
    human = [age, gender_encoded, air_pollution, alcohol_use, dust_allergy, occuPational_hazards,
            genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain,
            coughing_of_blood, fatigue, weight_loss,shortness_of_breath, wheezing, swallowing_difficulty,
            clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]

    # Dự đoán
    if st.button("Dự đoán"):
        prediction = model.predict([human])
        if prediction.item() == 0:
            label = "Low"
        elif prediction.item() == 1:
            label = "Medium"
        else:
            label = "High"
        st.success(f"Kết quả dự đoán: {label}")
