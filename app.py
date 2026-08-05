import streamlit as st
import pandas as pd
import joblib

# 1. Tải mô hình AI đã lưu
model = joblib.load('phishing_model.pkl')

# 2. Tải tập dữ liệu để lấy mẫu test
data = pd.read_csv("phishing_data.csv")
if 'Index' in data.columns:
    data = data.drop('Index', axis=1)

# 3. Thiết kế giao diện
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")
st.title("🛡️ Hệ thống Phát hiện Website Phishing")
st.write("Sử dụng Machine Learning (Random Forest) - Độ chính xác 96.79%")

st.markdown("---")
st.write(
    "💡 **Demo:** Do việc tự động trích xuất 31 đặc trưng từ 1 URL bất kỳ cần thời gian thu thập dữ liệu mạng. Ở bản demo này, chúng ta sẽ chọn ngẫu nhiên một website đã được trích xuất sẵn đặc trưng từ cơ sở dữ liệu để kiểm tra mô hình.")

# 4. Chỗ để người dùng chọn dòng dữ liệu test
sample_id = st.number_input(f"Nhập ID của trang web muốn kiểm tra (từ 0 đến {len(data) - 1}):", min_value=0,
                            max_value=len(data) - 1, value=10)

if st.button("🔍 Phân tích ngay"):
    # Lấy dữ liệu của dòng tương ứng (Bỏ cột Result cuối cùng)
    features = data.iloc[sample_id, :-1].values.reshape(1, -1)
    actual_label = data.iloc[sample_id, -1]  # Nhãn gốc trong file csv để đối chiếu

    # Cho AI dự đoán
    prediction = model.predict(features)[0]

    st.markdown("### Kết quả dự đoán:")

    # Hiển thị kết quả
    if prediction == -1:
        st.error("🚨 CẢNH BÁO: Đây là trang web LỪA ĐẢO (Phishing)!")
    else:
        st.success("✅ AN TOÀN: Trang web này hợp pháp (Legitimate).")

    st.write(
        f"*(Thông tin đối chiếu: Hệ thống gốc ghi nhận trang web này là {'Lừa đảo' if actual_label == -1 else 'An toàn'})*")