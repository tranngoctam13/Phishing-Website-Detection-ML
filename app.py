import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Tải mô hình AI đã lưu
model = joblib.load('phishing_model.pkl')

# 2. Tải tập dữ liệu để lấy mẫu test
data = pd.read_csv("phishing_data.csv")
if 'Index' in data.columns:
    data = data.drop('Index', axis=1)

# Lấy danh sách tên các đặc trưng (trừ cột nhãn cuối cùng)
feature_names = data.columns[:-1]

# 3. Thiết kế giao diện
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")
st.title("🛡️ Hệ thống Phát hiện Website Phishing")
st.write("Sử dụng Machine Learning (Random Forest) kết hợp XAI (SHAP)")

st.markdown("---")
st.write(
    "💡 **Demo:** Do việc tự động trích xuất 31 đặc trưng từ 1 URL bất kỳ cần thời gian thu thập dữ liệu mạng. Ở bản demo này, chúng ta sẽ chọn ngẫu nhiên một website đã được trích xuất sẵn đặc trưng từ cơ sở dữ liệu để kiểm tra mô hình."
)

# 4. Chỗ để người dùng chọn dòng dữ liệu test
sample_id = st.number_input(f"Nhập ID của trang web muốn kiểm tra (từ 0 đến {len(data) - 1}):", min_value=0,
                            max_value=len(data) - 1, value=10)

if st.button("🔍 Phân tích ngay"):
    # Thay vì dùng .values.reshape(1, -1), ta dùng .iloc[[id]] để giữ nguyên định dạng DataFrame và tên cột.
    # Điều này giúp SHAP vẽ biểu đồ có nhãn rõ ràng.
    features_df = data.iloc[[sample_id], :-1]
    actual_label = data.iloc[sample_id, -1]  # Nhãn gốc trong file csv để đối chiếu

    # Cho AI dự đoán
    prediction = model.predict(features_df)[0]

    st.markdown("### Kết quả dự đoán:")

    # Hiển thị kết quả
    if prediction == -1:
        st.error("🚨 CẢNH BÁO: Đây là trang web LỪA ĐẢO (Phishing)!")
    else:
        st.success("✅ AN TOÀN: Trang web này hợp pháp (Legitimate).")

    st.write(
        f"*(Thông tin đối chiếu: Hệ thống gốc ghi nhận trang web này là {'Lừa đảo' if actual_label == -1 else 'An toàn'})*")

    # ==========================================
    # BẮT ĐẦU PHẦN TÍCH HỢP XAI (SHAP)
    # ==========================================
    st.markdown("---")
    st.markdown("### 📊 Trí tuệ nhân tạo có thể giải thích (Explainable AI)")

    # Chữ hiển thị tùy theo dự đoán là Lừa đảo hay An toàn
    result_text = 'Lừa đảo' if prediction == -1 else 'An toàn'
    st.write(
        f"Biểu đồ dưới đây giải thích chi tiết các yếu tố đã tác động khiến AI kết luận URL này là **{result_text}**:")

    with st.spinner("Đang tính toán các giá trị đóng góp của đặc trưng..."):
        # 1. Khởi tạo bộ giải thích cho Random Forest
        explainer = shap.TreeExplainer(model)

        # 2. Tính toán giá trị SHAP cho dòng dữ liệu đang xét
        shap_values = explainer.shap_values(features_df)

        # 3. Xử lý linh hoạt cấu trúc dữ liệu của thư viện SHAP để tránh lỗi IndexError
        if isinstance(shap_values, list):
            # Nếu SHAP trả về dạng list (dành cho mỗi class)
            class_index = list(model.classes_).index(prediction)
            vals = shap_values[class_index][0]
            base_val = explainer.expected_value[class_index]
        elif len(shap_values.shape) == 3:
            # Nếu SHAP trả về mảng 3 chiều (samples, features, classes)
            class_index = list(model.classes_).index(prediction)
            vals = shap_values[0, :, class_index]
            base_val = explainer.expected_value[class_index]
        else:
            # Nếu SHAP trả về mảng 2 chiều duy nhất
            vals = shap_values[0]
            if isinstance(explainer.expected_value, (list, tuple)):
                base_val = explainer.expected_value[-1]
            else:
                base_val = explainer.expected_value

        # 4. Tạo đối tượng Explanation cho biểu đồ Waterfall
        shap_exp = shap.Explanation(values=vals,
                                    base_values=base_val,
                                    data=features_df.iloc[0],
                                    feature_names=feature_names)

        # 5. Vẽ biểu đồ và kết xuất lên giao diện Streamlit
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig)
        plt.clf()
