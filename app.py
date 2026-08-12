import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Kéo hàm bóc tách URL từ file vừa tạo vào đây
from feature_extractor import extract_features

# 1. Tải mô hình AI đã lưu
model = joblib.load('phishing_model.pkl')

# 2. Thiết kế giao diện
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")
st.title("🛡️ Hệ thống Phát hiện Website Phishing")
st.write("Sử dụng Machine Learning (Random Forest) kết hợp Phân tích trực tiếp (Live Extraction) và XAI (SHAP)")

st.markdown("---")
st.write(
    "💡 **Hướng dẫn:** Hãy dán một đường link (URL) bất kỳ vào ô bên dưới. Hệ thống sẽ tự động cào dữ liệu, phân tích mã nguồn và đưa ra cảnh báo.")

# 3. Tạo ô nhập URL thay vì nhập ID
url_input = st.text_input("🔗 Nhập URL cần kiểm tra (Ví dụ: https://youtube.com/):")

if st.button("🔍 Quét và Phân tích ngay"):
    if url_input == "":
        st.warning("Vui lòng nhập một URL hợp lệ để kiểm tra!")
    else:
        with st.spinner(
                f"Đang bóc tách đặc trưng và phân tích mã nguồn từ '{url_input}'... (Quá trình này có thể mất vài giây)"):

            # --- GỌI HÀM BÓC TÁCH ĐẶC TRƯNG ---
            features_df = extract_features(url_input)

            # --- CHO AI DỰ ĐOÁN ---
            prediction = model.predict(features_df)[0]

            st.markdown("### Kết quả dự đoán:")
            if prediction == -1:
                st.error("🚨 CẢNH BÁO: Đây là trang web LỪA ĐẢO (Phishing) hoặc Rất Đáng Ngờ!")
            else:
                st.success("✅ AN TOÀN: Trang web này có dấu hiệu hợp pháp (Legitimate).")

            # ==========================================
            # BẮT ĐẦU PHẦN TÍCH HỢP XAI (SHAP)
            # ==========================================
            st.markdown("---")
            st.markdown("### 📊 Trí tuệ nhân tạo có thể giải thích (Explainable AI)")

            result_text = 'Lừa đảo' if prediction == -1 else 'An toàn'
            st.write(
                f"Biểu đồ dưới đây giải thích các yếu tố trên website `{url_input}` đã tác động khiến AI kết luận đây là web **{result_text}**:")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)

            if isinstance(shap_values, list):
                class_index = list(model.classes_).index(prediction)
                vals = shap_values[class_index][0]
                base_val = explainer.expected_value[class_index]
            elif len(shap_values.shape) == 3:
                class_index = list(model.classes_).index(prediction)
                vals = shap_values[0, :, class_index]
                base_val = explainer.expected_value[class_index]
            else:
                vals = shap_values[0]
                if isinstance(explainer.expected_value, (list, tuple)):
                    base_val = explainer.expected_value[-1]
                else:
                    base_val = explainer.expected_value

            shap_exp = shap.Explanation(values=vals,
                                        base_values=base_val,
                                        data=features_df.iloc[0],
                                        feature_names=features_df.columns)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig)
            plt.clf()