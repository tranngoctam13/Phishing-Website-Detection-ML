import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Kéo hàm bóc tách URL từ file feature_extractor.py
from feature_extractor import extract_features

st.markdown("""
    <style>
    /* Ẩn menu mặc định của Streamlit và watermark */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Làm đẹp nút bấm Check */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 2px solid #ff3333;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------


# ==========================================
# 2. TỐI ƯU HÓA TẢI MÔ HÌNH (CACHE RESOURCE)
# ==========================================
@st.cache_resource  # Kỹ thuật cache giúp web không bị lag khi load lại
def load_model(model_name):
    if model_name == "Decision Tree":
        return joblib.load("dt_model.pkl")
    elif model_name == "Random Forest":
        return joblib.load("rf_model.pkl")
    else:
        return joblib.load("ada_model.pkl")


# ==========================================
# 3. SIDEBAR (THANH ĐIỀU HƯỚNG BÊN TRÁI)
# ==========================================
st.sidebar.title("⚙️ System Configuration")
st.sidebar.markdown("Please select your machine learning model:")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Decision Tree", "Random Forest", "AdaBoost"]
)

# Load model dựa trên lựa chọn và báo trạng thái
try:
    model = load_model(model_choice)
    st.sidebar.success(f"✅ {model_choice} model is loaded and ready!")
except Exception as e:
    st.sidebar.error("❌ Không tìm thấy file mô hình. Vui lòng kiểm tra lại!")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("📌 Tích hợp Live URL Extraction & Explainable AI (SHAP).")
st.sidebar.text("Developed by [Tran Ngoc Tam]")

# ==========================================
# 4. GIAO DIỆN CHÍNH (MAIN SCREEN)
# ==========================================
st.title("🛡️ Phishing Website Detection using ML")
st.markdown(
    "Hệ thống tự động phân tích mã nguồn HTML, trích xuất đặc trưng và sử dụng AI để phát hiện website lừa đảo theo thời gian thực.")
st.markdown("---")

# Tạo 2 cột để căn chỉnh UI đẹp hơn
col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input("🔗 Enter the URL", placeholder="https://youtube.com/")

with col2:
    st.write("")  # Căn chỉnh nút bấm cho ngang hàng với ô text
    st.write("")
    check_btn = st.button("🔍 Check!", type="primary", use_container_width=True)

if check_btn:
    if url_input == "":
        st.warning("Vui lòng nhập một URL hợp lệ!")
    else:
        with st.spinner(f"Đang phân tích '{url_input}'..."):

            # Cào dữ liệu và phân tích 31 đặc trưng
            features_df = extract_features(url_input)

            # Mô hình AI đưa ra dự đoán
            prediction = model.predict(features_df)[0]

            # Hiển thị kết quả
            st.markdown("### 🎯 Result:")
            if prediction == -1:
                st.error("🚨 **Attention! This web page is a potential PHISHING!**")
            else:
                st.success("✅ **This web page seems a legitimate!**")

            # ==========================================
            # 5. TRÍ TUỆ NHÂN TẠO CÓ THỂ GIẢI THÍCH (SHAP)
            # ==========================================
            st.markdown("---")
            st.markdown("### 📊 Explainable AI (SHAP Waterfall Plot)")

            try:
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

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_exp, show=False)
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.info(
                    "⚠️ Thuật toán AdaBoost chưa được tối ưu để vẽ biểu đồ SHAP. Vui lòng chọn Random Forest hoặc Decision Tree ở thanh menu bên trái.")