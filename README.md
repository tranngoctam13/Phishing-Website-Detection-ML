# 🛡️ Hệ thống Phát hiện Website Phishing

Đồ án ứng dụng Machine Learning để phân loại và phát hiện các trang web lừa đảo (Phishing) so với các trang web hợp pháp (Legitimate).

## 🚀 Công nghệ sử dụng
*   **Ngôn ngữ:** Python
*   **Thuật toán Machine Learning:** Random Forest Classifier (Đạt độ chính xác 96.79%)
*   **Giao diện Web Demo:** Streamlit
*   **Thư viện xử lý dữ liệu:** Pandas, Scikit-learn

## ⚙️ Cấu trúc thư mục
*   `main.py`: Mã nguồn tiến hành chia tập dữ liệu, huấn luyện mô hình và in ra báo cáo đánh giá (Classification Report).
*   `app.py`: Mã nguồn xây dựng giao diện Web UI bằng Streamlit.
*   `phishing_model.pkl`: Mô hình AI đã được huấn luyện thành công và lưu lại.
*   `phishing_data.csv`: Bộ dữ liệu (Dataset) đã được trích xuất 31 đặc trưng.

## 💻 Cách chạy Demo
1. Cài đặt các thư viện cần thiết: `pip install pandas scikit-learn streamlit joblib`
2. Khởi chạy giao diện web: `streamlit run app.py`
