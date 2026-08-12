# 🛡️ Phishing Website Detection ML

Hệ thống Phát hiện Trang web Lừa đảo (Phishing) tự động sử dụng Machine Learning, kết hợp Trích xuất đặc trưng trực tiếp (Live Feature Extraction) và Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI).

## ✨ Tính năng nổi bật
- **🔍 Phân tích URL Trực tiếp (Live Scanner):** Tự động cào mã nguồn HTML và tra cứu thông tin tên miền (WHOIS) để trích xuất 31 đặc trưng an ninh mạng theo thời gian thực từ một URL bất kỳ.
- **🤖 Machine Learning:** Sử dụng thuật toán `Random Forest Classifier` đạt độ chính xác lên đến **97%**.
- **📊 Trí tuệ nhân tạo có thể giải thích (XAI):** Tích hợp thư viện `SHAP` vẽ biểu đồ thác nước (Waterfall plot) giải thích minh bạch lý do AI kết luận trang web là An toàn hay Lừa đảo.
- **💻 Giao diện Web Trực quan:** Xây dựng bằng `Streamlit` thân thiện, dễ sử dụng cho người dùng cuối.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Explainable AI:** SHAP, Matplotlib
- **Web Scraping & Network:** BeautifulSoup4, Requests, Python-whois
- **Giao diện (Frontend):** Streamlit

## 🚀 Hướng dẫn cài đặt và sử dụng (Local)

1. Clone repository này về máy:
   ```bash
   git clone [https://github.com/tranngoctam13/Phishing-Website-Detection-ML.git](https://github.com/tranngoctam13/Phishing-Website-Detection-ML.git)



 1.  Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
2. Chạy giao diện Web:
   streamlit run app.py
