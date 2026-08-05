import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Tải dữ liệu
data = pd.read_csv("phishing_data.csv")
print("Dữ liệu đã được tải thành công!")
print(data.head())

# 2. Chuẩn bị dữ liệu
# Loại bỏ cột 'Index' vì nó không mang ý nghĩa dự đoán
if 'Index' in data.columns:
    data = data.drop('Index', axis=1)

# Tách đặc trưng (X) và nhãn (y)
# Giả định cột cuối cùng là cột nhãn (Result, class, ...)
X = data.iloc[:, :-1]  # Lấy tất cả các cột trừ cột cuối
y = data.iloc[:, -1]   # Lấy cột cuối cùng làm nhãn

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình Random Forest
print("\nĐang huấn luyện mô hình...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác của mô hình: {accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))
import joblib

# Lưu mô hình lại thành file để dùng cho giao diện Web
joblib.dump(model, 'phishing_model.pkl')
print("Đã lưu mô hình thành file phishing_model.pkl")