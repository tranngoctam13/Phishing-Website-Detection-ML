import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Tải dữ liệu
data = pd.read_csv("phishing_data.csv")
print("Dữ liệu đã được tải thành công!")

# 2. Chuẩn bị dữ liệu
# Loại bỏ cột 'index' hoặc 'Index' vì nó không mang ý nghĩa dự đoán
if 'index' in data.columns:
    data = data.drop('index', axis=1)
if 'Index' in data.columns:
    data = data.drop('Index', axis=1)

# Tách đặc trưng (X) và nhãn (y)
X = data.iloc[:, :-1]  # Lấy tất cả các cột trừ cột cuối
y = data.iloc[:, -1]  # Lấy cột cuối cùng làm nhãn

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN 3 MÔ HÌNH
# ==========================================
# Khai báo 3 thuật toán
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# Đặt tên file tương ứng cho từng thuật toán
file_names = {
    "Random Forest": "rf_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "AdaBoost": "ada_model.pkl"
}

# Vòng lặp tự động huấn luyện, đánh giá và lưu từng mô hình
for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"🔄 Đang huấn luyện mô hình: {name}...")

    # Huấn luyện
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Độ chính xác (Accuracy): {accuracy * 100:.2f}%")

    # Lưu ra file
    filename = file_names[name]
    joblib.dump(model, filename)
    print(f"💾 Đã lưu mô hình thành file: {filename}")

print(f"\n{'=' * 40}")
print("🎉 HOÀN TẤT! Đã huấn luyện và lưu thành công 3 mô hình.")