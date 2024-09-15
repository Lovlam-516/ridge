import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với dấu phân cách là dấu chấm phẩy
df = pd.read_csv('dulieu.csv', delimiter=';')

# Hiển thị 5 dòng đầu tiên của DataFrame
# print(df.head())

# Chia dữ liệu thành biến đầu vào và biến mục tiêu
X = df[['production_cost_usd']]  # Biến đầu vào
y = df['total_revenue_usd']  # Biến mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình hồi quy Ridge
ridge = Ridge(alpha=1.0)  # Tham số alpha có thể thay đổi để tinh chỉnh mô hình
ridge.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred = ridge.predict(X_test_scaled)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')


