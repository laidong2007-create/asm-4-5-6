import pandas as pd
import numpy as np
import re

# --- PHẦN 3: XỬ LÝ OUTLIERS & SKEW ---
print("===== PHẦN 3: OUTLIERS & SKEW =====")
# Giả sử ta có cột 'price' (giá nhà)
data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'price': [100, 120, 110, 5000, 105] # 5000 là outlier
})

# Phát hiện bằng IQR
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['price'] < lower_bound) | (data['price'] > upper_bound)]
print(f"Các bản ghi bị coi là Outlier:\n{outliers}")

# Chiến lược: Capping (giới hạn giá trị)
data['price_fixed'] = data['price'].clip(lower_bound, upper_bound)
print(f"Dữ liệu sau khi xử lý Outlier:\n{data[['price', 'price_fixed']]}")


# --- PHẦN 4: CHUẨN HÓA SỐ & BIẾN ĐỔI CATEGORICAL ---
print("\n===== PHẦN 4: CHUẨN HÓA & ENCODING =====")
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Scaling numerical (Min-Max)
scaler = MinMaxScaler()
data['price_scaled'] = scaler.fit_transform(data[['price_fixed']])

# 2. Label Encoding cho cột Categorical
data['area_type'] = ['Urban', 'Rural', 'Urban', 'Suburb', 'Rural']
le = LabelEncoder()
data['area_encoded'] = le.fit_transform(data['area_type'])

print(data[['price_fixed', 'price_scaled', 'area_type', 'area_encoded']])


# --- PHẦN 5: PHÁT HIỆN DUPLICATE DỰA TRÊN TEXT SIMILARITY ---
print("\n===== PHẦN 5: TEXT SIMILARITY (DUPLICATE) =====")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Mô tả nhà để so sánh
descriptions = [
    "Nhà đẹp mặt phố quận 1 sạch sẽ",
    "Nhà đẹp mặt phố quận 1 sạch sẽ", # Trùng lặp hoàn toàn
    "Nhà phố quận 1 rất đẹp và sạch", # Trùng lặp nội dung (khác từ ngữ)
    "Căn hộ chung cư giá rẻ"
]

# Sử dụng TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(descriptions)

# Tính Cosine Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Ma trận tương đồng (Cosine Similarity):")
print(similarity_matrix)

# Gợi ý merge (nếu tương đồng > 0.8)
for i in range(len(descriptions)):
    for j in range(i + 1, len(descriptions)):
        if similarity_matrix[i][j] > 0.8:
            print(f"Gợi ý MERGE: Bản ghi {i} và {j} vì độ tương đồng = {similarity_matrix[i][j]:.2f}")
