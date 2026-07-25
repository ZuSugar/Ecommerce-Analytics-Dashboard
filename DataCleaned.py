import pandas as pd
import numpy as np

# ==============================
# 1️⃣ LOAD DATA (ระบุ dtype ล่วงหน้า เพื่อลด RAM + เพิ่มความเร็ว)
# ==============================
dtype_map = {
    'Product Price': 'float32',
    'Quantity': 'float32',
    'Total Purchase Amount': 'float32',
    'Customer Age': 'float32',
    'Age': 'float32',
    'Returns': 'float32',
}

df = pd.read_csv("ecommerce_customer_data.csv", low_memory=False, dtype=dtype_map)
print("Original Shape:", df.shape)

# ==============================
# 2️⃣ DATA PROFILING
# ==============================
print("\nMissing Values:")
print(df.isnull().sum())

dup_count = df.duplicated().sum()
print("\nDuplicated Rows:", dup_count)

# ==============================
# 3️⃣ REMOVE DUPLICATES
# ==============================
if dup_count > 0:
    df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

# ==============================
# 4️⃣ FIX DATA TYPES
# ==============================
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')

# แปลง numeric columns ที่ยังไม่ถูก dtype (coerce errors ที่เหลือ)
numeric_cols = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'Age']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ==============================
# 5️⃣ HANDLE MISSING VALUES
# ==============================
df['Returns'] = df['Returns'].fillna(0)

df = df.dropna(subset=['Purchase Date', 'Product Price', 'Quantity', 'Total Purchase Amount'])

# ==============================
# 6️⃣ REMOVE INVALID DATA (ใช้ boolean mask รวมกันครั้งเดียว)
# ==============================
mask = (
    (df['Age'] >= 18) & (df['Age'] <= 100) &
    (df['Product Price'] >= 0) &
    (df['Quantity'] > 0)
)
df = df[mask]

# ==============================
# 7️⃣ CHECK DUPLICATE AGE COLUMNS
# ==============================
if 'Customer Age' in df.columns and 'Age' in df.columns:
    if (df['Customer Age'] == df['Age']).all():
        df = df.drop(columns=['Customer Age'])
        print("Dropped duplicate column: Customer Age")

# ==============================
# 8️⃣ DATA VALIDATION (vectorized — ไม่ใช้ loc loop)
# ==============================
calculated = df['Product Price'] * df['Quantity']
mismatch = ~np.isclose(df['Total Purchase Amount'], calculated, rtol=1e-3)
df.loc[mismatch, 'Total Purchase Amount'] = calculated[mismatch]

# ==============================
# 9️⃣ FEATURE ENGINEERING
# ==============================
df['Year'] = df['Purchase Date'].dt.year.astype('int16')
df['Month'] = df['Purchase Date'].dt.month.astype('int8')

bins = [0, 24, 34, 44, 54, 100]
labels = ['<25', '25-34', '35-44', '45-54', '55+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# ==============================
# 🔟 FINAL CHECK
# ==============================
print("\nFinal Shape:", df.shape)
print("\nFinal Missing Values:")
print(df.isnull().sum())

# ==============================
# 1️⃣1️⃣ EXPORT
# ==============================
output_file = "cleaned_ecommerce_data.xlsx"