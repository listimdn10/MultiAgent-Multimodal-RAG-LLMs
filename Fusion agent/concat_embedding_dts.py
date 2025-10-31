import pandas as pd

# Đọc 2 file
df1 = pd.read_excel("MLP-embedding-part1.xlsx")
df2 = pd.read_excel("MLP-embedding-part2.xlsx")

# Gộp lại theo chiều dọc (tức là nối thêm dòng)
df_total = pd.concat([df1, df2], ignore_index=True)

# Ghi ra file mới
df_total.to_excel("total-for-MLP.xlsx", index=False)

print("✅ Đã gộp thành công vào file total-for-MLP.xlsx")
