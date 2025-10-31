import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
import json

# Load mô hình CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()

# Load file Excel
df = pd.read_excel("../solidity_code_with_labels-3k-partially.xlsx")

# Tạo cột mới để chứa embedding
embeddings = []

for idx, row in df.iterrows():
    code = str(row["code"])  # Đảm bảo là string
    try:
        # Tokenize
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        
        # Inference không cần gradient
        with torch.no_grad():
            outputs = model(**inputs)

        # Lấy embedding từ token [CLS]
        emb = outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # [768]
        
        # Lưu dưới dạng chuỗi JSON để lưu vào Excel
        embeddings.append(json.dumps(emb))

    except Exception as e:
        print(f"❌ Lỗi dòng {idx+1}: {e}")
        embeddings.append("")  # Tránh lỗi, để trống nếu fail

# Gắn vào cột mới
df["codebert_embedding"] = embeddings

# Lưu lại vào Excel mới
df.to_excel("solidity_with_label_with_codebert-part2.xlsx", index=False)
print("✅ Đã tạo file mới với embedding.")
