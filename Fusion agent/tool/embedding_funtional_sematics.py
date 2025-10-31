import pandas as pd
import json
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load file Excel
df = pd.read_excel("../solidity_code_with_labels-3k-partially.xlsx")

# 3. Khởi tạo danh sách chứa embedding
embedding_list = []

# 4. Duyệt từng dòng
for idx, row in df.iterrows():
    text = str(row.get("functional_semantics", "")).strip()

    try:
        if text:
            emb = embeddings_model.embed_query(text)
            embedding_list.append(json.dumps(emb))  # lưu dưới dạng chuỗi JSON
        else:
            embedding_list.append("")
    except Exception as e:
        print(f"❌ Lỗi dòng {idx+1}: {e}")
        embedding_list.append("")

# 5. Gắn vào cột mới và lưu file
df["minilm_embedding"] = embedding_list
df.to_excel("solidity_with_minilm_embeddings.xlsx", index=False)
print("✅ Đã lưu xong vào file: solidity_with_minilm_embeddings.xlsx")
