import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

print("Bắt đầu tạo vector store...")

# --- 1. Tải Knowledge Base ---
try:
    with open("KB.md", "r", encoding="utf-8") as f:
        kb_content = f.read()
    print("✅ Đã đọc knowledge_base.md")
except Exception as e:
    print(f"❌ LỖI: Không thể đọc 'knowledge_base.md'. Bạn đã tạo file này chưa? {e}")
    exit()

# --- 2. Phân tách (Split) tài liệu ---
# Chúng ta dùng "---" làm dấu phân cách chính
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n---", "\n\n## ", "\n\n### ", "\n"],
    chunk_size=2000, # Tăng chunk size để cố gắng giữ trọn vẹn 1 lỗ hổng
    chunk_overlap=200
)
docs = text_splitter.create_documents([kb_content])
print(f"✅ Đã phân tách tài liệu thành {len(docs)} phần.")

# --- 3. Chọn mô hình Embedding (Cục bộ) ---
# Dùng mô hình nhẹ, phổ biến. Lần đầu chạy sẽ mất vài phút để tải về
print("Đang tải mô hình embeddings (có thể mất vài phút)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 4. Tạo và lưu ChromaDB ---
# Xóa DB cũ nếu có để tạo mới
db_path = "./chroma_db"
if os.path.exists(db_path):
    import shutil
    shutil.rmtree(db_path)

print(f"Đang tạo ChromaDB tại {db_path}...")
vector_store = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory=db_path
)

print("\n🎉 HOÀN THÀNH! Vector store đã được tạo và lưu.")