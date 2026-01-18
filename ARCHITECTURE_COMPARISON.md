# So Sánh Kiến Trúc: Single-Agent vs Multi-Agent

## 📊 Bảng So Sánh Tổng Quan

| **Tiêu Chí** | **Single-Agent (Monolithic)** | **Multi-Agent (CrewAI)** |
|--------------|-------------------------------|--------------------------|
| **Số lượng Agent** | 1 agent duy nhất | 5 agents chuyên biệt |
| **Số lượng Tool** | 1 tool monolithic | 5 tools chuyên môn hóa |
| **Tổng số Stages** | 7 stages trong 1 tool | 7 stages phân bổ cho agents |
| **Kiến trúc** | Monolithic (khối nguyên) | Modular (mô-đun hóa) |
| **Deployment** | 1 file output | 4-5 files output trung gian |

---

## 🔍 So Sánh Chi Tiết Từng Stage

| **Stage** | **Single-Agent** | **Multi-Agent** | **Ưu điểm Multi-Agent** |
|-----------|------------------|-----------------|------------------------|
| **1. Read Code** | Đọc trong MonolithicTool | Tự động trong pipeline | Tách biệt logic đọc file |
| **2. Semantic Analysis** | Gemini trong tool chính | **RAG Agent** với Gemini | Agent chuyên xử lý semantics |
| **3. RAG Retrieval** | ChromaDB + Neo4j trong tool | **RAG Agent** với ChromaDB + Neo4j + Unsloth | Có thể dùng Unsloth để reasoning |
| **4. Embedding** | Encode trực tiếp | **Embedding Agent** độc lập | Dễ swap model embeddings |
| **5. Fusion** | Load model trong tool | **Fusion Agent** với transformer | Chuyên biệt cho ML prediction |
| **6. Explanation** | Ollama trong tool | **Explainer Agent** với Ollama | Tách biệt interpretation layer |
| **7. Consensus** | Validate trong tool | **Consensus Agent** với KB retrieval | Independent validation layer |

---

## 🏗️ Kiến Trúc Code

### Single-Agent Architecture
```
MonolithicAuditTool
├── Stage 1: Read Code
├── Stage 2: Semantic (Gemini)
├── Stage 3: RAG (ChromaDB + Neo4j)
├── Stage 4: Embeddings
├── Stage 5: Fusion Model
├── Stage 6: Explanation (Ollama)
└── Stage 7: Consensus

1 Agent → 1 Tool → 7 Stages → 1 Output
```

### Multi-Agent Architecture
```
RAG Agent
├── Semantic Analysis (Gemini)
├── RAG Retrieval (ChromaDB + Neo4j)
└── Reasoning (Unsloth)
    ↓ rag_output.json

Embedding Agent
├── Read RAG output
├── Generate Code/Semantic/CFG embeddings
└── Output embeddings
    ↓ parser_output.json

Fusion Agent
├── Load embeddings
├── Fusion Transformer prediction
└── Find vulnerable lines
    ↓ fusion_output_agent.json

Explainer Agent
├── Read Fusion output
├── Generate explanation (Ollama)
└── Root cause + Solution
    ↓ explainer_output.json

Consensus Agent
├── Compare RAG + Explainer
├── Validate with KB
└── Final consensus
    ↓ consensus_output.json

5 Agents → 5 Tools → 7 Stages → 5 Outputs
```

---

## 🔧 So Sánh Kỹ Thuật

| **Khía Cạnh** | **Single-Agent** | **Multi-Agent** |
|---------------|------------------|-----------------|
| **Models sử dụng** | Gemini, Ollama, Transformer | Gemini, Unsloth, Ollama, Transformer |
| **Vector Databases** | ChromaDB + Neo4j | ChromaDB + Neo4j |
| **Error Handling** | 1 lỗi → toàn bộ fail | 1 agent fail → các agent khác vẫn chạy |
| **Retry Logic** | Không có | Có (Consensus có thể retry) |
| **Intermediate Results** | Không lưu | Lưu từng stage (JSON files) |
| **Debugging** | Khó (tất cả trong 1 tool) | Dễ (kiểm tra từng agent output) |
| **Testing** | Phải test toàn bộ | Test từng agent riêng biệt |

---

## 💡 Ưu/Nhược Điểm

### Single-Agent (Monolithic)

#### ✅ Ưu điểm:
- Đơn giản, dễ hiểu flow
- Chỉ 1 file output
- Ít overhead khi giao tiếp
- Phù hợp cho demo nhanh

#### ❌ Nhược điểm:
- **Khó maintain**: Sửa 1 stage phải hiểu toàn bộ
- **Không modular**: Không thể tái sử dụng từng phần
- **Single point of failure**: 1 lỗi → tất cả dừng
- **Khó scale**: Thêm tính năng = sửa tool lớn
- **Không tối ưu model**: Dùng cùng LLM cho mọi task
- **Khó debug**: Lỗi ở đâu không rõ ràng
- **Không có retry**: Fail là fail

### Multi-Agent (CrewAI)

#### ✅ Ưu điểm:
- **Modular**: Mỗi agent có trách nhiệm rõ ràng
- **Maintainable**: Sửa 1 agent không ảnh hưởng các agent khác
- **Scalable**: Thêm agent mới dễ dàng
- **Model specialization**: Mỗi agent dùng model tối ưu
  - RAG: Gemini (semantic) + Unsloth (reasoning)
  - Fusion: Transformer (ML prediction)
  - Explainer: Ollama (natural language)
- **Error isolation**: 1 agent fail không ảnh hưởng toàn bộ
- **Debuggable**: Kiểm tra output từng agent
- **Testable**: Test unit cho từng agent
- **Transparent**: Lưu kết quả trung gian
- **Retry logic**: Consensus có thể yêu cầu làm lại

#### ❌ Nhược điểm:
- Phức tạp hơn về cấu trúc
- Nhiều file output (có thể khó quản lý)
- Overhead khi agents giao tiếp
- Cần hiểu CrewAI framework

---

## 📈 So Sánh Hiệu Suất (Dự Kiến)

| **Metric** | **Single-Agent** | **Multi-Agent** |
|------------|------------------|-----------------|
| **Execution Time** | Nhanh hơn (~5-10%) | Chậm hơn chút do overhead |
| **Memory Usage** | Thấp hơn | Cao hơn (nhiều agents) |
| **Accuracy** | Trung bình | Cao hơn (consensus validation) |
| **Reliability** | Thấp (single point of failure) | Cao (isolated failures) |
| **Maintainability** | Thấp (monolithic) | Cao (modular) |
| **Scalability** | Thấp | Cao |

---

## 🎯 Khi Nào Dùng Gì?

### Dùng Single-Agent khi:
- ✅ Demo nhanh, prototype
- ✅ Dự án nhỏ, không cần mở rộng
- ✅ Đội ngũ nhỏ, không cần maintain lâu dài
- ✅ Yêu cầu performance tối đa

### Dùng Multi-Agent khi:
- ✅ Dự án production, lâu dài
- ✅ Cần maintain và mở rộng
- ✅ Đội ngũ lớn, nhiều người cùng làm
- ✅ Cần độ tin cậy cao
- ✅ Cần tối ưu từng bước với model khác nhau
- ✅ Cần debug và test từng phần

---

## 💻 So Sánh Code Complexity

### Single-Agent
```python
# 1 file, ~400 dòng
# Tất cả logic trong 1 class MonolithicAuditTool

class MonolithicAuditTool:
    def _run(self):
        # Stage 1-7 tất cả ở đây
        # 300+ dòng code trong 1 hàm
        pass
```

### Multi-Agent
```python
# 5+ files, mỗi file ~100-200 dòng
# Mỗi agent có file riêng, dễ đọc

# rag_agent.py
class RAGRetrieveTool:
    def _run(self): # ~50 dòng
        pass

# embedding_agent.py  
class EmbeddingGeneratorTool:
    def _run(self): # ~80 dòng
        pass

# agent_fusion.py
class FusionPredictorTool:
    def _run(self): # ~100 dòng
        pass

# explainer.py
class ExplainerTool:
    def _run(self): # ~60 dòng
        pass

# consensus_agent.py
class ConsensusTool:
    def _run(self): # ~100 dòng
        pass
```

**Kết luận**: Multi-agent dễ đọc, dễ maintain hơn nhiều!

---

## 🔄 So Sánh Data Flow

### Single-Agent Data Flow
```
Input (sample.sol)
    ↓
[MonolithicAuditTool]
│ ├─ Code Reading
│ ├─ Semantic Analysis  
│ ├─ RAG Retrieval
│ ├─ Embedding Generation
│ ├─ Fusion Prediction
│ ├─ Explanation
│ └─ Consensus
    ↓
single_agent_output.json
```

### Multi-Agent Data Flow
```
Input (sample.sol)
    ↓
[RAG Agent] → rag_output.json
    ↓
[Embedding Agent] → parser_output.json
    ↓
[Fusion Agent] → fusion_output_agent.json
    ↓
[Explainer Agent] → explainer_output.json
    ↓
[Consensus Agent] → consensus_output.json
```

**Lợi ích Multi-Agent**: Có thể kiểm tra kết quả từng bước!

---

## 📝 Kết Luận

### Về mặt Kỹ Thuật:
Multi-agent architecture **vượt trội** về:
- ✅ Maintainability (dễ maintain)
- ✅ Scalability (dễ mở rộng)
- ✅ Reliability (độ tin cậy)
- ✅ Testability (dễ test)
- ✅ Model Specialization (tối ưu từng task)

### Về mặt Học Thuật (Thesis):
Multi-agent là **lựa chọn đúng đắn** vì:
1. **Separation of Concerns**: Mỗi agent có trách nhiệm rõ ràng
2. **Model Diversity**: Sử dụng đúng model cho đúng task
3. **Fault Tolerance**: Hệ thống robust hơn
4. **Industry Standard**: Phù hợp với xu hướng microservices
5. **Research Value**: Thể hiện hiểu biết sâu về software architecture

### Khuyến Nghị:
Sử dụng **Multi-Agent CrewAI** cho production system, chỉ dùng **Single-Agent** làm baseline để so sánh và chứng minh lợi ích của kiến trúc đa tác nhân.

---

## 📚 Tài Liệu Tham Khảo

- CrewAI Documentation: https://docs.crewai.com/
- Multi-Agent Systems Theory
- Microservices Architecture Patterns
- Software Design Principles (SOLID, Separation of Concerns)
