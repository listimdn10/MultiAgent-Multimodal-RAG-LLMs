# RQ4 — Ảnh hưởng của knowledge node và knowledge graph

RQ4 so sánh biểu diễn có cấu trúc tri thức (knowledge node / knowledge graph)
với biểu diễn phẳng hoặc không có tri thức. Thư mục này gồm:

| File | Vai trò |
|---|---|
| `modalities.ipynb` | Notebook thử nghiệm các modality: embedding văn bản và embedding đồ thị CFG |
| `eval_kb_vs_kg.py` | Script tính Accuracy / Precision / Recall / F1 để so sánh Knowledge Base (flat) với Knowledge Graph trên Neo4j |

Mã nguồn tạo knowledge node và truy xuất tri thức nằm ở ngoài thư mục này:
`embedding_agent.py`, `bi-modal/embedding_agent.py` (thêm knowledge node vào biểu diễn CFG),
`rag_agent.py`, `bi-modal/rag_agent.py`, `knowledge_base/` (truy xuất knowledge graph cho nhánh Bi-modal).

## 1. `modalities.ipynb`

Notebook có 2 thí nghiệm độc lập, mỗi thí nghiệm gồm một cell cài đặt (`pip install`) và một cell chạy.

### 1.1 So sánh embedding model cho văn bản (cell 0–2)

- **Dữ liệu:** `solidity_with_minilm_embeddings-part2.xlsx` (đọc CSV trước, lỗi thì đọc Excel).
  Đầu vào là cột `functional_semantics`, nhãn là cột `label` (mã hóa bằng `LabelEncoder`).
- **Embedding model:**
  `all-MiniLM-L6-v2` (baseline), `all-mpnet-base-v2`, `sentence-t5-base` (cần `sentencepiece`),
  `paraphrase-TinyBERT-L6-v2`.
- **Phân loại:** `MLPClassifier` với hidden layers (256, 128), ReLU, Adam, `max_iter=1000`,
  `early_stopping=True`, `random_state=42`.
- **Chia tập:** 80/20, `stratify=y`, `random_state=42`.
- **Đầu ra:** Accuracy từng model, biểu đồ cột ngang xếp từ cao xuống thấp.

### 1.2 So sánh mô hình GNN trên CFG của hợp đồng Solidity (cell 3–4)

Pipeline tạo đồ thị cho từng hợp đồng trong `solidity_code_with_labels-3k-full.xlsx` (cột `code`, `label`):

1. Đọc `pragma solidity` để chọn phiên bản, cài và chọn bằng `solc-select`.
2. Biên dịch `solc --bin-runtime` ra bytecode runtime. Hợp đồng không biên dịch được bị bỏ qua.
3. Chạy `EtherSolve.jar` để sinh CFG (`report.json`).
4. Chuyển CFG thành `torch_geometric.data.Data`:
   - mỗi node là một basic block; đặc trưng là 20 opcode đầu (padding bằng 0, cắt phần dư),
     id opcode lấy từ từ điển `opcode_vocab` (0 = `UNK`);
   - cạnh lấy từ `successors` của EtherSolve.
5. Mỗi mô hình dùng `Embedding` (hidden 64), trung bình theo chuỗi opcode, 2 lớp tích chập đồ thị,
   `global_mean_pool`, rồi `Linear` phân loại.

Bốn mô hình được so sánh: **GATv2** (4 head ở lớp 1), **GCN**, **GraphSAGE**, **GIN** (kèm BatchNorm).
Huấn luyện bằng Adam (lr 0.005), 20 epoch, batch 32, chia train/test 80/20 (`random_state=42`).
Kết quả là Accuracy từng mô hình, lưu biểu đồ vào `graph_model_comparison.png`.

**Yêu cầu và lưu ý khi chạy:**

- Cần `torch`, `torch-geometric`, `solc-select`, Java và `EtherSolve.jar` ở thư mục gốc repo
  (đường dẫn trong notebook là `EtherSolve.jar`, hãy chỉnh `ETHERSOLVE_JAR` nếu chạy ở nơi khác).
- Notebook gọi `cp`, `solc` bằng shell nên phù hợp Linux/Colab hơn Windows thuần.
- Notebook ghi file tạm `temp.sol`, `temp.evm`, `report.json` và thư mục `output/` ở thư mục chạy.
- Hai file dữ liệu `.xlsx` không nằm trong repo; cần tự cung cấp.
- Phần chia train/test của GNN không stratify và các mô hình khởi tạo trước khi đặt seed,
  nên kết quả có thể dao động giữa các lần chạy.

## 2. `eval_kb_vs_kg.py` — so sánh Knowledge Base và Knowledge Graph

Script đo Accuracy, Precision, Recall, F1 (macro và weighted, kèm chỉ số theo từng lớp) cho hai cấu hình
dùng chung một nguồn tri thức (`experiment/rq1_detection/knowledge-graph-relationship.csv`):

| Cấu hình | Cách truy xuất |
|---|---|
| **KB (flat)** | Mỗi dòng CSV là một tài liệu độc lập. Tính cosine giữa embedding mã nguồn truy vấn và embedding của các trường (`vulnerable code`, `Detail Behaviors`), lấy top-k. Không dùng quan hệ. |
| **KG (Neo4j)** | Truy vấn vector index (`vcode_index`, `behavior_index`, ...) rồi đi theo quan hệ `HAS_VCODE`, `HAS_PURPOSE`, `HAS_BEHAVIORS`, `EXPOSES` về nút `File`. Điểm của mọi nút seed được gộp theo `File`. |

Nhãn dự đoán là nhãn của `File` có tổng điểm cao nhất (mặc định là tên file bỏ đuôi, ví dụ `SWC-100`).
Cả hai cấu hình dùng encoder `all-MiniLM-L6-v2`, cùng `top-k` và cùng tập index nên chỉ khác ở việc có
duyệt đồ thị hay không. Script không gọi LLM, kết quả tất định.

### Chuẩn bị

```bash
pip install sentence-transformers scikit-learn numpy pandas neo4j python-dotenv openpyxl
```

Neo4j phải đã được nạp bằng `experiment/rq1_detection/build_neo4j_to_vector_indexes.py --execute`
(đủ node/quan hệ và các vector index). Cung cấp thông tin kết nối qua biến môi trường
`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` hoặc qua `--neo4j-env-file`.
Không commit file chứa mật khẩu.

Dataset đánh giá (`.jsonl`, `.csv` hoặc `.xlsx`) cần các cột `id`, `code`, `label`.
Nhãn `label` phải cùng không gian với nhãn của knowledge base (ví dụ `SWC-100`); nếu khác,
tạo file JSON ánh xạ `{"SWC-100": "Access Control", ...}` và truyền bằng `--label-map`.

### Chạy

```bash
# so sánh KB và KG
python experiment/rq4_knowledge/eval_kb_vs_kg.py \
    --dataset path/to/eval.jsonl --neo4j-env-file path/to/neo4j.env

# chỉ chạy KB (không cần Neo4j) để kiểm tra nhanh
python experiment/rq4_knowledge/eval_kb_vs_kg.py --dataset path/to/eval.jsonl --only kb --limit 20
```

| Tham số | Ý nghĩa |
|---|---|
| `--dataset` | Dữ liệu đánh giá (bắt buộc) |
| `--kb-csv` | CSV nguồn của KB flat (mặc định CSV của RQ1) |
| `--label-map` | JSON ánh xạ nhãn KB sang nhãn dataset |
| `--indexes` | Index dùng chung cho KB và KG, mặc định `vcode_index,behavior_index` |
| `--top-k` | Số láng giềng mỗi index, mặc định 3 |
| `--exclude-self` | Bỏ `File` có tên trùng `id` của mẫu, tránh rò rỉ khi mẫu test nằm trong KB |
| `--only` | `kb`, `kg` hoặc `both` |
| `--limit` | Chỉ đánh giá N mẫu đầu |
| `--out-dir` | Thư mục kết quả, mặc định `experiment/rq4_knowledge/results` |

### Đầu ra

Trong `results/` script ghi hai file theo mốc thời gian:

- `kb_vs_kg_<time>.json`: tham số chạy, Accuracy, macro/weighted Precision/Recall/F1 và chỉ số theo lớp;
- `kb_vs_kg_<time>_predictions.csv`: nhãn thật và dự đoán của từng mẫu cho mỗi cấu hình.

### Giới hạn cần biết

- Knowledge base chỉ có khoảng 69 mục, phần lớn mỗi nhãn một mục. Phương pháp này là truy xuất
  láng giềng gần nhất, không phải mô hình đã huấn luyện, nên dataset đánh giá nên là tập độc lập với KB.
- Nếu mẫu đánh giá nằm trong KB mà không dùng `--exclude-self`, cả hai cấu hình sẽ trả lại đúng đáp án
  và số liệu bị thổi phồng.
- Script này chưa được chạy với một Neo4j thật trong phiên viết; hãy chạy `--only kb --limit 20`
  trước, sau đó chạy `both` để kiểm tra kết nối.

## 3. Phạm vi tái lập

Các hình và bảng RQ4 của bài báo được tạo từ dự đoán đã lưu của các cấu hình có kiểm soát. Gói này có mã
để xem knowledge node và ngữ cảnh truy xuất được hình thành như thế nào, nhưng chưa có lệnh độc lập nào
tái tạo toàn bộ huấn luyện và suy luận của RQ4. Không mô tả nó như một benchmark huấn luyện chạy lại hoàn
toàn khi chưa công bố checkpoint và manifest chia dữ liệu tương ứng.
