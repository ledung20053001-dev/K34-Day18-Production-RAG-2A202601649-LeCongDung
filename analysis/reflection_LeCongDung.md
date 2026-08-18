# Individual Reflection — Lab 18: Production RAG

**Họ tên:** Lê Công Dũng  
**Project:** Trợ lý tra cứu chính sách nội bộ

## Phần 1: Mapping bài giảng vào code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Trên 25 tài liệu đọc được, threshold `0.85` tạo **201 semantic chunks**, trong khi `chunk_basic(chunk_size=500)` tạo **50 chunks**. Ngưỡng cao làm nhiều cặp câu liền kề bị tách; kết quả giữ ranh giới ý tốt hơn nhưng có nguy cơ tạo chunk quá nhỏ. Cần tune threshold trên retrieval metrics thay vì dùng mặc định cho mọi corpus. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 bắt tốt số liệu và cụm từ chính xác; Dense Search bù vocabulary gap. RRF hợp nhất hai thứ hạng mà không phải chuẩn hóa hai thang điểm khác nhau. Một tài liệu xuất hiện cao ở cả hai danh sách sẽ được cộng điểm và ưu tiên ổn định hơn. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()` | Luồng đúng là retrieve top-20 rồi chấm trực tiếp từng cặp query–document và lấy top-3. Benchmark lexical fallback trong mock mode đạt trung bình **0.054 ms** (min `0.040 ms`, max `0.247 ms`) trên 3 documents; đây chỉ là kiểm tra overhead, **không phải latency hay precision của `bge-reranker-v2-m3`**. Precision model thật chưa được đo do không tải weights. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Report thật cho Production: Faithfulness **0.7583**, Answer Relevancy **0.7869**, Context Precision **0.9000**, Context Recall **0.8667**. So với baseline, chỉ Answer Relevancy tăng `+0.0223`; Faithfulness giảm mạnh nhất `-0.0833`. Bottom-5 có 4 câu bị gắn lỗi Faithfulness và 1 câu lỗi Context Precision. |
| Contextual embeddings | M5 | `_enrich_single_call()`, `contextual_prepend()`, `enrich_chunks()` | Combined mode tạo summary, hypothetical questions, contextual info và metadata trong một API call/chunk. Tuy nhiên report không có ablation M5 on/off, nên chưa thể khẳng định enrichment cải thiện retrieval. Production thực tế có Context Precision `0.9000` và Recall `0.8667`, đều thấp hơn baseline; cần chạy ablation để tách ảnh hưởng của enrichment khỏi chunking/search/reranking. |

### Nhận xét kỹ thuật

- Semantic threshold không có giá trị tối ưu tuyệt đối. `0.85` khá nghiêm với corpus chính sách ngắn, tạo số chunk gấp khoảng 4 lần baseline.
- Hybrid retrieval giải quyết hai nhóm query khác nhau: truy vấn chứa mã, số tiền, số ngày phù hợp BM25; truy vấn diễn đạt lại phù hợp Dense Search.
- CrossEncoder nên chỉ chạy sau candidate retrieval vì độ chính xác cao hơn bi-encoder nhưng chi phí tính toán tăng theo số cặp query–document.
- Bốn metric RAGAS chẩn đoán các tầng khác nhau: Recall/Precision nghiêng về retrieval, Faithfulness/Relevancy nghiêng về generation. Dữ liệu hiện tại cho thấy ưu tiên đầu tiên là Faithfulness, sau đó là Context Recall.
- Enrichment cần cache theo hash của chunk để tránh gọi lại LLM khi nội dung không thay đổi.

## Phần 2: Khó khăn và cách giải quyết

### 1. Thiếu thư viện đọc PDF

- **Exact error:** `ModuleNotFoundError: No module named 'pypdf'`
- **Hiện tượng:** `load_documents()` dừng khi gặp PDF, khiến test `test_compare_all_strategies` và pipeline không chạy hết.
- **Cách debug:** Chạy riêng `pytest tests/test_m1.py -q --tb=short`, lần theo stack trace tới `_extract_pdf_text()`.
- **Giải pháp:** Bọc import `pypdf` trong `try/except`, cảnh báo và trả chuỗi rỗng để bỏ qua PDF khi dependency chưa có. Markdown vẫn được index bình thường.
- **Kiến thức thiếu → bổ sung:** Cần phân biệt PDF có text layer với PDF scan. Bước tiếp theo là tìm hiểu OCR (Tesseract/PaddleOCR) thay vì coi mọi PDF rỗng là lỗi parser.

### 2. Windows console không in được tiếng Việt

- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3: character maps to <undefined>`
- **Hiện tượng:** Pipeline đã bắt được lỗi thiếu `pypdf`, nhưng lại crash khi in ký tự cảnh báo `⚠️` trên console CP1252.
- **Cách debug:** Đọc traceback và xác nhận lỗi nằm ở `print()`, không nằm ở chunking hoặc PDF parsing.
- **Giải pháp:** Cấu hình `sys.stdout` và `sys.stderr` sang UTF-8 ở entry point `src/pipeline.py` với `errors="replace"`.
- **Kiến thức thiếu → bổ sung:** Tìm hiểu sự khác nhau giữa encoding của file, terminal và Python I/O trên Windows; luôn test CLI bằng dữ liệu Unicode thật.

### 3. Không có Qdrant và các package retrieval trong môi trường tối giản

- **Exact errors/dependencies:** `qdrant-client`, `rank-bm25` và `underthesea` không có trong environment; Dense Search thật cũng yêu cầu model `BAAI/bge-m3`.
- **Cách debug:** Dùng `importlib.util.find_spec()` kiểm tra từng dependency, sau đó chạy test theo module để tách lỗi logic khỏi lỗi môi trường.
- **Giải pháp:** Lazy-load Qdrant/model; thêm tokenizer và BM25 fallback; tạo `RAG_MOCK_MODE=1` với dense vector hashing trong bộ nhớ. Nhờ đó có thể kiểm tra toàn bộ control flow mà không tải weights hay gọi dịch vụ ngoài.
- **Kiến thức thiếu → bổ sung:** Cần học thêm contract testing cho external service và dùng Qdrant in-memory/test container để kiểm tra payload, vector dimension và `query_points()` mà không phụ thuộc production server.

### 4. Mock mode ban đầu vẫn gọi OpenAI

- **Exact error:** `LLM generation failed: Connection error.`
- **Hiện tượng:** Pipeline chạy mock nhưng `run_query()` vẫn thấy API key và thử gọi OpenAI cho cả 20 câu, làm thời gian tăng lên khoảng 49 giây.
- **Cách debug:** Quan sát warning lặp đúng 20 lần và đối chiếu nhánh điều kiện trong `run_query()`.
- **Giải pháp:** Thêm điều kiện `and not MOCK_MODE` trước API call. Lần chạy sau hoàn thành trong khoảng **0.3 giây**, không dùng mạng.
- **Kiến thức thiếu → bổ sung:** Mock mode phải được thiết kế xuyên suốt mọi boundary (model, vector DB, LLM, evaluator), không chỉ ở retrieval.

### 5. Production chưa vượt baseline ở phần lớn metric

- **Hiện tượng:** Production chỉ tăng Answer Relevancy từ `0.7646` lên `0.7869`; Faithfulness giảm `0.8417 → 0.7583`, Context Precision giảm `0.9250 → 0.9000`, Context Recall giảm `0.9083 → 0.8667`.
- **Cách debug:** Đọc đồng thời hai file trong `reports/`, tính delta theo cùng metric và kiểm tra danh sách failures. Không dùng root-level mock report cũ để kết luận.
- **Giải pháp đề xuất:** Chạy ablation theo từng bước (hierarchical chunking, enrichment, hybrid search, reranking), giữ nguyên test set và generation settings; ưu tiên điều tra 4/5 Bottom-5 có lỗi Faithfulness.
- **Kiến thức thiếu → bổ sung:** Học thiết kế controlled experiment cho RAG. Một pipeline nhiều module hơn không mặc nhiên tốt hơn; cần cô lập contribution của từng module và theo dõi cả quality, latency, cost.

## Phần 3: Action Plan cho project

## Project: Trợ lý tra cứu chính sách nội bộ

### Hiện tại

- **RAG pipeline hiện tại:** Hierarchical chunking → Combined enrichment → BM25 + Dense Search → RRF → CrossEncoder reranking → LLM answer → RAGAS evaluation.
- **Known issues:** Production đã có report RAGAS trên 20 câu nhưng thấp hơn baseline ở 3/4 metric; report chưa lưu answer/context/per-question scores nên failure analysis chưa xác nhận được root cause; PDF chưa OCR; metadata hiệu lực phiên bản chưa đầy đủ; chưa có ablation cho enrichment/chunking/reranking.

### Plan áp dụng

1. [ ] **Chunking strategy:** Dùng hierarchical chunking làm mặc định (child khoảng 256, parent khoảng 2048) để retrieve chính xác trên child nhưng cung cấp parent cho LLM. Dùng structure-aware chunking cho Markdown; chỉ dùng semantic chunking sau khi tune threshold trên validation set.
2. [ ] **Search:** Dùng Hybrid BM25 + `BAAI/bge-m3` Dense Search qua RRF. BM25 xử lý tốt số tiền, ngày tháng và tên chính sách; Dense xử lý paraphrase. Bổ sung metadata filter cho loại chính sách và trạng thái hiệu lực.
3. [ ] **Reranking:** Có; dùng `BAAI/bge-reranker-v2-m3` trên top-20 candidates và lấy top-3. Benchmark latency thật, đồng thời chuẩn bị FlashRank hoặc lexical fallback khi tài nguyên hạn chế.
4. [ ] **Evaluation:** Tiếp tục dùng RAGAS cho bốn metric cốt lõi; bổ sung version accuracy, numeric exact match, negation accuracy, source/version citation và p95 latency. Lưu answer, contexts và toàn bộ per-question scores vào report; dùng ablation để so sánh từng module với baseline.
5. [ ] **Enrichment:** Dùng combined single-call để sinh contextual info, 2–3 hypothetical questions, summary và metadata. Cache theo content hash; ưu tiên metadata `version`, `effective_date`, `status`, `policy_type` để giải quyết xung đột tài liệu.

### Timeline

- **Tuần 1:** Cài dependencies, chạy Qdrant bằng Docker, xử lý PDF text/OCR và chuẩn hóa metadata nguồn.
- **Tuần 2:** Index hierarchical + structure-aware chunks; chạy BM25, Dense và Hybrid retrieval; đo Recall@K/MRR trên 20 câu hiện có.
- **Tuần 3:** Tích hợp CrossEncoder, tune candidate top-k/RRF và thêm version-aware filtering; bổ sung test cho policy cũ/mới.
- **Tuần 4:** Bật combined enrichment và cache; tối ưu prompt trả lời có citation, xử lý negation và câu hỏi số học.
- **Tuần 5:** Chạy lại RAGAS cùng custom metrics; mục tiêu vượt baseline ở Faithfulness, Context Precision và Context Recall; phân tích Bottom-5 bằng answer/context đầy đủ và đo latency/chi phí.
- **Tuần 6:** Hoàn thiện monitoring, fallback, báo cáo so sánh naive–production và kiểm thử end-to-end trước khi triển khai.
