# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Lê Công Dũng
**Nguồn dữ liệu:** `reports/naive_baseline_report.json` và `reports/ragas_report.json`

> Report production chỉ lưu câu hỏi, worst metric, score, diagnosis và suggested fix; không lưu generated answer, retrieved contexts hoặc điểm đầy đủ theo từng câu. Vì vậy các mục **Got** được ghi là “không có dữ liệu” và root cause chi tiết được trình bày dưới dạng giả thuyết cần xác minh, không suy diễn thành kết luận.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ Production − Naive |
|---|---:|---:|---:|
| Faithfulness | 0.8417 | 0.7583 | -0.0833 |
| Answer Relevancy | 0.7646 | 0.7869 | +0.0223 |
| Context Precision | 0.9250 | 0.9000 | -0.0250 |
| Context Recall | 0.9083 | 0.8667 | -0.0417 |

### Nhận xét tổng quan

- Production tăng **Answer Relevancy 0.0223**, cho thấy câu trả lời nhìn chung sát trọng tâm hơn.
- Ba metric còn lại đều giảm so với baseline. Giảm mạnh nhất là **Faithfulness (-0.0833)**, tiếp theo là **Context Recall (-0.0417)** và **Context Precision (-0.0250)**.
- Production vẫn đạt trên `0.75` ở cả bốn metric, nhưng pipeline phức tạp hơn chưa tạo cải thiện tổng thể so với baseline.
- Baseline report không lưu failures; Production report lưu 10 failures. Bottom-5 dưới đây lấy đúng 5 phần tử đầu theo thứ tự trong `reports/ragas_report.json`.

## Bottom-5 Failures

### #1 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành v2.0, mật khẩu phải đổi mỗi 120 ngày; quy định 90 ngày thuộc phiên bản cũ đã bị thay thế.
- **Got:** Không được lưu trong `reports/ragas_report.json`.
- **Worst metric:** Faithfulness = **0.0000**.
- **Error Tree:** Output được context hỗ trợ? → Không theo RAGAS → Context có thể chứa cả v1 và v2? → Cần kiểm tra log → Generator có phân biệt chính sách hiện hành? → Cần xác minh.
- **Diagnosis từ report:** `LLM hallucinating`.
- **Root cause giả thuyết:** Pipeline có thể đưa đồng thời quy định 90 ngày và 120 ngày vào context, sau đó generator chọn hoặc trộn sai phiên bản. Không thể xác nhận nếu thiếu answer/context theo câu.
- **Suggested fix:** Thêm metadata `version`, `status=current|superseded`, ưu tiên v2 khi retrieve/rerank; prompt bắt buộc nêu nguồn và không dùng chính sách đã bị thay thế.

### #2 — Phê duyệt thiết bị 55 triệu

- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Không được lưu trong report.
- **Worst metric:** Faithfulness = **0.0000**.
- **Error Tree:** Output được context hỗ trợ? → Không theo RAGAS → Đúng ngưỡng “trên 50 triệu” được retrieve? → Cần kiểm tra → LLM áp dụng đúng điều kiện số học? → Cần xác minh.
- **Diagnosis từ report:** `LLM hallucinating`.
- **Root cause giả thuyết:** Generator có thể nhầm ranh giới phê duyệt `5–50 triệu` với `trên 50 triệu`, hoặc thêm người phê duyệt không xuất hiện trong context.
- **Suggested fix:** Giữ nguyên bảng/ngưỡng phê duyệt trong một chunk structure-aware; thêm kiểm tra numeric/rule-based trước generation và yêu cầu trích dẫn dòng quy định.

### #3 — Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần CEO phê duyệt; nghỉ trên 14 ngày còn làm phát sinh nghĩa vụ tự đóng phần bảo hiểm của nhân viên.
- **Got:** Không được lưu trong report.
- **Worst metric:** Context Precision = **0.5000**.
- **Error Tree:** Output sai/chưa đủ? → Không xác định vì thiếu answer → Có quá nhiều context không liên quan? → Có theo RAGAS → Search/reranker lọc đúng khoảng 16–30 ngày? → Có khả năng chưa tốt.
- **Diagnosis từ report:** `Too many irrelevant chunks`.
- **Root cause giả thuyết:** Query chứa nhiều từ phổ biến như “nghỉ”, “phép”, “ngày”, “phê duyệt”, khiến các chính sách nghỉ khác chen vào top contexts.
- **Suggested fix:** Dùng metadata filter `policy_type=unpaid_leave`; tăng trọng số cụm `không_lương`; rerank theo cả loại phép và khoảng ngày; giảm số context gửi vào LLM nếu điểm rerank thấp.

### #4 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; cần xác nhận cấu hình kỹ thuật từ phòng CNTT và ít nhất 3 báo giá vì giá trị trên 10 triệu.
- **Got:** Không được lưu trong report.
- **Worst metric:** Faithfulness = **0.5000**.
- **Error Tree:** Output được context hỗ trợ hoàn toàn? → Chỉ một phần theo score → Context có đủ ba điều kiện? → Cần kiểm tra → Multi-hop synthesis có bỏ sót hoặc thêm thông tin? → Có khả năng.
- **Diagnosis từ report:** `LLM hallucinating`.
- **Root cause giả thuyết:** Đây là câu multi-hop cần kết hợp ngưỡng phê duyệt, quy định thiết bị CNTT và yêu cầu báo giá; generation có thể thêm/bỏ một điều kiện hoặc gắn sai thẩm quyền.
- **Suggested fix:** Retrieve theo ba sub-query, gom bằng chứng theo từng điều kiện, rồi dùng prompt có checklist: người phê duyệt, xác nhận CNTT, số báo giá.

### #5 — Phạt tạm ứng quá hạn

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; mức 2%/tháng trên 15 triệu là 300.000 VNĐ/tháng, tương đương khoảng 50.000 VNĐ cho 5 ngày theo pro-rata.
- **Got:** Không được lưu trong report.
- **Worst metric:** Faithfulness = **0.0000**.
- **Error Tree:** Output được context hỗ trợ? → Không theo RAGAS → Context có đủ hạn 15 ngày và mức 2%/tháng? → Cần kiểm tra → Phép tính pro-rata đúng? → Có khả năng sai.
- **Diagnosis từ report:** `LLM hallucinating`.
- **Root cause giả thuyết:** LLM có thể tính sai số ngày quá hạn, áp dụng 2% cho toàn kỳ thay vì pro-rata, hoặc sinh con số không được context hỗ trợ.
- **Suggested fix:** Tách retrieval dữ kiện khỏi computation; dùng hàm tính xác định cho `principal × monthly_rate × overdue_days / 30`; đưa công thức và dữ kiện nguồn vào answer.

## Case Study (cho presentation)

**Question chọn phân tích:** “Nghỉ phép không lương 20 ngày cần ai phê duyệt?”

**Lý do chọn:** Đây là failure duy nhất trong Bottom-5 có worst metric thuộc retrieval (`Context Precision = 0.5`), phù hợp để minh họa Error Tree thay vì chỉ quy mọi lỗi cho LLM.

**Error Tree walkthrough:**

1. Output đúng? → Chưa xác minh được vì report không lưu generated answer.
2. Context đúng? → Precision 0.5 cho thấy có context liên quan nhưng bị trộn nhiều chunk không liên quan.
3. Query rewrite OK? → Có thể chưa giữ đủ cụm phân biệt “nghỉ không lương” và điều kiện “20 ngày”.
4. Fix ở bước: phrase-aware BM25 → metadata filter loại phép → numeric-aware reranking → chỉ gửi top contexts đủ bằng chứng.

## Nếu có thêm 1 giờ, sẽ optimize

- Sửa `save_report()` để lưu toàn bộ per-question metrics, answer, contexts và ground truth; khi đó root cause có thể được xác nhận thay vì suy đoán.
- Ưu tiên tối ưu Faithfulness vì đây là metric giảm mạnh nhất và xuất hiện ở 4/5 Bottom-5.
- Thêm metadata phiên bản/hiệu lực và kiểm thử riêng cho mật khẩu v1–v2.
- Thêm numeric unit tests cho ngưỡng mua sắm, tạm ứng và lương thử việc.
- Chạy lại cùng test set để so sánh trước/sau bằng đúng bốn metric và latency.
