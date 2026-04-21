# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Nguồn dữ liệu phân tích:** `reports/summary.json`, `reports/benchmark_results.json` (timestamp 2026-04-21).
- **Tổng số cases:** 55 / model.
- **Model được hệ thống chọn hiện tại:** `gpt-4o` (Agent_Model_1).

### Kết quả Model 1 (gpt-4o)
- **Pass/Fail:** 45 / 10 (Pass rate: 81.82%)
- **Điểm LLM-Judge trung bình:** 4.1091 / 5.0
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.7727
  - Relevancy: 0.3446
- **Retrieval:** Hit Rate = 1.0, MRR = 1.0
- **Chi phí Agent (chỉ input/output):** 0.046848

### Kết quả Model 2 (gpt-5.4-mini)
- **Pass/Fail:** 48 / 7 (Pass rate: 87.27%)
- **Điểm LLM-Judge trung bình:** 4.2500 / 5.0
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.8727
  - Relevancy: 0.3089
- **Retrieval:** Hit Rate = 1.0, MRR = 1.0
- **Chi phí Agent (chỉ input/output):** 0.017554

### Nhận định nhanh
- Model 2 tốt hơn về `avg_score`, `faithfulness`, `pass_rate`, và rẻ hơn đáng kể.
- Tuy nhiên hệ thống **chưa chuyển model** vì policy hiện tại yêu cầu `relevancy` không được giảm (`relevancy_delta = -0.0357`).

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| False Refusal (trả lời “không tìm thấy” dù context có đáp án) | 7/10 fail của Model 1 | Prompt của agent quá bảo thủ + câu trả lời mẫu ngắn trong KB làm model dễ kết luận “không đủ thông tin”. |
| Policy Inference Failure (câu hỏi điều kiện/giả định) | 2/10 | Context có rule nhưng cần suy luận “if-then”; agent hiện tại không có bước rule reasoning riêng. |
| Label/Question Ambiguity | 1/10 | Một số câu hỏi kỳ vọng câu trả lời cụ thể hơn mức context nêu trực tiếp (ví dụ actor trong flow thông báo). |

Chi tiết theo nguồn tài liệu trên các fail của Model 1:
- `policy_refund_v4`: 4
- `access_control_sop`: 3
- `sla_p1_2026`: 2
- `hr_leave_policy`: 1

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: “Chính sách hoàn tiền này áp dụng từ ngày nào?” (score 1.0)
1. **Symptom:** Agent trả lời “Tôi không tìm thấy thông tin này trong tài liệu.”
2. **Why 1:** Agent từ chối dù context có câu “kể từ ngày 01/02/2026”.
3. **Why 2:** Prompt ưu tiên an toàn theo hướng từ chối khi “không đủ thông tin”.
4. **Why 3:** Bộ context top-k chứa nhiều câu policy khác nhau, làm tín hiệu bị nhiễu.
5. **Why 4:** Retriever hiện tại dùng lexical overlap trên QA snippets, không rerank theo intent chính.
6. **Root Cause:** Chiến lược generation quá bảo thủ kết hợp retrieval dạng snippet-level không có reranking.

### Case #2: “Quy trình xử lý sự cố P1 bắt đầu bằng bước nào?” (score 1.0)
1. **Symptom:** Agent trả lời “không tìm thấy”, dù context có “Bước 1: Tiếp nhận…”.
2. **Why 1:** Agent không trích xuất được mốc quy trình từ context ngắn.
3. **Why 2:** Context ghép kèm các đoạn “Bước 1” từ tài liệu khác (refund/escalation) gây mơ hồ.
4. **Why 3:** Retrieval không có bước disambiguation theo `source_doc` hoặc section-level filter.
5. **Why 4:** Pipeline chỉ có retrieval + generation, chưa có post-check “answer must cite evidence”.
6. **Root Cause:** Thiếu ràng buộc trích dẫn chứng cứ theo đúng tài liệu/section trước khi finalize câu trả lời.

### Case #3: “Khách hàng cần liên hệ với ai để được hỗ trợ hoàn tiền?” (score 1.25)
1. **Symptom:** Agent trả lời “không tìm thấy”.
2. **Why 1:** Câu trả lời kỳ vọng là “CS Agent”, nhưng context diễn đạt gián tiếp trong bước quy trình.
3. **Why 2:** Agent chưa có logic chuyển đổi từ mô tả quy trình sang vai trò liên hệ trực tiếp.
4. **Why 3:** Dữ liệu synthetic có một số câu hard/ambiguous khiến model ưu tiên safe fallback.
5. **Why 4:** Chưa có calibration rule cho các câu hỏi “ai/phòng ban chịu trách nhiệm”.
6. **Root Cause:** Khoảng cách giữa cách diễn đạt ground truth và cách diễn đạt context chưa được bridge bằng prompt/pattern matching.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **Retriever nâng cấp:** thêm reranking theo cross-encoder hoặc judge nhẹ để giảm nhiễu top-k.
- [ ] **Evidence-first prompting:** buộc agent trích 1-2 câu chứng cứ trước khi trả lời kết luận.
- [ ] **Rule reasoning cho policy questions:** thêm template xử lý câu điều kiện “nếu… thì…”.
- [ ] **Chuẩn hoá synthetic labels:** giảm các ground truth mang tính suy diễn quá mạnh khi context chỉ nói gián tiếp.
- [ ] **Đo chunk-level retrieval quality:** ngoài Hit/MRR theo doc-id, thêm metric “contains-answer-span”.
- [ ] **Review policy switch model:** hiện model 2 tốt hơn chất lượng/chi phí nhưng bị chặn bởi `relevancy`; cân nhắc điều chỉnh ngưỡng `MODEL_SWITCH_REQUIRE_NON_DECREASING_RAGAS` hoặc định nghĩa lại relevancy metric.
