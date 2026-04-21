# GROUP_NAME: C401-B6
# Báo cáo Cá nhân: Hoàng Văn Bắc

## Tổng quan Đóng góp
Tôi là thành viên chuyên phát triển engine đánh giá RAGAS. Đóng góp của tôi tập trung vào việc implement metrics Faithfulness và Answer Relevancy, giúp đánh giá chất lượng generation một cách chính xác.

## Những gì Tôi Đã Làm
- **RAGAS Faithfulness**: Implement logic tách answer thành atomic statements và verify với context bằng LLM. Tính score faithfulness = verified / total_statements.
- **Answer Relevancy**: Sinh lại câu hỏi từ answer và so sánh với question gốc bằng Jaccard overlap.
- **Token Tracking**: Thêm tracking token usage trong evaluator để tính cost.
- **Commits Chính**:
  - 5c1565e: update generator eval
  - 5247d44: update generator eval
  - 4ff265a: feat: implement RAGAS faithfulness + answer relevancy + token tracking

## Kết quả Đạt được
- Faithfulness score 0.77-0.87, Relevancy 0.31-0.34.
- Metrics giúp phát hiện hallucination và cải thiện agent.

## Bài học Rút ra
- LLM-based evaluation cần prompt kỹ để tránh bias; async giúp xử lý nhiều cases nhanh.
- RAGAS là công cụ mạnh để benchmark RAG systems.
