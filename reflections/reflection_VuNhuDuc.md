# GROUP_NAME: C401-B6
# Báo cáo Cá nhân: Vũ Như Đức

## Tổng quan Đóng góp
Tôi là thành viên chịu trách nhiệm phát triển Multi-Judge Consensus Engine và Retrieval Eval. Đóng góp của tôi đảm bảo hệ thống đánh giá khách quan và chính xác.

## Những gì Tôi Đã Làm
- **LLM Judge**: Implement multi-judge với 2 models, tính agreement rate, xử lý tie-breaker khi disagreement cao. Chấm theo rubric accuracy/completeness/safety/professionalism.
- **Retrieval Eval**: Tính Hit Rate và MRR cho top-k retrieval, chứng minh retrieval quality trước generation.
- **Integration**: Update main.py để sử dụng real judge thay vì mock.
- **Commits Chính**:
  - de5d7c6: update llm judge
  - d0ee93: Update main to use real judge
  - 3c061e: update retrieval eval

## Kết quả Đạt được
- Avg judge score 4.1-4.25/5, agreement rate cao.
- Hit Rate/MRR = 1.0, chứng minh retrieval tốt.

## Bài học Rút ra
- Multi-judge tăng reliability nhưng cần calibrate disagreement threshold.
- Retrieval metrics quan trọng để debug hallucination.
