# GROUP_NAME: C401-B6
# Báo cáo Cá nhân: Trần Anh Tú

## Tổng quan Đóng góp
Tôi là thành viên chịu trách nhiệm phát triển Main Agent. Đóng góp của tôi tập trung vào retrieval và generation logic, làm nền tảng cho hệ thống.

## Những gì Tôi Đã Làm
- **Main Agent**: Implement agent sử dụng lexical overlap để retrieve top-k contexts từ golden dataset.
- **Generation**: Tích hợp LLM (OpenAI/NVIDIA) để trả lời dựa trên context, với prompt SYSTEM_PROMPT ưu tiên faithfulness.
- **Fallback Handling**: Thêm logic fallback khi model không available.
- **Commits Chính**:
  - 6c616b3: feat: add main agent
  - 899e987: add agent v1

## Kết quả Đạt được
- Agent trả lời chính xác dựa trên context, giảm hallucination.
- Integration với evaluator và judge.

## Bài học Rút ra
- Retrieval đơn giản nhưng hiệu quả; cần reranking cho advanced cases.
- Prompt engineering quan trọng để agent không bịa thông tin.
