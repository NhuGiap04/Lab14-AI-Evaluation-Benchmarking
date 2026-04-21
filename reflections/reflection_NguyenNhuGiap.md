# GROUP_NAME: C401-B6
# Báo cáo Cá nhân: Nguyễn Như Giáp

## Tổng quan Đóng góp
Tôi là thành viên chịu trách nhiệm tích hợp pipeline, tính toán chi phí token, và quản lý synthetic data generation. Đóng góp của tôi tập trung vào việc làm cho hệ thống chạy ổn định và hiệu quả.

## Những gì Tôi Đã Làm
- **Synthetic Data Generation**: Hoàn thành script data/synthetic_gen.py để tạo golden dataset với 55 test cases, bao gồm hard cases. Sử dụng OpenAI API để sinh QA pairs từ docs, với logic deduplication và concurrency.
- **Token Cost & Usage**: Thêm tính toán chi phí token trong main.py và ngine/generator_eval.py, bao gồm agent input/output cost, judge cost, và báo cáo chi tiết trong 
eports/summary.json.
- **Pipeline Integration**: Merge các branch, update main.py để chạy benchmark so sánh 2 models, và đảm bảo async performance (<2 phút cho 50+ cases).
- **Commits Chính**:
  - e654c5: Merge branch 'feat/openai'
  - 9c97aa9: update main
  - 746a50: Add token cost
  - 240fa75: done synthetic generation

## Kết quả Đạt được
- Dataset 55 cases với Hit Rate/MRR = 1.0.
- Chi phí agent giảm 30%, pass rate 81-87%.
- Hệ thống regression tự động quyết định model switch dựa trên policy.

## Bài học Rút ra
- Async/concurrency quan trọng cho hiệu năng; cần test kỹ merge conflicts.
- Token tracking giúp tối ưu cost mà không giảm chất lượng.
