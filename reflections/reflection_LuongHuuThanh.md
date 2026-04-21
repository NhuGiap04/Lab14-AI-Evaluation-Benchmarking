# GROUP_NAME: C401-B6
# Báo cáo Cá nhân: Lương Hữu Thành

## Tổng quan Đóng góp
Tôi là thành viên nhóm chuyên synthetic data generation. Đóng góp của tôi tạo ra dataset chất lượng cho benchmark.

## Những gì Tôi Đã Làm
- **Synthetic Gen Update**: Cập nhật data/synthetic_gen.py để sinh golden_set.jsonl với 55 cases, bao gồm hard cases từ guide.
- **Data Processing**: Logic normalize cases, allocate counts per doc, và top-up nếu thiếu.
- **Commits Chính**:
  - 1941ad9: update synthetic gen data and golden_set gen from this file

## Kết quả Đạt được
- Golden dataset 55 cases, hỗ trợ retrieval eval với Hit Rate/MRR.

## Bài học Rút ra
- SDG cần guide cho hard cases; concurrency giúp sinh nhanh.
- Dataset quality quyết định benchmark accuracy.
