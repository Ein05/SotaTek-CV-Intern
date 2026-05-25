---
title: Zero-Shot CAD Pattern Detector
colorFrom: cyan
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
---

# Bộ nhận diện ký hiệu CAD Zero-Shot

Đây là hệ thống Nhận diện ký hiệu dạng Zero-Shot dành cho bản vẽ kỹ thuật (sơ đồ CAD / BOM). Hệ thống cho phép người dùng tải lên ảnh bản vẽ và một ảnh ký hiệu mẫu được cắt ra, sau đó tự động phát hiện tất cả các vị trí xuất hiện của ký hiệu đó trên bản vẽ.

## Tính năng nổi bật
- **Khả năng Zero-Shot mạnh mẽ**: Nhận diện tổng quát cho bất kỳ ký hiệu nào (Cầu chì, Điện trở, Điốt, v.v.) mà không cần huấn luyện lại mô hình.
- **Đa quy mô & Đa góc quay**: Hỗ trợ dải thay đổi kích thước (từ 0.02x đến 2.5x) và 8 hướng xoay góc khác nhau (từ 0° đến 315°).
- **Bộ lọc CAD nâng cao**:
  - Loại bỏ văn bản và nhiễu dựa trên chiều cao của các thành phần liên thông (Connected Component).
  - Bộ lọc bất đối xứng (Asymmetry filter) để loại bỏ nhiễu dạng đối xứng (ví dụ: các điểm nối dây) đối với ký hiệu bất đối xứng.
  - Loại bỏ khung tên bản vẽ và bảng thống kê vật tư (BOM) ở góc phải bản vẽ (tránh nhiễu khu vực bảng lưới x >= 1050).
- **Giao diện trực quan**: Thanh trượt điều chỉnh đa tham số trực tiếp trong thời gian thực, các chế độ cấu hình sẵn (Presets) chạy thử nhanh, phóng to thu nhỏ bản vẽ kết quả, và xuất tọa độ dạng JSON.

## Chạy ứng dụng cục bộ

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

2. Khởi chạy ứng dụng Gradio:
   ```bash
   python gradio_app.py
   ```

3. Mở trình duyệt tại địa chỉ `http://localhost:7860`.

## Chạy thử nghiệm xác minh cục bộ

Chạy lệnh dưới đây để kiểm tra hoạt động của công cụ lõi trên cả 3 trường hợp mẫu (Cầu chì, Điện trở, Điốt) và đảm bảo số lượng phát hiện chính xác:
```bash
python verify_examples.py
```
Script này sẽ chạy quy trình xử lý trên các ảnh mẫu trong thư mục `examples/` và lưu các ảnh trực quan hóa kết quả dưới dạng `verify_*_output.png`.
