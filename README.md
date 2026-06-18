# FCOS Object Detection Model from Scratch

Dự án này triển khai mô hình FCOS (Fully Convolutional One-Stage Object Detection) từ đầu sử dụng PyTorch. Mô hình không sử dụng hộp neo (anchor-free), tối ưu hóa kiến trúc với các tính năng:
- **EMA** (Exponential Moving Average) cho trọng số mô hình.
- **TTA** (Test Time Augmentation) lật ngang + đa kích thước khi suy luận.
- **CBAM** (Convolutional Block Attention Module) trên các đặc trưng FPN.
- **SPPF** (Spatial Pyramid Pooling - Fast) trên tầng C5.
- **DCNv2** (Deformable Convolutional Networks v2) trong các nhánh dự đoán của head.

---

## 1. Cài đặt môi trường

Sử dụng Conda để kích hoạt môi trường làm việc:

```bash
conda activate xulyanh
```

Hoặc cài đặt các thư viện cần thiết qua `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 2. Cách huấn luyện

Để bắt đầu quá trình huấn luyện với cấu hình mặc định (tự động bật EMA, CBAM, SPPF, DCNv2):

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Mô hình tốt nhất trong quá trình val sẽ được lưu tại: `./models/best.pth`.

---

## 3. Cách chạy suy luận (Inference)

Chạy suy luận trên tập ảnh bất kỳ và lưu kết quả đầu ra dưới dạng tệp tin JSON (tự động sử dụng EMA và TTA mặc định):

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Các tham số bổ sung nếu cần tùy chỉnh:
- `--weights`: Đường dẫn tới tệp trọng số (mặc định: `./models/best.pth`).
- `--tta`: Bật/Tắt Test Time Augmentation (`True` hoặc `False`, mặc định: `True`).
- `--use_ema`: Sử dụng trọng số EMA (`True` hoặc `False`, mặc định: `True`).
- `--conf_thresh`: Ngưỡng độ tin cậy để loại bỏ hộp bao (mặc định: `0.01`).
- `--iou_thresh`: Ngưỡng IoU cho thuật toán NMS (mặc định: `0.5`).

---

## 4. Vị trí đặt mô hình hoặc trọng số mô hình

Trọng số tốt nhất thu được sau khi huấn luyện được đặt tại:
- `./models/best.pth`
