# Chạy ViT baseline cho DS1 và DS2

**Thái Văn Huy · 23130068**

Chép thư mục `experiments` và `reports` vào gốc repo **BrainTumorDetect**. Mã thử nghiệm dùng `data/ds12_index.py` và `data/skull_crop.py` trên nhánh `dev` của dự án. Kết quả đã chạy nằm trong [báo cáo tuần 2](../reports/vit_baseline/BAO_CAO_TUAN2.md).

## Dữ liệu và cách chia tập

| Bộ dữ liệu | Bài toán | Chia tập |
| --- | --- | --- |
| DS1 · navoneel | Nhị phân có u và không u, 253 ảnh | 5-fold phân tầng theo lớp và nhóm ảnh trùng. Mỗi lượt giữ một fold để test. Trong phần còn lại, dành khoảng 20% số nhóm cho validation. |
| DS2 · indk214 classification | Bốn lớp, 7.023 ảnh | Chia lại toàn bộ ảnh classification theo tỷ lệ mục tiêu 70/15/15. Các ảnh trùng pixel đi cùng một tập. Cột `original_split` lưu nguồn Training/Testing của Kaggle. |

Mã đọc ảnh gốc theo tên đầy đủ, gồm cả hai tệp navoneel khác đuôi có tên cơ sở `Y92`. Nhóm trùng được xác định từ kích thước và ma trận pixel xám. Cả hai chế độ dùng chung `splits.csv`, crop vùng đầu và CLAHE của repo, sau đó resize và chuẩn hóa theo checkpoint. Mỗi chế độ đánh giá test bằng trọng số ở epoch có validation loss thấp nhất. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

## Cài và chạy trong Git Bash trên Windows

Hai bộ ảnh đã giải nén vào `Downloads/navoneel` và `Downloads/indk214` được tự tìm. Nếu bạn đặt nơi khác, thêm `--data "D:/MRI_Datasets/navoneel"` hoặc đường dẫn tương ứng của indk214. Môi trường tham chiếu dùng Python 3.12.

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/requirements-vit.txt
```

Lệnh cài trên dùng PyTorch CPU. Máy chạy GPU NVIDIA có thể chọn gói CUDA tương ứng theo trang PyTorch, rồi cài các thư viện còn lại như trên. Chương trình tự chọn CUDA khi PyTorch nhận GPU. [PyTorch Get Started](https://pytorch.org/get-started/locally/)

Hai lệnh sau dùng cấu hình của báo cáo tuần 2, ảnh **112 × 112** và **FP32 trên CPU**. Chọn thư mục mới với `--output` khi chạy lại.

```bash
python -m experiments.train_vit_baseline --dataset ds1 --image-size 112 --device cpu --output runs/tuan2_ds1
python -m experiments.train_vit_baseline --dataset ds2 --image-size 112 --device cpu --output runs/tuan2_ds2
```

Có thể thêm `--bf16` trên phần cứng hỗ trợ BF16. Kích thước mặc định là **224 × 224** theo cấu hình ViT của repo. Lần đầu chương trình tải checkpoint ViT-Tiny từ Hugging Face. Nếu đã có file checkpoint `.safetensors`, truyền thêm `--weights "duong/dan/model.safetensors"`. [Checkpoint ViT-Tiny](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)

| File đầu ra | Nội dung |
| --- | --- |
| `config.json` | Backbone, kích thước ảnh, seed, optimizer và số epoch |
| `splits.csv` | Ảnh, lớp, fold, tập con và mã nhóm ảnh trùng |
| `*_history.csv` | Train/validation loss và accuracy theo epoch |
| `*_predictions.csv` | Dự đoán trên test cùng xác suất từng lớp |
| `summary.json` | Accuracy, precision, recall, F1 và confusion matrix |

DS1 dùng recall lớp có u. DS2 dùng macro recall của bốn lớp và ghi thêm recall phát hiện có u. Mã huấn luyện cập nhật log sau mỗi fold hoàn tất. [Cách tính recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
