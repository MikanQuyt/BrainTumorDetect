# Tuần 3 — Thái Văn Huy (23130068)

Phần việc: train ViT-LoRA **r=8, α=16** trên DS1 (5-fold CV) và DS2; chọn ngưỡng
0.35–0.40 để hướng tới Recall có u **>96%**; đánh giá Mean Box IoU và Pointing Game
trên DS2. Notebook chạy Colab nằm đúng tại **`notebooks/train.ipynb`**.

## 1. File mới và phạm vi

| File | Mục đích |
|---|---|
| `data/prepare_huy_week3.py` | Chuẩn bị ảnh/mask, tránh đè tên Y92, giữ fold DS1 và sửa xung đột nhóm ảnh trùng trong split DS2 |
| `engine/train_vit_lora_huy.py` | Huấn luyện, resume, chọn ngưỡng trên validation và đánh giá test |
| `notebooks/train.ipynb` | Cài môi trường, lấy dữ liệu, mount Drive, chạy và xem kết quả |
| `experiments/requirements-huy-week3.txt` | Phiên bản thư viện cho lượt chạy của Huy; giữ torch/torchvision CUDA của Colab |
| `tests/test_huy_week3.py` | Kiểm tra leakage, mask, threshold, checkpoint, early stopping và Auto-Prompt |
| `experiments/HUONG_DAN_TUAN3_HUY.md` | Hướng dẫn này |

Dùng lại `ViTLoRA`, `AutoPromptGenerator`, loss, `run_epoch` và
`UnifiedBrainMRIDataset` của nhóm. Không thay kiến trúc chung, không ghi đè annotation
của thành viên khác. Ảnh và annotation riêng sinh trong `dataset/huy_week3/` (đã được
quy tắc `.gitignore` hiện tại loại khỏi Git).

## 2. Chạy trên Colab theo từng bước

1. Tải **`train.ipynb`** và **`BrainTumorDetect_Huy_week3.zip`** đi kèm. ZIP là bản source
   để notebook chạy được ngay; không chứa ảnh MRI hoặc checkpoint.
2. Mở Google Colab → File → Upload notebook → chọn `train.ipynb`.
3. Runtime → Change runtime type → chọn GPU.
4. Ở ô cấu hình, sửa **`RUN_DIR`** thành đúng thư mục của Huy. Nếu thư mục nhóm
   chỉ hiện ở “Shared with me”, thêm shortcut vào My Drive rồi sử dụng đường dẫn đó.
   Đường dẫn mặc định `/content/drive/MyDrive/Huy/tuan3` yêu cầu đã có shortcut **Huy**
   trong My Drive trỏ tới thư mục Huy thuộc ProjectAI. Notebook kiểm tra thư mục cha
   tồn tại trước khi tạo thư mục kết quả, tránh tự tạo nhầm một thư mục Huy mới.
5. Giữ `SOURCE_MODE="archive"`. Chạy ô mount Drive rồi ô nạp source; lần đầu upload
   `BrainTumorDetect_Huy_week3.zip`. Notebook lưu ZIP này vào Drive để phiên sau tự đọc lại.
6. Chạy ô cài thư viện và kiểm tra GPU, sau đó chạy các kiểm tra offline.
7. Chạy ô tải hai dataset Kaggle. Nếu Kaggle yêu cầu đăng nhập, tải ZIP bằng tài khoản
   của mình, lưu lên Drive, đổi `DATA_MODE="drive_zip"` và điền hai đường dẫn ZIP.
8. Chạy ô chuẩn bị dữ liệu. Kiểm tra `audit.json`: DS1 có 253 record và 253 đường dẫn
   riêng; DS2 có 7.023 ảnh classification và 2.192 ảnh-mask; chương trình không báo leakage.
9. Chạy ô DS1 và ô DS2. Có thể đặt `DS1_FOLD=1` để chạy từng fold; dùng lại cùng thư mục
   run khi chạy các fold tiếp theo. `None` chạy lần lượt đủ 5 fold.
10. Chạy ô xem kết quả. Chỉ dùng số liệu xuất từ lượt chạy thật để viết báo cáo.

Sau khi push source lên GitHub, có thể dùng `SOURCE_MODE="github"` và
`REPO_REF="thai_van_huy_tuan3"`; nhánh đó phải tồn tại trên remote trước khi chọn cách này.

## 3. Cách chia tập và tiền xử lý

- DS1 giữ nguyên danh sách **train/validation/test của từng fold tuần 2**. Mỗi ảnh xuất
  hiện trong test đúng một lần. Validation dùng chọn epoch và ngưỡng; outer test dùng
  báo cáo cuối. Không dùng outer test làm validation.
- DS2 bắt đầu từ split classification tuần 2: **4.930 / 1.064 / 1.029**, sau khi kiểm tra
  tên ảnh, hash ảnh gốc và tính hợp lệ của split gốc. Nhóm trùng được mở rộng bằng ảnh
  crop/CLAHE và các liên kết qua segmentation; các nhóm mới có thể nối nhiều split cũ.
- Khi có xung đột DS2, cả nhóm được đưa vào cùng tập theo quy tắc cố định
  **test > validation > train**. Mọi ảnh test cũ vẫn ở test; không đưa ảnh validation/test
  cũ vào train. Giữ toàn bộ ảnh, không bỏ qua bước kiểm tra nhóm trùng sau khi sửa split.
  Quy tắc này được áp dụng trước huấn luyện, không dùng dự đoán hoặc chỉ số trên test.
- Mẫu chuyển tập nằm trong `ds2_split_changes.csv`; `audit.json` ghi số nhóm xung đột,
  số ảnh chuyển, kích thước classification trước/sau. Kích thước các tập DS2 có thể khác
  tuần 2, nên không xem hai kết quả là đối chứng trên một split hoàn toàn giống nhau.
- Bổ sung segmentation để học box; ảnh segmentation trùng classification phải cùng split.
  Các nhóm segmentation còn lại được chia phân tầng theo nhóm với seed cố định.
- Đây là kiểm tra trùng chính xác ở cấp ảnh; chưa xác minh ảnh gần trùng hoặc độc lập bệnh nhân.
- Tên đầu ra giữ cả đuôi gốc: `Y92.jpg.png` khác `Y92.png.png`. Cắt mask cùng khung với
  MRI, kiểm tra bảo toàn số pixel u, rồi suy box `xyxy` với biên phải/dưới loại trừ.
- Train sử dụng augmentation đồng bộ ảnh/mask/box của DataLoader chung. Validation/test
  không augmentation. ViT nhận 224 × 224, prompt/mask đánh giá nằm trong hệ 1024 × 1024.

## 4. Cấu hình huấn luyện và hai checkpoint

| Mục | Mặc định |
|---|---|
| Backbone | `google/vit-base-patch16-224-in21k` |
| LoRA | r=8, α=16, query/value của 12 tầng; backbone gốc đóng băng |
| DS1 | Head nhị phân; không box head vì không có ground truth box/mask |
| DS2 | Head bốn lớp + ViTDet box head; GIoU + L1 chỉ trên mẫu có box |
| Loss phân loại | Focal loss, class weights từ train, label smoothing 0.05 |
| Epoch tối đa | 1000; không bắt buộc chạy đủ |
| Chọn mô hình | Validation total loss thấp nhất |
| Early stopping | 25 epoch không cải thiện đáng kể; min_delta=0.0001 |
| Scheduler | ReduceLROnPlateau theo validation loss, patience=5, giảm LR ×0.5 |
| Optimizer | AdamW; LR LoRA=0.0003, head/box=0.001 |
| Batch size | 8; AMP trên CUDA |

AdamW/backprop cập nhật trọng số; scheduler điều chỉnh learning rate. Không có quy tắc
loss bắt buộc phải về 0.0x. Với Focal Loss, loss nhỏ cũng không tự chứng minh phân loại tốt.
Đánh giá bằng dữ liệu validation/test và Recall, Specificity, FP/FN, AUC cùng chỉ số box.

Mỗi fold có hai file trong thư mục Drive đã cấu hình:

- **`loss_best.pth`**: LoRA + classification head (+ box head DS2) tại validation loss
  thấp nhất; dùng đánh giá/suy luận. Cần nạp đúng pretrained backbone, không phải một
  bản ViT đầy đủ có thể mở độc lập.
- **`loss_latest.pth`**: thêm optimizer, scheduler, AMP scaler, epoch, history, counters,
  RNG và bản sao best adapters để tiếp tục train. Ghi sau mỗi **epoch hoàn chỉnh**, qua
  file tạm rồi thay file đích. Ngắt giữa epoch thì chạy lại epoch đó.

Khi mở phiên Colab mới: giữ nguyên `RUN_DIR`, source và cấu hình; chạy lại các ô.
Trainer tự resume latest và tự bỏ qua fold đã hoàn tất. Nếu dùng phiên khác, phải có
quyền đọc/ghi cùng thư mục Drive. Không chạy hai phiên đồng thời vào một thư mục run.
Code kiểm tra fingerprint dữ liệu, cấu hình, source và frozen backbone khi resume.
Đổi cấu hình hoặc batch size thì dùng thư mục run mới.

Tài liệu nền: [PyTorch — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
và [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 5. Threshold Moving và báo cáo kết quả

1. Nạp best checkpoint sau huấn luyện.
2. Lấy xác suất trên **validation classification**; DS2 không trộn các ảnh segmentation
   toàn dương vào tập chọn ngưỡng.
3. Quét `0.35, 0.36, 0.37, 0.38, 0.39, 0.40`. Trong các ngưỡng có Recall **>0.96**,
   ưu tiên Specificity cao nhất, rồi F1, rồi ngưỡng cao hơn. Nếu không đạt, ghi rõ
   `validation_target_met=false`; chọn Recall cao nhất trong phạm vi, không tự mở rộng range.
4. Cố định ngưỡng, đánh giá test một lần và so với ngưỡng 0.50.

DS1 dùng `P(tumor)`; DS2 dùng **`1 - P(no_tumor)`**, không dùng xác suất riêng một loại u.
Recall có u khác Macro Recall bốn lớp. Khi hạ ngưỡng, phải xem cả FP/Specificity vì Recall
có thể tăng cùng số báo động nhầm. Mỗi fold DS1 có ngưỡng chọn từ validation riêng.
Tổng hợp đủ 253 dự đoán outer-test mới xuất `ds1/cv_summary.json`.

DS2 đánh giá **toàn bộ ảnh segmentation test**, Auto-Prompt chỉ nhận ảnh, không nhận GT box/mask:

- Mean Box IoU: box head tự sinh, thêm padding 8% theo bộ Auto-Prompt chung, so với box từ mask.
- Pointing Game Accuracy: peak point dự đoán từ Rollout/LayerCAM rơi trong mask u thật.
  Chỉ số này không đòi điểm trùng chính xác với trọng tâm hình học khối u.
- Báo cả chất lượng box/point chưa qua cổng phát hiện và chất lượng toàn pipeline.
  Ở bản toàn pipeline, ảnh có u bị detector bỏ sót vẫn nằm trong mẫu số và được tính 0.
- Có kết quả tổng và theo từng loại u, không chỉ báo kết quả một số ảnh được chọn.

Tuần 2 là ViT-Tiny/112, tuần 3 là ViT-Base/224: cần ghi rõ khác biệt này, chưa thể quy mọi
cải thiện chỉ cho LoRA. Không hứa trước Recall >96% hoặc khẳng định LoRA loại bỏ overfitting.

Đầu ra theo mỗi fold: `config.json`, `history.csv`, `learning_curves.png`,
`threshold_selection.json`, `validation_threshold_curve.csv`, `validation_predictions.csv`,
`test_predictions.csv`, `test_metrics.json`, `loss_best.pth`, `loss_latest.pth`.
DS2 có thêm `localization_predictions.csv` và `localization_summary.json`.
`test_metrics.json` chỉ được ghi cuối cùng khi toàn bộ đánh giá hoàn tất.

Notebook chuyển stdout/stderr của tiến trình con ra output trực tiếp. Log tiền xử lý,
kiểm tra và huấn luyện được lưu trong `logs/` dưới `RUN_DIR`, gồm
`prepare_data.log`, `pipeline_tests.log`, `train_ds1.log`, `train_ds2.log`.
Nếu một lệnh dừng, đọc traceback trong log trước dòng `CalledProcessError` để xác định
nguyên nhân; không suy ra nguyên nhân chỉ từ mã thoát của tiến trình.

## 6. Kiểm tra tại máy và lệnh chạy

Chạy ở thư mục gốc repo. Dùng môi trường Python riêng; cài torch/torchvision tương ứng
CPU hoặc CUDA từ PyTorch, rồi cài requirements của Huy. Không thay torch CUDA của Colab bằng gói CPU.

```bash
python -m pip install -r experiments/requirements-huy-week3.txt
python -m unittest tests.test_huy_week3 -v
python -m data.prepare_huy_week3
python -m engine.train_vit_lora_huy --dataset ds1 --epochs 1000 --out runs/huy_week3
python -m engine.train_vit_lora_huy --dataset ds2 --epochs 1000 --out runs/huy_week3
```

Các kiểm tra offline dùng dữ liệu giả lập và backbone nhỏ, không tải pretrained hoặc
chạy thực nghiệm MRI thật. Có kiểm tra resume khớp lượt chạy liên tục, early stopping,
threshold, ảnh trùng, mask và một lượt box head/Auto-Prompt. Máy có GPU NVIDIA dùng thêm
`--device cuda --amp`. Trên máy cá nhân, `runs/huy_week3` lưu trên đĩa máy; notebook hỗ trợ
sao chép tự động sang thư mục Google Drive for desktop nếu cấu hình `DRIVE_SYNC_DIR`
như mục 6.1.

Source đã được kiểm tra bằng **68/68 tests của repo**, gồm 15 tests mới của Huy,
trên CPU với torch 2.6.0 và requirements của Huy. Lượt MRI thực hoàn tất ngày
19/11/2026: DS1 đủ năm fold (253 dự đoán OOF), DS2 dừng ở epoch 104 và dùng best
checkpoint epoch 102. Recall DS1 sau chỉnh ngưỡng là **95,48%, chưa vượt 96%**;
DS2 là **99,73%**, Mean Box IoU **0,6261**, Pointing Game **78,72%**.

Số liệu đã được đối chiếu lại với CSV từng mẫu, ngưỡng chọn trên validation, lịch sử
huấn luyện và fingerprint source. Chưa tải lại mask gốc để chấm lại từng point hit
trong lần audit cuối. Notebook local runtime, GPU Windows và đồng bộ checkpoint qua
Drive for desktop đã được quan sát hoạt động trong lượt chạy thật. Xem báo cáo tại
`reports/tuan3/vit_lora_huy/BAO_CAO_TUAN3.md` và dữ liệu kiểm chứng đi kèm.

### 6.1. Giao diện Colab với GPU Windows qua local runtime

Dùng lại môi trường `braintumor_tuan3`, source, dữ liệu và checkpoint đã chạy bằng CMD.
Chỉ thay notebook `notebooks/train.ipynb` bằng bản hỗ trợ local runtime; giữ nguyên các
file huấn luyện/mô hình vì checkpoint kiểm tra fingerprint của chúng. Notebook tự chọn
`RUNTIME_MODE="local"` trên Windows, mặc định `RUN_DS1=False`, `RUN_DS2=True`,
`NUM_WORKERS=0`; bỏ qua cài lại requirements, tải dữ liệu và tiền xử lý.

1. Trong lượt CMD đang train, đợi dòng tổng kết epoch dạng
   `E.../1000 train=... val=... best=... patience=... lr=...` rồi nhấn **Ctrl+C**.
   Dòng này xuất hiện sau khi checkpoint và history đã ghi xong. Nếu đã sang epoch mới,
   phần epoch mới chưa hoàn thành sẽ được chạy lại khi resume.
2. Khi CMD đã trở về dấu nhắc và vẫn ở môi trường `braintumor_tuan3`, cài Jupyter:

   ```bat
   python -m pip install "notebook>=7.4,<8" ipykernel
   ```

3. Từ thư mục gốc dự án, khởi động server chỉ trên máy cá nhân, giữ xác thực token:

   ```bat
   python -m notebook --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0 --ServerApp.allow_origin=https://colab.research.google.com --ServerApp.allow_credentials=True
   ```

4. Mở bản `train.ipynb` mới bằng Colab. Chọn **Connect → Connect to a local runtime**,
   dán URL `http://127.0.0.1:8888/?token=...` do Jupyter vừa in ra. Token chỉ dùng để
   kết nối trong trình duyệt, không đưa vào báo cáo hoặc Git.
5. Chạy ô cấu hình, xác nhận `LOCAL_PROJECT_DIR` là thư mục source thực tế. Sau đó chạy
   từ trên xuống. Chế độ local yêu cầu checkpoint DS2 tồn tại và phát hiện tiến trình
   CMD khác đang ghi cùng thư mục. Dòng `Checkpoint epoch` và `RESUME ... next epoch`
   cho biết vị trí tiếp tục; không cố định cứng về epoch 39/40.

Không chạy lại lệnh train CMD khi notebook đang train. Notebook vẫn gọi đúng trainer
hiện tại, giữ batch size 8, AMP, optimizer/scheduler và cấu hình huấn luyện. Thông tin
Python/PyTorch/GPU cùng `pip freeze` được lưu trong `RUN_DIR/environment/` để bàn giao.

### 6.2. Đồng bộ checkpoint từ local runtime lên Drive

Local runtime không hỗ trợ `drive.mount()` như runtime Google quản lý. Việc notebook
và output ô được lưu trên Drive không tự tải các file `.pth` trong máy lên Drive.

Nếu đã có Google Drive for desktop và shortcut `Huy` trong My Drive, mở đúng thư mục
`Huy/tuan3` của nhóm trong File Explorer, sao chép đường dẫn thực tế rồi đặt:

```python
DRIVE_SYNC_DIR = Path(r"duong_dan_thuc_te_den_Huy\tuan3")
```

Giữ `RUN_DIR` là thư mục local đang chứa checkpoint mới nhất. Sau mỗi dòng tổng kết
epoch đã lưu và khi tiến trình kết thúc, notebook sao chép các file `.pth`, `.json`,
`.csv`, `.png`, `.txt` đã cập nhật sang `DRIVE_SYNC_DIR`, giữ cấu trúc thư mục. File đích
được thay qua file tạm; file nguồn thay đổi trong lúc sao chép sẽ được thử lại ở lần
sau. Ứng dụng Google Drive chịu trách nhiệm tải các file trong thư mục đó lên cloud.
Nếu đồng bộ gặp lỗi, checkpoint local vẫn được giữ và notebook thông báo để xử lý.

`DRIVE_SYNC_DIR=None` nghĩa là **chưa bật tự đồng bộ Drive**, không tự đoán ổ đĩa hay
tạo một thư mục Drive giả trên đĩa local. Cần giữ ứng dụng Drive chạy và kiểm tra cloud
đã nhận file mới trước khi bàn giao. Giữ notebook `train.ipynb` trong `notebooks/` khi
đưa source vào GitHub. Không thêm warm-up hoặc đổi scheduler giữa lượt đang resume;
nhóm cần chốt yêu cầu “lấy đà” riêng nếu ý là lịch learning rate mới.

Hướng dẫn nền: [Google Colab — Local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
và [Google Drive for desktop](https://support.google.com/drive/answer/10838124).

## 7. Đưa vào repo, push branch và tạo PR vào dev

Gói ZIP là source snapshot và kết quả của Huy. Khi đưa vào repo làm việc, chỉ áp dụng
sáu file nguồn/hướng dẫn ở mục 1, báo cáo trong `reports/tuan3/vit_lora_huy/` và biểu đồ
trong `image/tuan3/huy/`; giữ nguyên đường dẫn và không dùng cả snapshot đè lên dev mới.
Gói ZIP có `_handoff/Huy_week3_changes.patch` chứa đúng các thay đổi của Huy; có thể dùng
`git apply --check` rồi `git apply` với đường dẫn patch đó thay cho chép từng file.

Trước tiên kiểm tra working tree và lấy dev mới nhất. Nếu đang có việc chưa commit,
giữ lại thay đổi đó trước khi chuyển nhánh.

```bash
git status
git switch dev
git pull --ff-only origin dev
git switch -c thai_van_huy_tuan3
```

Nếu nhánh đã tồn tại trên máy, dùng `git switch thai_van_huy_tuan3` thay lệnh tạo nhánh.
Áp dụng patch của Huy, chạy kiểm tra rồi xem diff:

```bash
python -m unittest tests.test_huy_week3 -v
git add data/prepare_huy_week3.py engine/train_vit_lora_huy.py notebooks/train.ipynb experiments/requirements-huy-week3.txt experiments/HUONG_DAN_TUAN3_HUY.md tests/test_huy_week3.py reports/tuan3/vit_lora_huy image/tuan3/huy
git diff --cached --stat
git diff --cached --check
git commit -m "Add Huy week 3 ViT-LoRA training notebook and resumable experiments"
git push -u origin thai_van_huy_tuan3
```

Trên GitHub → Compare & pull request → **base: dev**, **compare: thai_van_huy_tuan3**.
PR cần mô tả đúng kết quả MRI thực đã kiểm tra, gồm cả mục tiêu DS1 chưa đạt. Báo cáo,
CSV/JSON và biểu đồ của Huy nằm trong `reports/tuan3/vit_lora_huy/` và `image/tuan3/huy/`. Ghi backbone, kích thước ảnh, epoch tốt nhất,
ngưỡng validation, Recall/Specificity/FP/FN, AUC, Mean Box IoU và Pointing Game.
Checkpoint đặt trong đúng thư mục Huy trên Drive; không force-add dataset/checkpoint lên Git.
