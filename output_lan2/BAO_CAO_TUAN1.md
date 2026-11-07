# Thống kê và tiền xử lý hai bộ ảnh MRI

**Thái Văn Huy · 23130068**

Số liệu được tính trên hai bộ dữ liệu Kaggle sau khi giải nén. Các bảng dưới đây tổng hợp kết quả từ [dataset_statistics.json](dataset_statistics.json). [Kaggle navoneel](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data) · [Kaggle indk214](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification)

## Số lượng và phân bố nhãn

**Navoneel** gồm 253 ảnh phân loại theo hai thư mục `yes` và `no`. ZIP có 506 tệp ảnh vì sao chép cả hai thư mục vào `brain_tumor_dataset/`. So sánh từng cặp tệp cho thấy hai bản giống nhau theo byte. Lượt xử lý dùng bản `yes/` và `no/` ở gốc thư mục navoneel sau khi giải nén.

| Nhãn | Số ảnh | Tỷ lệ |
| --- | ---: | ---: |
| `yes` · Có u | 155 | 61,265% |
| `no` · Không u | 98 | 38,735% |
| **Tổng** | **253** | **100%** |

**Indk214 classification** có 7.023 ảnh và hai split gốc. Các lớp khá gần nhau về số lượng, trong đó `notumor` chiếm tỷ lệ lớn nhất.

| Nhãn gốc | Training | Testing | Tổng | Tỷ lệ |
| --- | ---: | ---: | ---: | ---: |
| `glioma` | 1.321 | 300 | 1.621 | 23,081% |
| `meningioma` | 1.339 | 306 | 1.645 | 23,423% |
| `notumor` | 1.595 | 405 | 2.000 | 28,478% |
| `pituitary` | 1.457 | 300 | 1.757 | 25,018% |
| **Tổng** | **5.712** | **1.311** | **7.023** | **100%** |

**Indk214 segmentation** có 2.192 ảnh và 2.192 mask cùng kích thước. Bộ này chia thư mục theo ba loại u. Mỗi ảnh ghép với một tệp cùng tên có hậu tố `_mask.png`.

| Nhãn | Cặp ảnh–mask | Tỷ lệ |
| --- | ---: | ---: |
| Glioma | 554 | 25,274% |
| Meningioma | 708 | 32,299% |
| Pituitary tumor | 930 | 42,427% |
| **Tổng** | **2.192** | **100%** |

## Kích thước và định dạng ảnh gốc

| Tập | Chiều rộng, min–max | Chiều cao, min–max | Trung vị rộng × cao | Định dạng |
| --- | ---: | ---: | ---: | --- |
| Navoneel | 150–1.920 px | 168–1.427 px | 278 × 331 px | 251 JPEG, 2 PNG |
| Indk214 classification | 150–1.920 px | 168–1.446 px | 512 × 512 px | 7.023 JPEG xám |
| Indk214 segmentation | 512 px | 512 px | 512 × 512 px | 2.192 PNG xám và 2.192 PNG mask |

Navoneel có 217 ảnh RGB, 30 ảnh xám, 5 ảnh RGBA và 1 ảnh bảng màu. Ở phần classification indk214, 4.742 ảnh có kích thước 512 × 512 px, chiếm 67,52%. Toàn bộ ảnh và mask của phần segmentation có kích thước 512 × 512 px.

## Ảnh trùng và split

Hai ảnh được tính là trùng khi có cùng kích thước và cùng toàn bộ giá trị pixel sau khi đọc ảnh về thang xám. Mỗi cụm giữ một bản để tính cột **Số ảnh lặp thêm**. Thống kê này đo ảnh trùng chính xác ở mức pixel.

| Tập | Cụm trùng | Số ảnh lặp thêm | Cụm có nhãn mâu thuẫn |
| --- | ---: | ---: | ---: |
| Navoneel, sau khi bỏ bản sao thư mục | 22 | 25 | 0 |
| Indk214 classification | 300 | 426 | 0 |
| Indk214 segmentation | 0 | 0 | 0 |

Ở phần classification indk214, 189 nội dung ảnh xuất hiện ở cả hai split. Chúng tương ứng với 263 file train và **216 file test, chiếm 16,48% số file test**. Điểm đánh giá trên split gốc vì thế có thể cao hơn khả năng tổng quát hóa trên ảnh mới. Code giữ split gốc để đối chiếu với dữ liệu nguồn.

## Crop vùng đầu

1. Chuyển bản dùng để dò khung sang ảnh xám, làm mượt bằng Gaussian và ước lượng mức nền từ bốn góc.
2. Tạo ảnh nhị phân theo ngưỡng thích nghi. Phép mở giúp giảm nhiễu và chữ nhỏ, rồi phép đóng nối các khoảng đứt ngắn.
3. Chọn contour ngoài có diện tích lớn nhất. Lấy hình chữ nhật bao contour và thêm biên an toàn bằng 2,5% cạnh ngắn của ảnh, tối thiểu 3 pixel.
4. Cắt ảnh gốc theo khung và lưu PNG. Ở phần segmentation, cắt mask bằng chính khung đó và so số pixel u trước và sau crop.

Khung vùng đầu là ước lượng từ cường độ ảnh. Với ảnh chụp vốn đã sát hoặc có nền nhiễu, khung tính được có thể phủ cả ảnh. Tọa độ box u được tính trực tiếp từ ground-truth mask. Môi trường kiểm tra kết quả dùng OpenCV 4.14.0, NumPy 2.3.5 và Pillow 13.0.0.

| Tập | Đã xử lý | Khung nhỏ hơn ảnh | Giữ nguyên khung | Trung vị diện tích ảnh cắt bỏ |
| --- | ---: | ---: | ---: | ---: |
| Navoneel | 253 | 218 | 35 | 10,388% |
| Indk214 classification | 7.023 | 6.552 | 471 | 23,515% |
| Indk214 segmentation | 2.192 | 2.071 | 121 | 28,057% |
| **Tổng** | **9.468** | **8.841** | **627** | |

Diện tích cắt bỏ được tính bằng `1 - diện tích khung crop / diện tích ảnh gốc`, với trung vị trên cả các ảnh giữ nguyên khung. Không có ca nào phải kích hoạt cơ chế giữ ảnh vì nguy cơ cắt mất pixel u.

## Ground truth và annotation

Mask gốc thường có nền mức xám 3 và vùng u mức 255. Năm mask còn có mức xám trung gian. Ngưỡng `>= 128` tách vùng u và đưa mask xuất về hai mức 0 và 255. Cả 2.192 mask đều có pixel u.

| Nhãn | Pixel u trung vị | Tỷ lệ u trung vị trên ảnh gốc | Kích thước box u trung vị |
| --- | ---: | ---: | ---: |
| Glioma | 4.268 px | 1,628% | 76 × 79 px |
| Meningioma | 4.774 px | 1,821% | 88 × 84 px |
| Pituitary tumor | 1.988,5 px | 0,759% | 52 × 53 px |

Pituitary có nhiều ảnh nhất trong phần segmentation, đồng thời có tỷ lệ pixel u trung vị thấp nhất. Chênh lệch giữa nền và vùng u vì vậy đáng chú ý khi đọc thống kê theo pixel.

Ví dụ với `DATASET/Segmentation/Glioma/enh_1841.png` trong [ds_annotations.json](ds_annotations.json).

| Trường | Giá trị |
| --- | --- |
| Kích thước ảnh gốc | 512 × 512 px |
| Khung crop trên ảnh gốc | `[80, 30, 443, 460]` |
| Box u trên ảnh gốc | `[164, 188, 252, 273]` |
| Box u sau crop | `[84, 158, 172, 243]` |
| Kích thước sau crop | 363 × 430 px |
| Số pixel u trước và sau crop | 5.973 |

Đối chiếu 11.660 PNG xuất ra với ảnh và mask gốc cho thấy tất cả ảnh crop khớp pixel với vùng cắt, ảnh và mask cùng kích thước, và cả 2.192 box u trước và sau crop khớp với giới hạn pixel u. Cả 2.192 mask giữ nguyên số pixel u đã xác định sau khi nhị phân hóa.
