# Khảo sát BraTS2020 (tập training, 369 ca)

## Cấu trúc và kích thước
- Mỗi ca gồm 4 modality (FLAIR, T1, T1ce, T2) và mask phân vùng seg, định dạng NIfTI 3D.
- Toàn bộ 369 ca: kích thước 240x240x155, spacing 1x1x1 mm, hướng trục LPS, không ca nào khác.
- Không ca nào thiếu modality hay seg; shape giữa các modality luôn đồng nhất.
- => Không cần resize hay đổi hướng; cắt lát ngang theo trục thứ ba (z).

## Lát cắt
- 155 lát/ca. Lát có não: 98-149 lát/ca (trung bình 137.9); lát rỗng trung bình 17.1/ca.
- Ca ít lát não nhất: 366 (98), 360 (100), 357 (102), 349 (105), 150 (117).

## Nhãn và tổn thương
- Tập nhãn: {0,1,2,4}, không có nhãn lạ (nhãn 3 không tồn tại trong BraTS2020).
- 340 ca đủ nhãn 1,2,4; 27 ca không có nhãn 4; 2 ca thiếu nhãn 1 hoặc nhãn 2.
- Mask Whole Tumor (WT) = seg > 0, đúng cho cả các ca thiếu nhãn.
- Số lát có u: 23-102/ca (trung vị 68). Thể tích WT: 7.3-361.8 cm3 (trung vị 90.7).
- Diện tích u lớn nhất trên một lát: 404-6444 px => bounding box có kích thước rất đa dạng.
- Ca u nhỏ nhất: 079 (7.3 cm3), 099, 082, 341, 177.

## Tỷ lệ lát u / lát lành
- Tổng lát có u: 24,422; lát có não không u: 26,476; tỷ lệ u = 0.480 (gần cân bằng sẵn).

## Voxel u nằm ngoài vùng khác 0 của cả 3 modality
- 36 ca; trung vị 6.5 voxel; lớn nhất 588 voxel (ca 335, khoảng 0.43% thể tích WT của ca đó).
- Ảnh hưởng không đáng kể; lát có u luôn được giữ.

## Quyết định cho pipeline
- Z-score theo từng volume, từng modality, trên voxel khác 0; nền giữ bằng 0.
- Chia train/val/test theo ca, không chia theo lát.
- Cân bằng u : lành ở tập train; luôn giữ mọi lát có u.