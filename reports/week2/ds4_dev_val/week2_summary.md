# Tuan 2: Zero-shot SAM vs MedSAM voi oracle GT box

## 1. Dice / IoU (hop chuan)

**Dice**
| dataset | method | n_slices | n_cases | slice_mean | slice_sd | slice_median | case_mean | case_sd |
|---|---|---|---|---|---|---|---|---|
| ds4dev | SAM | 3433 | 55 | 0.791 | 0.190 | 0.851 | 0.786 | 0.097 |
| ds4dev | MedSAM | 3433 | 55 | 0.766 | 0.177 | 0.828 | 0.760 | 0.088 |
| ds4dev | Box-fill | 3433 | 55 | 0.664 | 0.148 | 0.708 | 0.659 | 0.081 |
| ds4_own | SAM | 3313 | 55 | 0.787 | 0.189 | 0.844 | 0.777 | 0.119 |
| ds4_own | MedSAM | 3313 | 55 | 0.788 | 0.171 | 0.848 | 0.782 | 0.087 |
| ds4_own | Box-fill | 3313 | 55 | 0.665 | 0.143 | 0.695 | 0.664 | 0.086 |

**IoU**
| dataset | method | slice_mean | slice_sd | slice_median | case_mean |
|---|---|---|---|---|---|
| ds4dev | SAM | 0.687 | 0.208 | 0.741 | 0.680 |
| ds4dev | MedSAM | 0.649 | 0.197 | 0.706 | 0.641 |
| ds4dev | Box-fill | 0.513 | 0.150 | 0.548 | 0.509 |
| ds4_own | SAM | 0.680 | 0.208 | 0.730 | 0.670 |
| ds4_own | MedSAM | 0.678 | 0.194 | 0.736 | 0.670 |
| ds4_own | Box-fill | 0.513 | 0.147 | 0.533 | 0.513 |

## 2. So sanh cap MedSAM - SAM (bootstrap theo ca, Wilcoxon tren trung binh theo ca)

| dataset | metric | n_slices | n_cases | delta_mean | ci95_low | ci95_high | win_rate_medsam | tie_rate | case_delta_mean | cohen_dz_case | wilcoxon_p_case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4dev | DICE | 3433 | 55 | -0.025 | -0.036 | -0.014 | 0.366 | 0.000 | -0.026 | -0.601 | <0.001 |
| ds4dev | IOU | 3433 | 55 | -0.038 | -0.051 | -0.025 | 0.366 | 0.000 | -0.039 | -0.739 | <0.001 |
| ds4_own | DICE | 3313 | 55 | 0.002 | -0.010 | 0.014 | 0.477 | 0.000 | 0.005 | 0.081 | 0.669 |
| ds4_own | IOU | 3313 | 55 | -0.002 | -0.017 | 0.012 | 0.477 | 0.000 | -0.000 | -0.003 | 0.558 |

## 3. Theo do ro net cua bien (T1 = mo nhat)

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4dev | do ro net bien | T1_mo_nhat | 1144 | 46 | 0.244 | 0.675 | 0.756 | 0.745 | -0.011 | -0.030 | 0.006 | 0.458 |
| ds4dev | do ro net bien | T2_trung_binh | 1144 | 50 | 0.676 | 0.862 | 0.801 | 0.774 | -0.027 | -0.042 | -0.011 | 0.348 |
| ds4dev | do ro net bien | T3_net_nhat | 1145 | 43 | 0.862 | 1.982 | 0.817 | 0.779 | -0.038 | -0.052 | -0.023 | 0.293 |
| ds4_own | do ro net bien | T1_mo_nhat | 1104 | 46 | 0.155 | 0.687 | 0.739 | 0.757 | 0.018 | 0.001 | 0.036 | 0.545 |
| ds4_own | do ro net bien | T2_trung_binh | 1104 | 50 | 0.687 | 0.851 | 0.798 | 0.811 | 0.013 | -0.005 | 0.033 | 0.509 |
| ds4_own | do ro net bien | T3_net_nhat | 1105 | 44 | 0.851 | 2.325 | 0.824 | 0.798 | -0.026 | -0.037 | -0.015 | 0.376 |

## 4. Theo kich thuoc u

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4dev | kich thuoc u | T1_nho | 1144 | 55 | 50 | 850 | 0.675 | 0.632 | -0.043 | -0.061 | -0.024 | 0.341 |
| ds4dev | kich thuoc u | T2_vua | 1144 | 52 | 851 | 2135 | 0.824 | 0.799 | -0.024 | -0.041 | -0.009 | 0.371 |
| ds4dev | kich thuoc u | T3_lon | 1145 | 37 | 2138 | 5051 | 0.876 | 0.867 | -0.008 | -0.024 | 0.007 | 0.388 |
| ds4_own | kich thuoc u | T1_nho | 1104 | 55 | 50 | 782 | 0.686 | 0.659 | -0.027 | -0.044 | -0.008 | 0.399 |
| ds4_own | kich thuoc u | T2_vua | 1104 | 52 | 782 | 2017 | 0.810 | 0.823 | 0.013 | -0.007 | 0.036 | 0.489 |
| ds4_own | kich thuoc u | T3_lon | 1105 | 37 | 2018 | 5330 | 0.864 | 0.883 | 0.019 | 0.006 | 0.032 | 0.542 |

## 5. Upper-bound lam moc cho cac tuan sau

- **ds4dev**: mo hinh tot nhat = SAM; Dice (TB theo ca) = 0.7857; IoU (TB theo ca) = 0.6799
- **ds4_own**: mo hinh tot nhat = MedSAM; Dice (TB theo ca) = 0.7822; IoU (TB theo ca) = 0.6703

_Do ro net bien = gradient Sobel trung binh trong dai hep quanh bien mask chuan tren kenh dau (da chuan hoa theo khoang cuong do); gia tri nho = bien mo._