# Tuan 2: Zero-shot SAM vs MedSAM voi oracle GT box

## 1. Dice / IoU (hop chuan)

**Dice**
| dataset | method | n_slices | n_cases | slice_mean | slice_sd | slice_median | case_mean | case_sd |
|---|---|---|---|---|---|---|---|---|
| ds4 | SAM | 3313 | 55 | 0.787 | 0.189 | 0.844 | 0.777 | 0.119 |
| ds4 | MedSAM | 3313 | 55 | 0.788 | 0.171 | 0.848 | 0.782 | 0.087 |
| ds4 | Box-fill | 3313 | 55 | 0.665 | 0.143 | 0.695 | 0.664 | 0.086 |
| ds2 | SAM | 329 | 329 | 0.844 | 0.119 | 0.884 | 0.844 | 0.119 |
| ds2 | MedSAM | 329 | 329 | 0.846 | 0.099 | 0.870 | 0.846 | 0.099 |
| ds2 | Box-fill | 329 | 329 | 0.824 | 0.072 | 0.847 | 0.824 | 0.072 |

**IoU**
| dataset | method | slice_mean | slice_sd | slice_median | case_mean |
|---|---|---|---|---|---|
| ds4 | SAM | 0.680 | 0.208 | 0.730 | 0.670 |
| ds4 | MedSAM | 0.678 | 0.194 | 0.736 | 0.670 |
| ds4 | Box-fill | 0.513 | 0.147 | 0.533 | 0.513 |
| ds2 | SAM | 0.745 | 0.155 | 0.792 | 0.745 |
| ds2 | MedSAM | 0.745 | 0.137 | 0.770 | 0.745 |
| ds2 | Box-fill | 0.707 | 0.097 | 0.735 | 0.707 |

## 2. So sanh cap MedSAM - SAM (bootstrap theo ca, Wilcoxon tren trung binh theo ca)

| dataset | metric | n_slices | n_cases | delta_mean | ci95_low | ci95_high | win_rate_medsam | tie_rate | case_delta_mean | cohen_dz_case | wilcoxon_p_case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | DICE | 3313 | 55 | 0.002 | -0.010 | 0.014 | 0.477 | 0.000 | 0.005 | 0.081 | 0.669 |
| ds4 | IOU | 3313 | 55 | -0.002 | -0.017 | 0.012 | 0.477 | 0.000 | -0.000 | -0.003 | 0.558 |
| ds2 | DICE | 329 | 329 | 0.003 | -0.009 | 0.015 | 0.511 | 0.000 | 0.003 | 0.024 | 0.668 |
| ds2 | IOU | 329 | 329 | -0.000 | -0.015 | 0.015 | 0.511 | 0.000 | -0.000 | -0.001 | 0.609 |

## 3. Theo do ro net cua bien (T1 = mo nhat)

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | do ro net bien | T1_mo_nhat | 1104 | 46 | 0.155 | 0.687 | 0.739 | 0.757 | 0.018 | 0.001 | 0.036 | 0.545 |
| ds4 | do ro net bien | T2_trung_binh | 1104 | 50 | 0.687 | 0.851 | 0.798 | 0.811 | 0.013 | -0.005 | 0.033 | 0.509 |
| ds4 | do ro net bien | T3_net_nhat | 1105 | 44 | 0.851 | 2.325 | 0.824 | 0.798 | -0.026 | -0.037 | -0.015 | 0.376 |
| ds2 | do ro net bien | T1_mo_nhat | 110 | 110 | 0.397 | 0.784 | 0.803 | 0.814 | 0.011 | -0.009 | 0.032 | 0.555 |
| ds2 | do ro net bien | T2_trung_binh | 109 | 109 | 0.786 | 0.956 | 0.839 | 0.850 | 0.011 | -0.011 | 0.036 | 0.514 |
| ds2 | do ro net bien | T3_net_nhat | 110 | 110 | 0.957 | 1.401 | 0.889 | 0.875 | -0.014 | -0.033 | 0.003 | 0.464 |

## 4. Theo kich thuoc u

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | kich thuoc u | T1_nho | 1104 | 55 | 50 | 782 | 0.686 | 0.659 | -0.027 | -0.044 | -0.008 | 0.399 |
| ds4 | kich thuoc u | T2_vua | 1104 | 52 | 782 | 2017 | 0.810 | 0.823 | 0.013 | -0.007 | 0.036 | 0.489 |
| ds4 | kich thuoc u | T3_lon | 1105 | 37 | 2018 | 5330 | 0.864 | 0.883 | 0.019 | 0.006 | 0.032 | 0.542 |
| ds2 | kich thuoc u | T1_nho | 110 | 110 | 291 | 2148 | 0.852 | 0.826 | -0.026 | -0.048 | -0.005 | 0.418 |
| ds2 | kich thuoc u | T2_vua | 109 | 109 | 2188 | 5014 | 0.855 | 0.854 | -0.001 | -0.021 | 0.022 | 0.440 |
| ds2 | kich thuoc u | T3_lon | 110 | 110 | 5061 | 19638 | 0.824 | 0.859 | 0.035 | 0.016 | 0.054 | 0.673 |

## 5. Upper-bound lam moc cho cac tuan sau

- **ds4**: mo hinh tot nhat = MedSAM; Dice (TB theo ca) = 0.7822; IoU (TB theo ca) = 0.6703
- **ds2**: mo hinh tot nhat = MedSAM; Dice (TB theo ca) = 0.8463; IoU (TB theo ca) = 0.7451

_Do ro net bien = gradient Sobel trung binh trong dai hep quanh bien mask chuan tren kenh dau (da chuan hoa theo khoang cuong do); gia tri nho = bien mo._