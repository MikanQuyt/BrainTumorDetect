# Tuan 2: Zero-shot SAM vs MedSAM voi oracle GT box

## 1. Dice / IoU (hop chuan)

**Dice**
| dataset | method | n_slices | n_cases | slice_mean | slice_sd | slice_median | case_mean | case_sd |
|---|---|---|---|---|---|---|---|---|
| ds4 | SAM | 3313 | 55 | 0.787 | 0.189 | 0.844 | 0.777 | 0.119 |
| ds4 | MedSAM | 3313 | 55 | 0.788 | 0.171 | 0.848 | 0.782 | 0.087 |
| ds4 | Box-fill | 3313 | 55 | 0.665 | 0.143 | 0.695 | 0.664 | 0.086 |
| ds2 | SAM | 2192 | 2192 | 0.848 | 0.115 | 0.883 | 0.848 | 0.115 |
| ds2 | MedSAM | 2192 | 2192 | 0.845 | 0.099 | 0.869 | 0.845 | 0.099 |
| ds2 | Box-fill | 2192 | 2192 | 0.824 | 0.073 | 0.844 | 0.824 | 0.073 |

**IoU**
| dataset | method | slice_mean | slice_sd | slice_median | case_mean |
|---|---|---|---|---|---|
| ds4 | SAM | 0.680 | 0.208 | 0.730 | 0.670 |
| ds4 | MedSAM | 0.678 | 0.194 | 0.736 | 0.670 |
| ds4 | Box-fill | 0.513 | 0.147 | 0.533 | 0.513 |
| ds2 | SAM | 0.751 | 0.150 | 0.791 | 0.751 |
| ds2 | MedSAM | 0.743 | 0.137 | 0.769 | 0.743 |
| ds2 | Box-fill | 0.707 | 0.098 | 0.730 | 0.707 |

## 2. So sanh cap MedSAM - SAM (bootstrap theo ca, Wilcoxon tren trung binh theo ca)

| dataset | metric | n_slices | n_cases | delta_mean | ci95_low | ci95_high | win_rate_medsam | tie_rate | case_delta_mean | cohen_dz_case | wilcoxon_p_case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | DICE | 3313 | 55 | 0.002 | -0.010 | 0.014 | 0.477 | 0.000 | 0.005 | 0.081 | 0.669 |
| ds4 | IOU | 3313 | 55 | -0.002 | -0.017 | 0.012 | 0.477 | 0.000 | -0.000 | -0.003 | 0.558 |
| ds2 | DICE | 2192 | 2192 | -0.003 | -0.007 | 0.002 | 0.468 | 0.000 | -0.003 | -0.026 | <0.001 |
| ds2 | IOU | 2192 | 2192 | -0.007 | -0.013 | -0.002 | 0.468 | 0.000 | -0.007 | -0.055 | <0.001 |

## 3. Theo do ro net cua bien (T1 = mo nhat)

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | do ro net bien | T1_mo_nhat | 1104 | 46 | 0.155 | 0.687 | 0.739 | 0.757 | 0.018 | 0.001 | 0.036 | 0.545 |
| ds4 | do ro net bien | T2_trung_binh | 1104 | 50 | 0.687 | 0.851 | 0.798 | 0.811 | 0.013 | -0.005 | 0.033 | 0.509 |
| ds4 | do ro net bien | T3_net_nhat | 1105 | 44 | 0.851 | 2.325 | 0.824 | 0.798 | -0.026 | -0.037 | -0.015 | 0.376 |
| ds2 | do ro net bien | T1_mo_nhat | 731 | 731 | 0.339 | 0.764 | 0.798 | 0.816 | 0.018 | 0.009 | 0.027 | 0.532 |
| ds2 | do ro net bien | T2_trung_binh | 730 | 730 | 0.765 | 0.933 | 0.845 | 0.842 | -0.004 | -0.011 | 0.004 | 0.449 |
| ds2 | do ro net bien | T3_net_nhat | 731 | 731 | 0.934 | 1.522 | 0.901 | 0.878 | -0.023 | -0.030 | -0.017 | 0.421 |

## 4. Theo kich thuoc u

| dataset | strata_by | stratum | n | n_cases | range_low | range_high | dice_sam | dice_medsam | delta | ci95_low | ci95_high | win_rate_medsam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds4 | kich thuoc u | T1_nho | 1104 | 55 | 50 | 782 | 0.686 | 0.659 | -0.027 | -0.044 | -0.008 | 0.399 |
| ds4 | kich thuoc u | T2_vua | 1104 | 52 | 782 | 2017 | 0.810 | 0.823 | 0.013 | -0.007 | 0.036 | 0.489 |
| ds4 | kich thuoc u | T3_lon | 1105 | 37 | 2018 | 5330 | 0.864 | 0.883 | 0.019 | 0.006 | 0.032 | 0.542 |
| ds2 | kich thuoc u | T1_nho | 731 | 731 | 277 | 2069 | 0.856 | 0.823 | -0.033 | -0.041 | -0.026 | 0.398 |
| ds2 | kich thuoc u | T2_vua | 730 | 730 | 2077 | 4863 | 0.854 | 0.856 | 0.002 | -0.006 | 0.009 | 0.455 |
| ds2 | kich thuoc u | T3_lon | 731 | 731 | 4863 | 25461 | 0.833 | 0.856 | 0.023 | 0.015 | 0.031 | 0.550 |

## 5. Upper-bound lam moc cho cac tuan sau

- **ds4**: mo hinh tot nhat = MedSAM; Dice (TB theo ca) = 0.7822; IoU (TB theo ca) = 0.6703
- **ds2**: mo hinh tot nhat = SAM; Dice (TB theo ca) = 0.8479; IoU (TB theo ca) = 0.7508

_Do ro net bien = gradient Sobel trung binh trong dai hep quanh bien mask chuan tren kenh dau (da chuan hoa theo khoang cuong do); gia tri nho = bien mo._