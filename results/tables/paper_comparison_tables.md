# Paper Comparison Tables

These tables are presentation-ready summaries derived from `paper_comparison_results_raw.csv`.

## Paper Initial Point Comparison

| case | yuen_true_points | yuen_total_points | yuen_accuracy | yuen_neighbor_mean | yuen_neighbor_std | ieps_true_points | ieps_total_points | ieps_accuracy | ieps_neighbor_mean | ieps_neighbor_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| circle_noisy | 32 | 32 | 1.0000 | 13.8479 | 0.3333 | 32 | 32 | 1.0000 | 13.8614 | 0.3333 |
| u_shape_noisy | 32 | 32 | 1.0000 | 15.5970 | 6.4359 | 32 | 32 | 1.0000 | 16.3217 | 4.1277 |

## Paper Snake Initialization Comparison

| case | yuen_snake_precision | yuen_snake_recall | yuen_snake_f1 | yuen_snake_ms | ieps_snake_precision | ieps_snake_recall | ieps_snake_f1 | ieps_snake_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| circle_noisy | 1.0000 | 0.4635 | 0.6334 | 169.3601 | 1.0000 | 0.4635 | 0.6334 | 168.9994 |
| u_shape_noisy | 1.0000 | 0.3708 | 0.5410 | 166.2677 | 1.0000 | 0.3818 | 0.5526 | 176.2541 |

## Paper Scf Chen Comparison

| snr_db | chen_precision | chen_recall | chen_f1 | chen_ms | proposed_precision | proposed_recall | proposed_f1 | proposed_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.9000 | 1.0000 | 1.0000 | 1.0000 | 2.5131 | 1.0000 | 1.0000 | 1.0000 | 3.8076 |
| 23.9000 | 1.0000 | 1.0000 | 1.0000 | 2.4978 | 1.0000 | 1.0000 | 1.0000 | 3.0059 |
| 20.3000 | 1.0000 | 1.0000 | 1.0000 | 2.5744 | 1.0000 | 1.0000 | 1.0000 | 2.6930 |

## Paper Vase Method Comparison

| method | input_source | point_accuracy | precision | recall | f1 | elapsed_ms | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Simple Snake-style | data\vase.png | 0.9688 | 0.9677 | 0.2951 | 0.4523 | 164.0350 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Chen-style | data\vase.png | 0.9688 | 0.9480 | 0.9272 | 0.9375 | 4.0987 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Proposed SCF | data\vase.png | 0.9688 | 0.9504 | 0.9259 | 0.9380 | 3.9475 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
