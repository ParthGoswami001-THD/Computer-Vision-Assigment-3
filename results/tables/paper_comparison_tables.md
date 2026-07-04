# Paper Comparison Tables

These tables are presentation-ready summaries derived from `paper_comparison_results.csv`.

## Paper Initial Point Comparison

| case | yuen_true_points | yuen_total_points | yuen_accuracy | yuen_neighbor_mean | yuen_neighbor_std | ieps_true_points | ieps_total_points | ieps_accuracy | ieps_neighbor_mean | ieps_neighbor_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| circle_noisy | 32 | 32 | 1.0000 | 13.8479 | 0.3333 | 32 | 32 | 1.0000 | 13.8614 | 0.3333 |
| u_shape_noisy | 29 | 32 | 0.9062 | 18.7547 | 10.3325 | 25 | 32 | 0.7812 | 19.3102 | 5.5726 |

## Paper Snake Initialization Comparison

| case | yuen_snake_precision | yuen_snake_recall | yuen_snake_f1 | yuen_snake_ms | ieps_snake_precision | ieps_snake_recall | ieps_snake_f1 | ieps_snake_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| circle_noisy | 1.0000 | 0.4635 | 0.6334 | 180.3349 | 1.0000 | 0.4635 | 0.6334 | 181.4414 |
| u_shape_noisy | 0.9062 | 0.2473 | 0.3886 | 185.3135 | 0.7911 | 0.2156 | 0.3389 | 198.6006 |

## Paper Scf Chen Comparison

| snr_db | chen_precision | chen_recall | chen_f1 | chen_ms | proposed_precision | proposed_recall | proposed_f1 | proposed_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.9000 | 1.0000 | 1.0000 | 1.0000 | 4.3004 | 1.0000 | 1.0000 | 1.0000 | 5.3153 |
| 23.9000 | 1.0000 | 1.0000 | 1.0000 | 4.6917 | 1.0000 | 1.0000 | 1.0000 | 5.1876 |
| 20.3000 | 1.0000 | 1.0000 | 1.0000 | 3.4274 | 1.0000 | 1.0000 | 1.0000 | 3.5727 |

## Paper Vase Method Comparison

| method | input_source | point_accuracy | precision | recall | f1 | elapsed_ms | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Simple Snake-style | data\vase.png | 0.5625 | 0.5813 | 0.2194 | 0.3185 | 205.0206 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Chen-style | data\vase.png | 0.5625 | 0.5563 | 0.7922 | 0.6536 | 7.1028 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Proposed SCF | data\vase.png | 0.5625 | 0.5569 | 0.7922 | 0.6541 | 5.9226 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
