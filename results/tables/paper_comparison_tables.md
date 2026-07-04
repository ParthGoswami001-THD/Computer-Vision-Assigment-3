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
| circle_noisy | 1.0000 | 0.4635 | 0.6334 | 144.3401 | 1.0000 | 0.4635 | 0.6334 | 148.4812 |
| u_shape_noisy | 0.9062 | 0.2473 | 0.3886 | 156.5822 | 0.7911 | 0.2156 | 0.3389 | 156.4506 |

## Paper Scf Chen Comparison

| snr_db | chen_precision | chen_recall | chen_f1 | chen_ms | proposed_precision | proposed_recall | proposed_f1 | proposed_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.9000 | 1.0000 | 1.0000 | 1.0000 | 1.9373 | 1.0000 | 1.0000 | 1.0000 | 2.4125 |
| 23.9000 | 1.0000 | 1.0000 | 1.0000 | 1.9658 | 1.0000 | 1.0000 | 1.0000 | 2.1014 |
| 20.3000 | 1.0000 | 1.0000 | 1.0000 | 3.1234 | 1.0000 | 1.0000 | 1.0000 | 2.9925 |

## Paper Vase Method Comparison

| method | input_source | point_accuracy | precision | recall | f1 | elapsed_ms | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Simple Snake-style | data\vase.png | 0.9688 | 0.9677 | 0.2951 | 0.4523 | 171.3421 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Chen-style | data\vase.png | 0.9688 | 0.9480 | 0.9272 | 0.9375 | 4.3262 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
| Proposed SCF | data\vase.png | 0.9688 | 0.9504 | 0.9259 | 0.9380 | 4.7901 | mask_source=otsu_estimated_mask; Snake/Chen are compact approximations. |
