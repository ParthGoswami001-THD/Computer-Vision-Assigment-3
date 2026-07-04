# Real Vase Test Input

Place the real vase image here as:

```text
data/vase.png
```

Accepted alternatives are `data/vase.jpg`, `data/vase.jpeg`, `data/vase.webp`, and `data/vase.png.webp`.

Optional hand-labeled binary mask:

```text
data/vase_mask.png
```

If `vase_mask.png` is missing, the project estimates a proxy mask using Otsu thresholding and the largest plausible object component. Those metrics should be described as proxy/qualitative results, not true ground-truth evaluation.

If no local vase image is found, `main.py` still runs by using the synthetic vase-like fallback case and marks it as:

```text
synthetic_fallback_missing_real_vase_image
```
