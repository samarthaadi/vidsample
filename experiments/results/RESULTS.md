# vidsample experiments: shortcut test and CPU baseline

## Data

| dataset | train real/fake | test real/fake | ood real/fake |
|---|---|---|---|
| aigvdbench | 300/300 | 248/160 | 248/88 |
| genbuster200k | 300/300 | 100/100 | 120/120 |
| genvidbench | 279/281 | 121/119 | 100/100 |
| genvideo100k | 293/297 | 107/103 | – |

Clips without usable features (dropped): {'meta_raw': 0, 'clip_raw': 0, 'meta_norm': 0, 'clip_norm': 0}

## 1. Within-dataset AUC (train → in-distribution test / unseen generators)

| dataset | meta raw | meta norm | CLIP raw | CLIP norm |
|---|---|---|---|---|
| aigvdbench | 1.000 / 0.989 | 0.665 / 0.563 | 0.880 / 0.862 | 0.780 / 0.824 |
| genbuster200k | 0.208 / 0.143 | 0.211 / 0.222 | 0.977 / 0.989 | 0.963 / 0.985 |
| genvidbench | 1.000 / 1.000 | 0.753 / 0.728 | 0.997 / 0.747 | 0.983 / 0.706 |
| genvideo100k | 1.000 | 0.636 | 0.999 | 0.997 |

Cells are `test AUC / ood AUC`. Balanced accuracy at threshold 0.5:

| dataset | meta raw | meta norm | CLIP raw | CLIP norm |
|---|---|---|---|---|
| aigvdbench | 1.000 / 0.989 | 0.631 / 0.551 | 0.801 / 0.783 | 0.704 / 0.754 |
| genbuster200k | 0.275 / 0.217 | 0.265 / 0.287 | 0.930 / 0.938 | 0.900 / 0.929 |
| genvidbench | 1.000 / 1.000 | 0.700 / 0.730 | 0.971 / 0.695 | 0.925 / 0.665 |
| genvideo100k | 1.000 | 0.617 | 0.981 | 0.976 |

## 2. Best single metadata feature (AUC on the training set)

| dataset | raw | normalized |
|---|---|---|
| aigvdbench | 1.000 (aspect/height/width) | 0.650 (duration) |
| genbuster200k | 0.650 (bitrate/bits_per_pixel/file_bytes) | 0.616 (bitrate/bits_per_pixel/file_bytes) |
| genvidbench | 1.000 (has_audio) | 0.662 (bitrate/bits_per_pixel) |
| genvideo100k | 1.000 (has_audio) | 0.606 (duration) |

## 3. Cross-dataset AUC (metadata, raw): rows = trained on, columns = tested on

| train \ test | aigvdbench | genbuster200k | genvidbench | genvideo100k |
|---|---|---|---|---|
| aigvdbench | 1.000 | 0.500 | 0.500 | 0.500 |
| genbuster200k | 0.076 | 0.208 | 0.168 | 0.473 |
| genvidbench | 0.500 | 0.500 | 1.000 | 1.000 |
| genvideo100k | 0.500 | 0.500 | 1.000 | 1.000 |
| all | 1.000 | 0.248 | 1.000 | 1.000 |
| all but column | 0.816 | 0.863 | 0.997 | 0.873 |

## 3. Cross-dataset AUC (CLIP, raw): rows = trained on, columns = tested on

| train \ test | aigvdbench | genbuster200k | genvidbench | genvideo100k |
|---|---|---|---|---|
| aigvdbench | 0.880 | 0.689 | 0.811 | 0.449 |
| genbuster200k | 0.586 | 0.977 | 0.864 | 0.823 |
| genvidbench | 0.617 | 0.917 | 0.997 | 0.971 |
| genvideo100k | 0.450 | 0.829 | 0.925 | 0.999 |
| all | 0.803 | 0.854 | 0.978 | 0.965 |
| all but column | 0.621 | 0.791 | 0.953 | 0.800 |

## 3. Cross-dataset AUC (CLIP, normalized): rows = trained on, columns = tested on

| train \ test | aigvdbench | genbuster200k | genvidbench | genvideo100k |
|---|---|---|---|---|
| aigvdbench | 0.780 | 0.660 | 0.717 | 0.462 |
| genbuster200k | 0.576 | 0.963 | 0.828 | 0.823 |
| genvidbench | 0.603 | 0.857 | 0.983 | 0.951 |
| genvideo100k | 0.492 | 0.796 | 0.896 | 0.997 |
| all | 0.721 | 0.837 | 0.943 | 0.962 |
| all but column | 0.606 | 0.753 | 0.859 | 0.839 |

## 4. Per-generator accuracy (CLIP normalized, within-dataset, threshold 0.5)

**aigvdbench – test**: AccVideo_T2V 0.50, AnimateDiff_T2V 0.88, Cogvideox1.5_T2V 0.62, Cogvideox1.5_V2V 0.50, EasyAnimate_I2V 0.75, EasyAnimate_T2V 0.50, HunyuanVideo_T2V 0.88, IPOC_T2V 0.88, LTX_I2V 0.50, LTX_T2V 0.50, LTX_V2V 0.50, Open-Sora_T2V 1.00, Pyramid-Flow_I2V 0.50, Pyramid-Flow_T2V 0.88, RepVideo_T2V 0.50, SEINE_I2V 0.62, SVD_I2V 0.62, VideoCrafter_I2V 0.38, VideoCrafter_T2V 1.00, Wan2.1_T2V 0.75, real 0.75  
**aigvdbench – ood**: Gen2 0.50, Gen3 0.75, Jimeng 1.00, Luma 0.75, Opensora 0.75, Sora 0.62, causvid_24fps 0.88, kling 0.75, pika 0.75, vidu 0.75, wan 0.88, real 0.75  
**genbuster200k – test**: cogvideox 0.84, easyanimate 0.92, hunyuanvideo 0.80, ltxvideo 0.84, real 0.95  
**genbuster200k – ood**: gen3 0.87, jimeng 0.93, kling 0.93, luma 0.93, pika 1.00, sora 1.00, vidu 1.00, wanx 0.93, real 0.91  
**genvidbench – test**: ms 0.86, pika 0.89, t2vz 1.00, vc2 0.93, vript 0.93  
**genvidbench – ood**: cogvideo 0.80, mora 0.80, musev 0.20, svd 0.20, hdvg 0.83  
**genvideo100k – test**: DynamicCrafter 1.00, I2VGEN_XL 1.00, Latte 0.90, OpenSora 0.90, SD 1.00, SEINE 1.00, SVD 1.00, VideoCrafter 1.00, ZeroScope 1.00, pika 1.00, real 0.97  
