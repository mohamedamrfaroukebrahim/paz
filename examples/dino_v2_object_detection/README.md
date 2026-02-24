# DINOv2 Object Detection

This example demonstrates how to use [DINOv2](https://arxiv.org/abs/2304.07193)
as a backbone for object detection within the `paz` framework.

DINOv2 is a self-supervised Vision Transformer model trained with the DINO
method. It produces rich visual features that can be used for downstream tasks
including object detection.

## Models

Three model variants are provided, corresponding to DINOv2 ViT-S, ViT-B, and
ViT-L backbones:

| Model        | Embed dim | Heads | Depth | Patch size |
|--------------|-----------|-------|-------|------------|
| DINOv2ViTS   | 384       | 6     | 12    | 14         |
| DINOv2ViTB   | 768       | 12    | 12    | 14         |
| DINOv2ViTL   | 1024      | 16    | 24    | 14         |

## Usage

```python
from paz.models.detection.dino_v2 import DINOv2ViTS

model = DINOv2ViTS(num_classes=80, input_shape=(560, 560, 3))
model.summary()
```

## Demo

```bash
python demo.py
```

## Train

```bash
python train.py
```

## Test

```bash
pytest dino_v2_test.py
```

## References

- [DINOv2: Learning Robust Visual Features without Supervision](
  https://arxiv.org/abs/2304.07193)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition
  at Scale](https://arxiv.org/abs/2010.11929)
