# Stoformer Models

This directory contains implementations of Stoformer model architectures for image restoration tasks.

## Model Variants

### Stoformer
Implementation of Stoformer that uses a stochastic shifting window mechanism for attention operations. This provides better performance on some tasks by introducing randomness in the attention window positions during training, which helps the model generalize better to real-world scenarios.

### Stoformer with different Loss Functions

    1. Charbonnier Loss
    2. Huber Loss
    3. SSIM Loss

### Stoformer with Fast Infernece
Alternative implementation of Stoformer which uses effective sampling strategy to reduce the inference time.

### Stoformer with Q learning based Stochastic Window Shifting
Alternative implementation of Stoformer using Q learning based approach to learn a Policy for choosing the effective Window shifting.
{In Progress}

## Usage

To use these models in your code:

```python
# Original Stoformer
from Models.stoformer import build_stoformer

model = build_stoformer(img_size=256, window_size=8)

# Stoformer2
from Models.stoformer2 import build_stoformer2

model2 = build_stoformer2(img_size=256, window_size=8)
```

## Testing

To test a trained model on a single image:

```bash
# Test with original Stoformer
python Test/test.py --image_path path/to/input.png --gt_path path/to/groundtruth.png \
    --checkpoint_path path/to/checkpoint.pth --model_type stoformer

# Test with Stoformer2
python Test/test.py --image_path path/to/input.png --gt_path path/to/groundtruth.png \
    --checkpoint_path path/to/checkpoint.pth --model_type stoformer2
```

## Model Architecture

Both models follow a U-Net-like architecture with transformer blocks. Key differences:

- **Stoformer**: Uses fixed window attention with a focus on local patterns.
- **Stoformer2**: Implements stochastic window shifting during training to improve feature learning and generalization.

The models have the following components:
- Encoder: Progressively downsamples features while increasing channels
- Bottleneck: Processes deep features at lowest resolution
- Decoder: Upsamples and combines features through skip connections
- Input/Output projections: Convert between image and feature space 