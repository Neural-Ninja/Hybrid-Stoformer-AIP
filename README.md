# Hybrid-Stoformer-AIP-Project

This repository contains the implementation of the **Hybrid Stoformer** model for image restoration, as part of the **Course Project for Advanced Image Processing (AIP)** at **IISc, Bangalore**. The model builds on the **Stochastic Window Transformer for Image Restoration (Stoformer)**, originally proposed in the [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8b48442a4d494ce0dcabdb0d5456cb53-Abstract.html), with significant improvements for **faster and better inference** through **parallelized patch processing**.

### Current Work

The focus of this project is on the development of the **Hybrid Stoformer**, a model that combines:
- **ResNet-based CNN** for feature extraction.
- **Stoformer Transformer** for long-range dependency modeling.
- **Deep Q-learning (RL)** to improve the model's performance through reinforcement learning for adaptive image restoration tasks.

## Architecture

The **Hybrid Stoformer** is a hybrid architecture that leverages the strengths of both transformers and CNNs to solve the image restoration problem. It retains the core idea of Stoformer but integrates a **ResNet-based CNN** for better low-level feature extraction and a **Deep Q-learning** framework for learning and improving the restoration process.

### Key Components

#### 1. Stochastic Window Strategy (Stoformer)
The key innovation of the Stoformer model is its **stochastic window partition**:
- **During Training:** Windows are randomly shifted by a stochastic offset (ξh, ξw) ∈ [0,7]×[0,7] for each layer and training batch.
- **During Inference:** A layer expectation propagation algorithm approximates the expectation over all possible window positions.

This method maintains translation invariance and ensures that all local relationships within the image are effectively captured.

#### 2. U-shaped Architecture
The core model follows a U-shaped architecture with:
- **Base Channel Dimension:** 32
- **Window Size:** 8×8
- **Four-level Hierarchy:**
  - Level 1: 1 StoBlock, 32 channels, full resolution
  - Level 2: 2 StoBlocks, 64 channels, 1/2 resolution
  - Level 3: 8 StoBlocks, 128 channels, 1/4 resolution
  - Level 4: 8 StoBlocks, 256 channels, 1/8 resolution
- **Skip Connections:** Each encoder level connects to its corresponding decoder level.

#### 3. Stochastic Transformer Block (StoBlock)
Each StoBlock consists of:
- Layer Normalization
- Stochastic Window Multi-head Self Attention (StoWin-MSA)
- Layer Normalization
- MLP with an expansion ratio of 4
- Residual connections

#### 4. Parallelized Patch Processing for Faster Inference
A key improvement in the Hybrid Stoformer is **parallelized patch processing**. The image is divided into smaller patches, which are processed in parallel, drastically reducing inference time. This is especially beneficial for real-time applications or systems with limited resources.

#### 5. Hybrid Model: ResNet + Stoformer + RL
The hybridization of **ResNet** and **Stoformer** improves performance by combining:
- **ResNet CNNs** for better feature extraction and handling of low-level image details.
- **Stoformer** for maintaining long-range dependencies and capturing complex pixel relationships.
- **Deep Q-learning (RL)** to enable the model to adapt and improve over time based on feedback, which enhances its image restoration capabilities by learning from experience.

## Advantages Over Traditional Transformers

1. **Translation Invariance:** The Hybrid Stoformer preserves translation invariance, a key feature traditionally found in CNNs.
2. **Local and Global Context:** The hybrid model efficiently captures both local pixel relationships (via CNNs) and long-range dependencies (via transformers).
3. **Blocking Artifacts Elimination:** The stochastic window approach eliminates blocking artifacts common in fixed-window transformers.
4. **Better Generalization:** The model generalizes better with fewer artifacts, leading to high-quality restoration across various tasks.
5. **Faster Inference:** With parallelized patch processing, the inference time is significantly reduced, making the model suitable for real-time applications.

## Implementation Details

The model is trained using:
- **Loss Function:** Charbonnier loss (ε=10^-3)
- **Optimizer:** Adam (β1=0.9, β2=0.999)
- **Learning Rate:** 3×10^-4, gradually reduced to 10^-6 using cosine annealing.

---

## Licensing

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Citation

If you use this code in your research, please cite the following works:

**[1]** Li, J., Zhang, Y., Xu, Z., Wang, M., & Li, B. (2022). *Stochastic Window Transformer for Image Restoration*. In NeurIPS 2022. [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8b48442a4d494ce0dcabdb0d5456cb53-Abstract.html)

**[2]** Victor Azad, Ankur Kumar. (2025). *Hybrid Stoformer: A Hybrid Transformer-CNN Model for Image Restoration with Reinforcement Learning*. [GitHub Repository](https://github.com/Neural-Ninja/Hybrid-Stoformer-AIP) *(Accessed: 2025-04-21)*
