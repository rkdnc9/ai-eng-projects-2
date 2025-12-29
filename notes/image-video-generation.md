# Image and Video Generation

## Table of Contents

* [Overview](#overview)
* [Generative Modeling Techniques](#generative-modeling-techniques)
  * [VAE (Variational Autoencoders)](#vae-variational-autoencoders)
  * [GAN (Generative Adversarial Networks)](#gan-generative-adversarial-networks)
  * [Autoregressive Models](#autoregressive-models)
  * [Diffusion Models](#diffusion-models)
  * [Technique Comparison](#technique-comparison)
* [Text-to-Image Models](#text-to-image-models)
  * [Evolution and Current Landscape](#evolution-and-current-landscape)
  * [Data Preparation](#data-preparation)
  * [Model Architecture: UNET vs DiT](#model-architecture-unet-vs-dit)
  * [Training Process](#training-process)
  * [Diffusion Sampling](#diffusion-sampling)
* [Evaluation Methods](#evaluation-methods)
  * [Inception Score](#inception-score)
  * [FID (Fréchet Inception Distance)](#fid-fréchet-inception-distance)
  * [CLIP Similarity Score](#clip-similarity-score)
* [Text-to-Video Models](#text-to-video-models)
  * [Current Landscape](#current-landscape)
  * [Latent Diffusion Models](#latent-diffusion-models)
  * [Building Text-to-Video Models](#building-text-to-video-models)
  * [Production System Design](#production-system-design)
* [Key Insights](#key-insights)
  * [Fundamental Principles](#fundamental-principles)
  * [Important Distinctions and Comparisons](#important-distinctions-and-comparisons)
  * [Practical Considerations](#practical-considerations)
  * [Common Pitfalls and Misconceptions](#common-pitfalls-and-misconceptions)
* [Quick Recall / Implementation Checklist](#quick-recall--implementation-checklist)
* [References](#references)

---

**TL;DR**: Modern image and video generation uses diffusion models as the industry standard. Four main approaches exist (VAE, GAN, Autoregressive, Diffusion) - diffusion wins on quality, VAEs survive as compression networks in latent diffusion models. DiT (Diffusion Transformer) has replaced UNET for scalability. Text-to-video extends text-to-image to 3D (adding temporal dimension), requires latent space training for efficiency (512x compression via VAE), and uses cascaded super-resolution. Production systems need safety guardrails, prompt enhancement, and multiple post-processing stages. Quality metrics: FID for image quality, CLIP score for text-image alignment.

---

## Overview

This lecture provides a comprehensive foundation in image and video generation, covering the evolution from early generative modeling techniques (VAEs and GANs) to modern state-of-the-art diffusion models. The focus spans four major generative approaches, detailed text-to-image model architecture and training, evaluation methodologies, and the extension to text-to-video generation. Practical demonstrations compare leading models (ChatGPT, Gemini, Flux) and highlight challenges in diagram generation, text rendering, and video physics. The lecture concludes with production system design for deploying multimodal generation agents.

## Generative Modeling Techniques

![Diagram 1](https://kroki.io/mermaid/svg/eNp1kM1qwzAQhO99in0BQ5rSiw8F_yUt2A3Fbi7CB2FtbIEiufpp8NvXjkqxDNVxZna_1fSajgM0-QPMLyFHlKip5d8IlWIouOyhwW6Q_MuhaSGKXiAl56Ro_cBdyMgxeV8LOUmcVRp7jcbMu9ZeQXJ-uTjDlfRy6pc-kg9HBbdTDKW6QQSpcFpPQWZP6hGRxXBGPcGBGhvYTySbR1Ba-DQYQ6au4_0AJYFLKPPKpzN_9ApYIePuGrj_oX7tDaqkukcxQY4zsaMWmU_nvo4V6pX3Q-CFoFqoW2BvQKeuo8uHqPCpwne6AaCxc4H1qUmC1B9qocB-Fz3vwFgcTZDaEN8kc8Yup1kqGdWs_QEWH6Kl)

Four major techniques have emerged for generative modeling: VAE, GAN, Autoregressive, and Diffusion. Each offers different trade-offs between quality, speed, and conditioning flexibility. Modern production systems primarily use diffusion models, though VAEs remain critical as compression networks.

### VAE (Variational Autoencoders)

**History and Origins:**
* Introduced in 2013 paper "Auto-encoding Variational Bayes"
* Earlier versions existed in 1980s
* Among the first neural generative models

**Architecture:**

![Diagram 2](https://kroki.io/mermaid/svg/eNpNj8sKg0AMRff9ivyAINKFq4LvZylY6Ca4GDRYoY4yjtD262szKM4qnHPJ5HZKTE8oqxOsz8NMTouGbBAdgeO-HbcGy7qAj5FsxpZUzTmfYYBp37YkoRQfUnC2bWMDtiGWQpPU4Gw8ZB7hlYQEJzzCGB9C9UI2tIuIRYJ3MUwvgq-hsaE8JzynGNLhtJRhtp22f54xzze-H5szL_C26H9z05lNwabEal0vZ62WRvfjWnacZxPwTOAHLZhSLg==)

* **Encoder**: Input (e.g., 28x28=784 pixels) → Hidden layers (400 → 200) → Latent space
  * Outputs two vectors: mean and log-variance (e.g., 2D each)
  * Learns compressed representation of input distribution
* **Reparameterization trick**: Sample z = mean + variance * epsilon (epsilon ~ N(0,1))
  * Enables backpropagation through sampling operation
* **Decoder**: Latent vector z → Hidden layers (200 → 400) → Reconstructed output (784 pixels)
  * Learns to reconstruct original input from compressed representation

**Training Process:**
1. Pass image through encoder to get mean and log-variance
2. Sample latent vector z using reparameterization trick
3. Pass z through decoder to reconstruct image
4. Calculate loss: Binary cross-entropy or MSE between input and reconstruction
5. Update weights via backpropagation

**Sampling/Generation:**
* Sample random z from latent space (e.g., z ~ N(0,1) for 2D latent)
* Pass through decoder
* Output: Generated image

**Example Implementation (MNIST 28x28):**
* Input: 784 pixels
* Encoder: 784 → 400 → 200
* Latent: 2D (mean and variance)
* Decoder: 2 → 200 → 400 → 784
* Loss: MSE between input and reconstruction

**Characteristics:**
* **Quality**: Low - produces blurry, low-resolution images
* **Speed**: Very fast - single forward pass through decoder
* **Training stability**: Stable, straightforward loss function
* **Conditioning**: Limited flexibility for text/class conditioning

**Current Status:**
* No longer used for direct image generation
* **Critical role in latent diffusion models**: VAE encoder compresses images to latent space, VAE decoder reconstructs pixel images
* Example: Stable Diffusion uses VAE for 8x spatial compression

> **Key Insight**: VAEs are no longer competitive for image generation quality, but they survive as essential compression networks in latent diffusion models. Every modern text-to-video model uses a VAE encoder-decoder pair to make training computationally feasible.

### GAN (Generative Adversarial Networks)

**History and Origins:**
* Introduced by Ian Goodfellow at Google
* One of the most cited papers in machine learning (hundreds of thousands of citations)
* Revolutionized generative modeling in mid-2010s

**Architecture:**

![Diagram 3](https://kroki.io/mermaid/svg/eNpVkMEOgjAMhu8-RV-AF-CgEQYDYjwQPS0cFiiwCIxsJAaN7y50ErWnP_2-tkkbI8cWLmwHSx1FLodK93DWyiI8CvC8PQSC44BGTtoAL0gMCIQiljeEtJcNuj4TOcru0yEnEkzZ0qheDbSAOTF0kHJEOX7S6KKsSw8vQjEhLk7aWqjXcR_CTlqr6hlKbQyWUzcXP27ydbkPsdbddpITT8V1rOSEwOCOqmkn62hCNNso_6epewTlzOU3N2dWJw==)

* **Generator G**: Random noise z → Generated (fake) image
  * Learns to create realistic images from random vectors
* **Discriminator D**: Image (real or fake) → Probability (real vs fake)
  * Learns to distinguish real images from generated ones
* **Adversarial training**: G and D compete in a two-player game

**Training Process:**
1. **Train Discriminator**:
   * Pass real images → Label as real
   * Generate fake images with G → Label as fake
   * Loss: Binary cross-entropy for classification
   * Update D weights to improve classification

2. **Train Generator**:
   * Generate fake images
   * Pass through D → Get classification
   * Loss: Fool D (maximize D's probability of classifying fake as real)
   * Update G weights to better fool D

3. **Iterate**: Alternate between updating D and G

**Variants:**
* **StyleGAN / StyleGAN 2 / StyleGAN 3** (NVIDIA): High-quality face generation
  * Famous for thispersondoesnotexist.com (realistic AI-generated faces)
  * Progressive growing and style mixing techniques
* **Conditional GAN**: Add class or text conditioning to generation
* **CycleGAN**: Image-to-image translation without paired data

**Characteristics:**
* **Quality**: Medium - better than VAE, but less than diffusion
* **Speed**: Very fast - single forward pass
* **Training challenges**:
  * Mode collapse: Generator produces limited variety
  * Vanishing gradients: Discriminator becomes too strong, G can't learn
  * Training instability: Requires careful hyperparameter tuning
* **Conditioning**: Possible but complex (conditional GAN architectures)

**Current Status:**
* **Largely deprecated** for mainstream image generation
* Replaced by diffusion models for text-to-image tasks
* Some niche uses: Face generation, style transfer, image-to-image translation

### Autoregressive Models

**Concept:**
* Formulate image generation as **sequence generation**
* Generate pixels or patches sequentially, one at a time
* Each token conditioned on all previous tokens: P(x) = P(x1) * P(x2|x1) * P(x3|x1,x2) * ...

**Approaches:**

**1. Pixel CNN (Google DeepMind, 2016):**
* Paper: "Conditional Image Generation with Pixel CNN Decoders"
* CNN-based architecture
* Generates pixels sequentially using masked convolutions
* Ensures each pixel only sees previous pixels (causal masking)

**2. Image Transformer (Google Brain, 2018):**
* Transformer-based architecture
* Treats image as sequence of patches or pixels
* Self-attention mechanism with causal masking
* More flexible for conditioning on text

**3. DALI 1 (OpenAI, 2021):**
* Paper: "Zero Shot Text-to-Image Generation"
* First successful autoregressive text-to-image model
* 256x256 resolution
* Uses transformer with cross-attention for text conditioning
* Demonstrated high-quality text-conditional generation

**Characteristics:**
* **Quality**: High - coherent global structure, sharp images
* **Speed**: **Very slow** - sequential generation (can't parallelize)
  * Must generate thousands of tokens one by one
* **Conditioning**: Excellent (especially transformer-based)
  * Cross-attention naturally integrates text conditioning
* **Training**: Stable, similar to language models

**Current Status:**
* Occasional use, some hybrid approaches
* **DALI 2 switched to diffusion** (2022), marking industry shift
* Transformers survive in diffusion models (DiT architecture)

### Diffusion Models

**Introduction and Key Papers:**
* **DDPM** (2020, UC Berkeley): "Denoising Diffusion Probabilistic Models"
  * Breakthrough demonstrating high-quality image generation
  * 256x256 resolution, excellent quality
* Became state-of-the-art and **current industry standard**

**Core Concept:**

![Diagram 4](https://kroki.io/mermaid/svg/eNptkMsOgkAMRfd-RX8AMi5cYsL7IRrCws2ExYRpkCiOGYiGvxdLIKjMsqfnTttKi8cF0nwDw7O5e0Nxh7gRFUJnsQIMYw8Ot6WEk6pbLKjPobLLP6V-7t4xNmKXsEfWUWlcqh4xn59R9zD6nbVlk-oTDnimUdZltzQDQiHPsVHPr8yQSMRTbNs5c54mIhqvRcaEkpXIhMjh7x7T-mag9EtoCZlW5fCtaYA3TWk6orz-wOQNGxpi2g==)

Diffusion models learn to reverse a noise-adding process:

**Forward Process (No learning):**
1. Start with clean image (t=0)
2. Gradually add Gaussian noise over T timesteps (e.g., T=1000)
3. End with pure noise (t=T)
4. Fixed noise schedule (e.g., linear or cosine)

**Backward Process (Learned):**
1. Start with pure noise (t=T)
2. Model predicts noise at each timestep
3. Iteratively remove predicted noise
4. End with clean generated image (t=0)

**Why It Works:**
* Denoising is easier than generating from scratch
* Model learns gradual refinement over many steps
* Each step makes small correction, leading to high quality

**Training:**
1. Sample clean image from dataset
2. Sample random timestep t ~ Uniform(1, T)
3. Add noise to image according to timestep t → noisy image
4. Model predicts noise given: noisy image, t, and conditions (text)
5. Loss: MSE between true noise and predicted noise
6. Update model weights

**DALI 2 (OpenAI, 2022):**
* Switched from autoregressive (DALI 1) to diffusion
* 512 - 1K resolution
* Marked paradigm shift in text-to-image field
* Demonstrated superior quality and controllability

**Characteristics:**
* **Quality**: **Highest** - state-of-the-art, photorealistic results
* **Speed**: **Slow** - requires 20-50 iterative denoising steps
  * Trade-off: More steps = higher quality but slower
  * DDIM (2020): Reduced to 50 steps from 1000
  * Modern optimized: 20-50 steps optimal
  * Active research: 1-2 step models (distillation, consistency models)
* **Conditioning**: Excellent - easily add text, class, timestep conditions
* **Training**: Stable, well-understood loss function

**Current Status:**
* **Industry standard** for text-to-image and text-to-video
* 95%+ of modern generative models use diffusion
* All top models: GPT Image 1, Imagen 4, Flux, Sora, Vio 3

### Technique Comparison

| Technique | Quality | Speed | Conditioning | Training | Current Use |
|-----------|---------|-------|--------------|----------|-------------|
| **VAE** | Low (Blurry) | Very Fast | Limited | Stable | Compression in LDM |
| **GAN** | Medium | Very Fast | Complex | Unstable | Deprecated |
| **Autoregressive** | High | Very Slow | Excellent | Stable | Occasional |
| **Diffusion** | Highest (SOTA) | Slow (20-50 steps) | Excellent | Stable | Industry Standard |

**Key Takeaways:**
* **Diffusion dominates**: Quality wins, speed acceptable with optimizations
* **VAE repurposed**: No longer for generation, critical for compression
* **GAN mostly retired**: Training instability and mode collapse issues
* **Autoregressive niche**: High quality but too slow for production

## Text-to-Image Models

### Evolution and Current Landscape

**Historical Progression:**

**2013-2015: VAE Era**
* MNIST (28x28 handwritten digits)
* Blurry, low quality
* Proof of concept only

**2016-2019: GAN Era**
* StyleGAN: High-quality face generation (thispersondoesnotexist.com)
* 256x256, later 1024x1024
* Mode collapse limits diversity

**2020-2021: Early Diffusion**
* DDPM (UC Berkeley): 256x256 high-quality images
* DALI 1 (OpenAI): Autoregressive text-to-image, 256x256
* First impressive text-conditional results

**2022-2023: Diffusion Dominance**
* DALI 2 (OpenAI): Switch to diffusion, 512-1K
* Imagen 1 (Google): 64x64 base model
* Imagen 2 (Google): 1K resolution
* Stable Diffusion (Stability AI): Open-source latent diffusion
* DiT architecture emerges (transformers for diffusion)

**2024-2025: Modern Era**
* **GPT-4O Image** (OpenAI, March 2025)
* **GPT Image 1** (OpenAI, June 2025): Best text rendering
* **Imagen 3** (Google): 1K, multiple aspect ratios
* **Imagen 4 / Imagen 4 Ultra** (Google): Multi-language support, excellent text rendering
* **Flux** (Black Forest Labs): First open model in top 3
* **Sora** (OpenAI, Feb 2024): Text-to-video breakthrough
* Characteristics: Near-perfect text rendering, multiple aspect ratios, 1K+ resolution

**Current Rankings (Artificial Analysis Leaderboard):**
* Closed models dominate top spots
* **Best models**: GPT Image 1, Imagen 4, Flux
* **First open model**: Flux (#3)
* Other notable: Stable Diffusion, Mid-journey, Seedream (ByteDance)

**Practical Comparison Example:**

Prompt: "Taco studying machine learning at sunrise on mountain top"

* **ChatGPT (GPT Image 1)**:
  * Taco with laptop
  * Issue: Laptop facing different direction, physics suboptimal
  * Quality: Good text rendering

* **Gemini (Imagen 4)**:
  * Taco looking at book with "machine learning" text
  * Laptop with diagrams visible
  * Quality: Better physics, taco engaged with content
  * **Verdict**: Best for this prompt

* **Flux (Black Forest Labs)**:
  * Taco with iPad/device
  * Quality: Good composition, taco engaged
  * Note: Taco less realistic than Google's

**Key Observations:**
* Different models produce different styles and quality levels
* Text rendering historically challenging, now largely solved
* Diagram generation still difficult (requires logical structure, accurate text, meaningful arrows)
* More complex prompts reveal greater capability differences

### Data Preparation

> **Key Insight**: Data quality determines model quality - garbage in, garbage out. Text-to-image models require high-quality image-text pairs with accurate, detailed captions. Poor caption quality, misaligned pairs, or inappropriate content will degrade model performance.

**Training Data Format:**
* Image-text pairs: Each image paired with descriptive caption
* Large scale: Lyon 400M (2021, 400 million pairs), Lyon 5B (2022, 5 billion pairs)
* Source: Scraped from internet (websites, social media, stock photos)

**Text/Caption Preparation:**

1. **Handle Missing Captions**:
   * Problem: Many internet images lack captions
   * Solution: Use pre-trained vision-language models (VLMs)
   * Tools: ChatGPT, Kuhn, CLIP-based captioning models
   * Process: Pass image through VLM → Generate descriptive caption

2. **Enhance Existing Captions**:
   * Problem: Captions may be noisy, incomplete, or low-quality
   * Solution: Use VLMs to regenerate detailed descriptions
   * Example: "Cat" → "A fluffy orange tabby cat sitting on a windowsill, looking outside at birds, soft natural lighting from afternoon sun"
   * More detail = better text-image alignment

3. **Remove Poorly Matched Pairs**:
   * Problem: Caption doesn't describe image accurately
   * Solution: Use **CLIP** to measure text-image alignment
   * Process:
     * Encode text and image with CLIP
     * Calculate cosine similarity
     * Filter pairs below threshold (e.g., similarity < 0.3)

**Image Preparation:**

1. **Filter Small Images**:
   * Remove images below resolution threshold (e.g., < 256x256)
   * Low-resolution images hurt model quality

2. **Deduplication**:
   * Problem: Internet contains many duplicates
   * Solution:
     * Extract image embeddings (e.g., CLIP, ResNet features)
     * Cluster embeddings
     * Remove near-duplicates within clusters

3. **Filter Inappropriate Content**:
   * Remove violence, nudity, harmful content, copyrighted material
   * Use pre-trained classifiers or human review
   * Critical for safety and legal compliance

4. **Filter Low Aesthetic Quality**:
   * Problem: Many images are blurry, poorly composed, low quality
   * Solution: Use pre-trained aesthetic models
   * Score images on aesthetic quality → Filter low scores

5. **Adjust Dimensions**:
   * **Resize**: Scale to target resolution (e.g., 512x512, 1024x1024)
   * **Center crop**: Crop to square (or target aspect ratio)
   * **Normalize pixels**: Scale to [0, 1] or [-1, 1], standardize mean and std

**Final Training Data:**
* Clean image-text pairs
* Fixed dimensions (or standardized aspect ratios)
* Detailed, accurate captions
* High aesthetic and content quality
* Ready for model training

### Model Architecture: UNET vs DiT

Two primary architectures for diffusion models: UNET (convolution-based) and DiT (transformer-based).

**UNET Architecture:**

**Origins:**
* 2015 paper: "Convolutional Networks for Biomedical Image Segmentation"
* Originally designed for medical image segmentation, repurposed for diffusion

**Structure:**
* **Downsampling path**: Progressive convolution + downsampling
  * Input: 64x64 → 32x32 → 16x16 → 8x8 → 4x4 (bottleneck)
  * Increasing channel depth at each layer
  * Captures hierarchical features (low-level to high-level)
* **Upsampling path**: Transposed convolution + upsampling
  * 4x4 → 8x8 → 16x16 → 32x32 → 64x64 (output)
  * Decreasing channel depth
  * Reconstructs spatial dimensions
* **Skip connections**: Connect corresponding layers in down/up paths
  * Preserves fine-grained spatial information
  * Helps gradient flow

**Text Conditioning (Added for Diffusion):**
* Cross-attention layers inserted throughout network
* Text encoder (e.g., CLIP) → Text embeddings
* Image features attend to text embeddings
* Not in original UNET paper

**Characteristics:**
* Captures locality well (convolutions)
* Limited scalability with more data/compute
* Used in earlier diffusion models (DALI 2, early Stable Diffusion)

**DiT (Diffusion Transformer) Architecture:**

**Origins:**
* 2023 paper: "Scalable Diffusion Models with Transformers"
* Replaces UNET's convolutions with transformers

**Structure:**

![Diagram 5](https://kroki.io/mermaid/svg/eNptkU1vwjAMhu_7FT7txgGBEJu0SVBavqHS2MnikDWmZGubKgmi_Pu1NtN2WI5-niT269yp-gyH2QO0Z4I7a_wNlqXKCYbjZjg-Qq_3ClNMVcjO5nR7hkEzgNwZfeQrU-YRPgEb5KE_avojIJWdxYjYmGFSqBCogkdInf2kLAieMY7bB-Lyg7Q2Ve5Bm_IlEh4zT3CiNaTWm2BspQqIq8x2rkgJS3M8OFX5k3UlOZgWNvuCvghzFhb4RsWpN-n66N4RtmC2xMhZ738hXE1ok6Hm3uiStRUmRBoS667K3TNYMVn_8_tOhDULG9xfQn0J8HdUETYsbPG9qu85S33L9R2mjrTJQvtxt6Cf1bCyx67FLtOyDrKtVEocETnRUhnyG2KEjp0=)

**1. Patchify (Tokenization):**
* Input: Noisy image (e.g., 48x48 pixels)
* Divide into non-overlapping patches (e.g., 3x3 grid = 9 patches, each 16x16)
* Flatten each patch: 16x16 = 256 values
* Project: Linear layer 256 → C (embedding dimension)
* Output: Sequence of N embeddings (N = number of patches)

**2. Positional Encoding:**
* Add position information to each patch embedding
* Options:
  * **1D**: Position 1, 2, 3, ..., 9 (simple sequence)
  * **2D**: (row, col) like (1,1), (1,2), (1,3), (2,1), ... (preserves spatial structure)
  * **3D** (for video): (row, col, frame) (preserves spatial + temporal)
* Learned or sinusoidal encoding

**3. Transformer Blocks (Repeated N times):**
* **Self-attention**: Patches attend to each other
  * Captures global dependencies
  * Long-range relationships
* **Cross-attention**: Patches attend to text embeddings
  * Conditioning mechanism
  * Text controls generation
* **Feed-forward networks**: MLP after attention
* Residual connections and layer normalization

**4. Conditioning Inputs:**
* **Text embeddings**: From text encoder (CLIP, T5), fed to cross-attention
* **Timestep embedding**: Current diffusion timestep t, added to patch embeddings
* Optional: Class labels, style vectors, other conditions

**5. Unpatchify (Detokenization):**
* Reshape sequence of N embeddings back to image
* Linear projection: C → 256 per patch
* Reshape: 9 patches (16x16 each) → 48x48 image
* Output: Predicted noise (same shape as input noisy image)

**Extension to 3D (Video):**
* Input: 48x48x6 (6 frames)
* Patchify: 3x3x2 grid = 18 cubes (patches)
* Each patch: 16x16x2 frames
* Sequence length: 18 instead of 9
* 3D positional encoding: (row, col, frame)
* Rest identical to 2D

**Characteristics:**
* **Scalability**: Scales better with data and compute (transformers benefit from scale)
* **Global dependencies**: Self-attention captures long-range relationships
* **Quality**: Superior to UNET on large-scale data
* **Flexibility**: Easily extends to 3D (video), multiple conditions
* **Current adoption**: **All recent SOTA models use DiT**
  * GPT Image 1, Imagen 4, Flux

**UNET vs DiT Comparison:**

| Aspect | UNET | DiT |
|--------|------|-----|
| **Basis** | Convolution (CNN) | Self-attention (Transformer) |
| **Locality** | Strong (convolutions) | Moderate (global attention) |
| **Global dependencies** | Weak (limited receptive field) | Strong (all-to-all attention) |
| **Scalability** | Limited | Excellent |
| **Quality (large data)** | Good | Superior |
| **Current SOTA models** | None (deprecated) | All (GPT Image 1, Imagen 4, Flux) |

> **Key Insight**: DiT has replaced UNET as the standard architecture for diffusion models. Transformers scale better with data and compute, capture global structure more effectively, and achieve superior quality. Every recent state-of-the-art text-to-image and text-to-video model uses DiT, not UNET.

### Training Process

**Diffusion Training Loop:**

![Diagram 6](https://kroki.io/mermaid/svg/eNpNj00KwjAQhfeeYi4gxAOo2B-1ikWkGwldxGasAZOUJIpgvbs104VZDe99782kdaK7QZVNYHgrXuErwNFZ3YUaptMFJCTlprESXR2xJDrp6OgLSqlM68nM-EkYaTWUVnmEMJ8xxqgr55m6Xh9eWQOHoe5OiZTMOK95pTT6gB2MB5Cex3nDjw6lagKVU3wTrS0_obZPhJFA-c9sI1O8AyyALT9RK35af0bfw45n2DgUv3spsPvbTWBph7-wHvY8vaMwUGjRYv0F9TRZWQ==)

**Step 1: Prepare Training Sample**
* Sample clean image from dataset + caption
* Randomly sample timestep: t ~ Uniform(1, T) (e.g., t ~ Uniform(1, 1000))
* Sample random noise: epsilon ~ N(0, I)
* Create noisy image: noisy_image = add_noise(clean_image, epsilon, t)
  * Noise schedule determines how much noise at each t

**Step 2: Prepare Conditions**
* **Text condition**:
  * Encode caption with text encoder (CLIP, T5): text_emb = text_encoder(caption)
  * Text embeddings fed to cross-attention layers
* **Timestep condition**:
  * Embed timestep: t_emb = embed_timestep(t)
  * Learned embedding or sinusoidal encoding
  * Added to patch embeddings or passed separately

**Step 3: Forward Pass (Noise Prediction)**
* Input: noisy_image, text_emb, t_emb
* Model (DiT or UNET) predicts noise: predicted_noise = diffusion_model(noisy_image, text_emb, t_emb)
* Output: Same shape as noisy_image

**Step 4: Calculate Loss and Update**
* Loss: MSE between true noise and predicted noise
  * loss = MSE(predicted_noise, epsilon)
  * Simple, stable loss function
* Backpropagation: Calculate gradients
* Optimizer: Update model parameters (AdamW common)

**Repeat**: Iterate over dataset for multiple epochs

**Training Considerations:**
* **Batch size**: Large batches (256-1024) benefit training stability
* **Learning rate**: Carefully tuned, often with warmup and decay schedules
* **Mixed precision**: FP16 or BF16 for memory efficiency
* **Distributed training**: Multi-GPU, multi-node for large models
* **Timestep sampling**: Can use importance sampling (focus on harder timesteps)
* **Data augmentation**: Random flips, crops, color jitter

### Diffusion Sampling

**Goal**: Generate image from random noise given text prompt

**Traditional Sampling (1000 steps):**
1. Start with random noise: x_T ~ N(0, I) (t = T = 1000)
2. Encode text prompt: text_emb = text_encoder(prompt)
3. For t = T down to 1:
   * Encode timestep: t_emb = embed_timestep(t)
   * Predict noise: epsilon_pred = diffusion_model(x_t, text_emb, t_emb)
   * Remove predicted noise: x_{t-1} = denoise(x_t, epsilon_pred, t)
4. Final output: x_0 (clean generated image)

**Problem**: 1000 steps is **very slow** (1000 forward passes)

**DDIM (2020 Paper): Deterministic 50 steps**
* Reformulates sampling as deterministic process
* Skip timesteps: Sample at t = 1000, 980, 960, ..., 20, 0 (50 steps)
* Maintains quality with far fewer steps
* Trade-off: Slight quality loss vs 20x speedup

**Modern Optimized: 20-50 steps**
* Empirical finding: 20-50 steps often optimal
  * Below 20: Noticeable quality degradation
  * Above 50: Diminishing returns (minor quality gain, much slower)
* Production standard: 30-50 steps

**Extreme Optimization: 1-2 steps**
* Active research area
* Techniques:
  * **Distillation**: Train smaller model to mimic multi-step process in 1-2 steps
  * **Consistency models**: Enforce consistency between different timesteps
* Trade-off: Significant quality loss, but ultra-fast
* Use case: Real-time applications, draft generation

**Sampling Considerations:**
* **Guidance scale** (Classifier-free guidance):
  * Controls text-image alignment strength
  * Higher scale: Closer to text, less diversity
  * Lower scale: More creative, less aligned
  * Typical range: 5-15
* **Negative prompts**: Specify what NOT to generate
* **Seed**: Control randomness for reproducibility

**Speed-Quality Trade-off:**

| Steps | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **1-2** | Ultra-fast | Low | Real-time, draft |
| **20-30** | Fast | Good | Quick generation |
| **30-50** | Moderate | High | Production standard |
| **50-100** | Slow | Marginal improvement | Research, quality-critical |
| **1000** | Very slow | Overkill | Unnecessary |

## Evaluation Methods

Evaluating generative models requires metrics beyond human judgment for scalability and consistency.

### Inception Score

**Purpose**: Measure image quality and diversity

**Method:**
1. Generate 1000+ images with model
2. Pass through pre-trained **Inception V3** classifier
3. For each image i, get class probability distribution P(y|x_i)
4. Calculate marginal distribution: P(y) = average of P(y|x_i) over all images

**Quality Indicator:**
* Each image should have **sharp** probability distribution
* Model confident about object class
* Example: [0.01, 0.02, 0.9, 0.05, 0.02] (confident prediction)
* Counter-example: [0.2, 0.2, 0.2, 0.2, 0.2] (unclear/blurry image)

**Diversity Indicator:**
* Marginal distribution P(y) should be **flat**
* Model generates diverse classes
* Example: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] (covers all classes)
* Counter-example: [0.8, 0.05, 0.05, 0.05, 0.05] (mode collapse, only one class)

**Calculation:**
* Inception Score = exp(E[KL(P(y|x) || P(y))])
* Higher is better
* Typical range: 2-10 for natural images

**Limitations:**
* Doesn't compare to real images (no reference)
* Biased toward ImageNet classes (Inception V3 training data)
* Can be gamed (generate clear images of limited objects)

### FID (Fréchet Inception Distance)

**Purpose**: Measure image quality and diversity **relative to real images**

**Method:**
1. Generate 1000+ fake images with model
2. Collect 1000+ real images from dataset
3. Pass both through Inception V3 → Extract features (e.g., 2048-dim vectors)
4. Calculate statistics:
   * Fake: mean_fake, covariance_fake
   * Real: mean_real, covariance_real
5. Calculate **Fréchet distance** between Gaussian distributions:
   * FID = ||mean_fake - mean_real||^2 + Trace(covariance_fake + covariance_real - 2*sqrt(covariance_fake * covariance_real))

**Interpretation:**
* **Lower FID = more realistic** images
* Measures how close fake distribution is to real distribution
* Captures both quality and diversity

**Advantages over Inception Score:**
* Compares to real data (ground truth reference)
* More sensitive to quality issues
* Widely adopted in research and production

**Limitations:**
* Requires large sample size (1000+) for stable estimates
* Biased toward ImageNet features
* Doesn't capture semantic alignment with text

**Typical Values:**
* FID < 10: Excellent quality (SOTA models)
* FID 10-20: Good quality
* FID 20-50: Moderate quality
* FID > 50: Poor quality

### CLIP Similarity Score

**Purpose**: Measure **text-image alignment** (how well image matches text prompt)

**Method:**
1. For each (text_prompt, generated_image) pair:
   * Encode text: text_emb = CLIP_text_encoder(text_prompt)
   * Encode image: image_emb = CLIP_image_encoder(generated_image)
   * Normalize embeddings to unit vectors
2. Calculate cosine similarity: score = dot(text_emb, image_emb)
3. Average over all pairs

**Interpretation:**
* **Higher CLIP score = better text-image alignment**
* Range: -1 to 1 (typically 0.2 to 0.4 for good models)
* Measures semantic similarity, not pixel-level accuracy

**Use Cases:**
* Primary metric for text-to-image models
* Filter bad generations (low CLIP score)
* Rank models on leaderboards (Artificial Analysis)

**Limitations:**
* CLIP's understanding limited by its training data
* May miss fine-grained details or subtle misalignments
* Can be gamed (generate stereotypical images for prompts)

**Evaluation Summary:**

| Metric | Measures | Reference | Higher/Lower Better | Use Case |
|--------|----------|-----------|---------------------|----------|
| **Inception Score** | Quality + Diversity | No | Higher | General quality |
| **FID** | Quality + Diversity | Yes (Real images) | Lower | Main quality metric |
| **CLIP Score** | Text-Image Alignment | Yes (Text prompt) | Higher | Text-to-image models |

**Practical Evaluation:**
* Use **FID** for image quality
* Use **CLIP score** for text-image alignment
* Combine with **human evaluation** for production (preference studies, A/B tests)
* Leaderboards (e.g., Artificial Analysis) aggregate metrics + human ratings

## Text-to-Video Models

### Current Landscape

**Google: Vio Series**
* **Vio 1**: Early text-to-video model
* **Vio 2**: Improved quality and length
* **Vio 3**: Adds audio generation, longer videos (up to 60 seconds)

**OpenAI: Sora**
* **Release**: February 2024
* **Paper**: "Video Generation Models as World Simulators"
* **Breakthrough**: High-quality 5-20 second videos, up to 1K resolution
* Generates consistent motion, coherent physics (mostly)
* **Issues**: Still not production-ready for all prompts
  * Example: Taco disappears suddenly, laptop floats unnaturally
  * Physics inconsistencies, object permanence challenges
* Multiple aspect ratios: Portrait, square, landscape, custom

**Open-Source: Open Sora**
* Community-built reproduction of Sora architecture
* Open weights and training code
* Lower quality than commercial models but improving

**Other Models:**
* **Seedance** (ByteDance): Currently ranked #1 on leaderboards
* **Kling**: Chinese model, high quality
* **VAN** (Alibaba): First open video model in top 10

**Current Rankings (Artificial Analysis Leaderboard - Text-to-Video):**
* Seedance (ByteDance) #1
* Closed models dominate top spots
* VAN (Alibaba): First open model in top 10
* Rapid progress: New models every few months

**Key Observation:**
* Text-to-video **much harder** than text-to-image
* Temporal consistency, physics, motion, object permanence
* Computational cost **orders of magnitude higher**
* Still active research area, not fully production-ready

### Latent Diffusion Models

**Motivation: Computational Efficiency**

**Problem with Pixel Space:**
* Example video: 1280x720 resolution, 120 frames (5 seconds at 24 FPS)
* Total pixels: 1280 * 720 * 120 = **110,592,000 pixels** (110 million)
* Training diffusion model: Forward pass for each denoising step
* Memory requirements: Enormous (can't fit in GPU memory)
* Training time: Prohibitively slow

> **Key Insight**: Training diffusion models in pixel space for video is computationally infeasible. Latent diffusion models compress videos by 500x using VAE encoders, making video generation practical. Every modern text-to-video model (Sora, Vio, Seedance) uses latent space training.

**Solution: Latent Space Training**

**Compression with VAE:**
* Train VAE encoder-decoder on images/videos first (separate training)
* **Encoder**: Pixel space → Latent space (8x spatial + 8x temporal compression)
  * 1280x720x120 → 160x90x15 = **216,000 pixels** (216K)
  * **Compression factor: 512x** (110M → 216K)
* **Decoder**: Latent space → Pixel space (reconstruction)

**Example Compression:**
* Original: 1280 (width) x 720 (height) x 120 (frames) = 110M
* After encoder: 160 x 90 x 15 = 216K
* **512x smaller** - fits in memory, trains faster

**Latent Diffusion Training:**
1. Pre-compute latents: Pass all training videos through VAE encoder, save latents
2. Train diffusion model on latents (not pixels)
3. All denoising happens in compressed latent space
4. Decoder used only during inference (generation)

**Quality Trade-off:**
* Slight quality loss from VAE compression (lossy)
* **Negligible if VAE well-trained** (common practice)
* Massive computational savings outweigh minor quality loss

**Latent Diffusion for Images:**
* **Stable Diffusion**: Famous example of latent diffusion
* 8x spatial compression (e.g., 512x512 → 64x64 latents)
* Enabled open-source high-resolution image generation
* Same principle extends to video

**Key Components:**
* **VAE Encoder**: Compression network (trained separately)
* **Diffusion Model**: Operates in latent space (main training)
* **VAE Decoder**: Decompression to pixel space (inference only)
* VAE is same architecture from generative modeling techniques (repurposed for compression)

### Building Text-to-Video Models

![Diagram 7](https://kroki.io/mermaid/svg/eNptkctuwjAQRff9ivmBSOljWyog4RlCChFSNWLhkgEsJXZkmxap9N_reCK1i2bl3HN8PXFORrRnKJM78M8QS7o6KIxuWreHKBrAiKNUHXRFZh-0USDjnjTvVFVSnSzDBDdCVbqBxwRyLS2Be76P45jrUvRxIo_Hi5VawcqX1v695L1jdsJ6gqVsyDpqoR-F8zSsp1gYquTB9Wc8JtwwDXSGG2r0BzFkMgtk_uVgAPHLd8jmXXZ7I3uDBSZ0MCS6eXnD4s-hLOb6Bksc1yQUZMKRcv03L4Oa4W6Ygq_5vaksgBVm-jMyZGEnK9KMVgHluG2Fk6KG7aUlE23I6vriurt5urKYB3GNM3k6kwFvcL4OeeH_QtNq81_DQ99QBPMVJ1J5revprH6YH31lkJk=)

Text-to-video extends text-to-image by adding the **temporal dimension**.

#### 1. Data Preparation

**Video-Text Pairs:**
* Source: Internet (YouTube, stock videos, user uploads)
* Scale: No Lyon 5B equivalent for video (smaller datasets)
* Typical: Millions of video-text pairs (much less than image)

**Video Filtering:**
1. **Inappropriate content**: Violence, nudity, harmful content, licensing issues, logos
2. **Quality**: Remove low-resolution, blurry, shaky videos
3. **Length**: Remove very short clips (< 2 seconds) or very long (> 60 seconds)
4. **Duplicates**: Use video embeddings + clustering to deduplicate

**Video Standardization:**
1. **Sample fixed duration**: Extract 5-10 second clips from longer videos
2. **Standardize frame rate**: Re-encode to fixed FPS (e.g., 24 FPS)
3. **Adjust dimensions**: Resize and crop to target resolution (e.g., 720p, 1080p)
4. **Pre-compute latents**: Pass through VAE encoder, save compressed latents
   * Saves massive computation during training (don't recompute every time)

**Caption Preparation** (Same as images):
1. Missing captions → Generate with VLMs
2. Enhance captions → Detailed descriptions
3. Filter misaligned pairs → CLIP video-text similarity

**Combining Images and Videos:**
* **Problem**: Video datasets much smaller than image datasets
* **Solution**: Train on **both images and videos**
  * Images treated as 1-frame videos
  * Provides spatial understanding (objects, colors, concepts)
  * Videos provide temporal understanding (motion, actions)

**Training Strategies:**
* **Strategy 1: Pre-train on images, fine-tune on videos**
  * Step 1: Train text-to-image model on Lyon 5B
  * Step 2: Continue training on video data (adapt to temporal dimension)
  * Advantage: Faster convergence, learns spatial first then temporal
* **Strategy 2: Joint training**
  * Train single model on combined image + video data from start
  * Advantage: Simpler pipeline, single training run

#### 2. Model Architecture

**Extend DiT to 3D:**

**Key Difference: 3D Patchify**
* Input: 48x48x6 (6 frames)
* Divide into 3D grid: 3x3x2 = **18 cubes** (patches)
* Each cube: 16x16x2 frames
* Flatten and project: 512 values → C embedding dim
* Sequence length: 18 (instead of 9 for images)

**3D Positional Encoding:**
* Position: (row, col, frame)
* Example: (1,1,1), (1,1,2), (1,2,1), (1,2,2), ...
* Maps three numbers → vector
* Preserves spatial AND temporal relationships

**Transformer Blocks:**
* Identical to 2D DiT
* Self-attention: Patches attend to each other (spatial and temporal)
* Cross-attention: Attend to text embeddings
* Feed-forward networks

**3D Unpatchify:**
* Reshape sequence of 18 embeddings → 48x48x6 video
* Inverse of patchify operation

**Text Conditioning:**
* Same as text-to-image
* Text encoder (CLIP, T5) → embeddings
* Cross-attention layers throughout model

**Timestep Conditioning:**
* Same diffusion timestep embedding t
* Indicates noise level

#### 3. Training

**Similar to Text-to-Image:**
1. Sample video + caption from dataset (or use pre-computed latents)
2. Randomly sample diffusion timestep t
3. Add noise to video (3D noise)
4. Model predicts 3D noise given: noisy video, text, t
5. Loss: MSE between true noise and predicted noise
6. Update weights

**Efficiency Techniques:**
* **Latent space training**: Train on pre-computed VAE latents (512x compression)
* **Mixed precision**: FP16/BF16 to save memory
* **Gradient checkpointing**: Trade compute for memory
* **Distributed training**: Multi-GPU, multi-node (essential for video)
* **Data parallelism**: Split batch across GPUs

**Training Challenges:**
* **Memory**: Video models **much larger** than image models
* **Compute**: Training takes weeks to months on large clusters
* **Data scarcity**: Limited high-quality video-text pairs
* **Cost**: Extremely expensive (tens to hundreds of millions of dollars)

#### 4. Sampling/Generation

**Base Model Generation:**
1. Start with random 3D latent noise (t=1000)
2. Encode text prompt: text_emb = text_encoder(prompt)
3. Iteratively denoise (20-50 steps):
   * Predict noise: epsilon_pred = diffusion_model(latent_t, text_emb, t)
   * Remove noise: latent_{t-1} = denoise(latent_t, epsilon_pred, t)
4. Output: Clean latents (low resolution, low frame count)

**VAE Decoder:**
* Input: Clean latents
* Output: Pixel space video (low resolution)
* Example: 160x90x15 latents → 1280x720x120 video

**But often generate at lower res first:**
* Base model outputs: 60 frames, 320x180 resolution
* Not final quality - need super-resolution

**Spatial Super-Resolution:**
* Separate model trained for upsampling
* Input: Low-res video (e.g., 320x180)
* Output: 4x higher resolution (e.g., 1280x720)
* Keeps frame count unchanged (60 frames → 60 frames)
* Uses same diffusion or specialized upsampling architecture

**Temporal Super-Resolution:**
* Separate model trained for frame interpolation
* Input: Low frame rate (e.g., 12 FPS, 60 frames = 5 seconds)
* Output: 2x frame rate (e.g., 24 FPS, 120 frames = 5 seconds)
* Keeps spatial resolution unchanged
* Generates intermediate frames using motion prediction

**Why Cascaded Approach?**
* Training base model at full resolution is **prohibitively expensive**
* Separate super-resolution models are **more efficient**
* Allows independent optimization of each stage
* Final quality: High-res (1K+), high frame rate (24-30 FPS), 5-20 seconds

**Sora Example:**
* Base: Latent space generation (low res, low FPS)
* Decoder: VAE decoder to pixel space
* Super-resolution: Multiple stages to reach 1K
* Final: 5-20 second videos, multiple aspect ratios

### Production System Design

![Diagram 8](https://kroki.io/mermaid/svg/eNpNkFFugzAQRP97ir0AV0iVECAkJI1AjVSt-LCcbaGltmUWqVXp3QtrpOI_z7wd7_jNK9dAUT7AdLb43JOHq7efjmuIog3ssFKvxN-QGzcwxA3pj1rgnfjxz-w__ooUz9J4sSPssaR30lyv9BfqR0gwpENiGmU0-UAkEpbirb2ThYwMecWtNXC2d-oCkwqTYaGYDEPllCZ4GnjaKwCZAAe8bRPYk54ml_SDGDle2y_qlkF5Kdi52EcMWbAUXlU9CnBaVz39V13dpWKB1eDIRyX1thvmFiGlkJQzpq1R3bI4sIX5z-s_WYFyJw==)

**Full Pipeline: User Prompt → Final Video**

**1. Prompt Autocomplete (Optional):**
* Suggest completions as user types
* Helps users craft better prompts
* Improves engagement

**2. Safety Service (Input Guardrails):**
* Check if prompt requests inappropriate content
  * Violence, nudity, illegal activity, hate speech
* Classifier-based or LLM-based moderation
* **Reject** if unsafe - return error to user

**3. Prompt Enhancer:**
* Fix spelling and grammar errors
* Add detail and clarity
* Paraphrase for better model understanding
* Example: "cat" → "A fluffy orange cat sitting by a window, looking outside, soft lighting"
* Often uses LLM (GPT-4, Claude) to enhance

**4. Video Generation Component:**
* Text encoder → Text embeddings
* Sample random latent noise
* Iterative denoising (20-50 steps in latent space)
* Output: Clean latents

**5. Visual Decoder (VAE):**
* Decompress latents to pixel space
* Output: Low-res video

**6. Output Guardrails (Harm Detector):**
* Check if generated video contains inappropriate content
* Classifier-based detection (violence, nudity, etc.)
* **Reject** if unsafe - don't show to user

**7. Spatial Super-Resolution:**
* Upsample resolution (e.g., 4x)
* 320x180 → 1280x720

**8. Temporal Super-Resolution:**
* Increase frame rate (e.g., 2x)
* 12 FPS → 24 FPS

**9. Final Output:**
* High-resolution, high frame rate video
* Delivered to user

**Safety Considerations:**
* **Input guardrails**: Prevent malicious prompts
* **Output guardrails**: Catch generated harmful content
* **Both necessary**: Model may generate unexpected content even from innocent prompts
* **Human review**: For edge cases, human moderators review flagged content

**Practical Features (Sora Example):**
* **Aspect ratios**: Portrait, square, landscape, custom
* **Resolution**: Up to 1K (1920x1080)
* **Duration**: 5-20 seconds (user selectable)
* **Number of outputs**: Generate 1, 2, or 4 videos (user picks best)
* **Model selection**: T2V (text-to-video) or I2V (image-to-video)
  * I2V: User uploads image, generates video from that starting frame

**Performance Optimization:**
* **Caching**: Cache text embeddings for repeated prompts
* **Batching**: Generate multiple videos in parallel
* **Model serving**: GPU inference servers (NVIDIA Triton, TorchServe)
* **Queue management**: Handle burst traffic, prioritize requests
* **Cost control**: Expensive inference, need quotas and rate limits

## Key Insights

### Fundamental Principles

* **Diffusion models are the current standard**: 95%+ of modern text-to-image and text-to-video models use diffusion, not GAN or autoregressive. Quality wins despite slower speed.

* **VAEs repurposed for compression**: VAEs no longer competitive for generation quality, but essential as encoder-decoder pairs in latent diffusion models. Every video model uses VAE compression.

* **Transformers dominate architecture**: DiT (Diffusion Transformer) has replaced UNET as the standard diffusion architecture. All recent SOTA models (GPT Image 1, Imagen 4, Flux) use DiT for better scalability and global dependency modeling.

* **Latent space training is mandatory for video**: Training diffusion in pixel space for video is computationally infeasible (110M pixels per sample). VAE compression (512x) makes video generation practical.

* **Cascaded generation is standard**: Base model generates low-res output, separate super-resolution models upscale spatially and temporally. More efficient than training at full resolution.

* **Data quality determines model quality**: High-quality captions, filtered inappropriate content, deduplicated data, and aesthetic filtering are critical. Poor data = poor results.

### Important Distinctions and Comparisons

* **VAE vs GAN vs Autoregressive vs Diffusion quality hierarchy**: Diffusion > Autoregressive > GAN > VAE. Diffusion wins on quality, autoregressive too slow, GAN has mode collapse, VAE too blurry.

* **UNET vs DiT architecture**: DiT scales better with data/compute due to transformer architecture. UNET limited by convolution's local receptive field. All modern models use DiT.

* **Pixel space vs latent space training**: Latent space 500x more efficient for video. Slight quality loss from VAE compression is negligible with good encoder-decoder.

* **Pre-train on images then fine-tune on videos vs joint training**: Pre-training learns spatial concepts first (objects, colors), fine-tuning adds temporal (motion). Joint training simpler but may be less efficient. Both strategies used in practice.

* **FID vs Inception Score vs CLIP Score**: FID measures quality relative to real images (lower better). Inception Score measures quality/diversity without reference (higher better). CLIP Score measures text-image alignment (higher better). FID + CLIP Score are primary metrics for text-to-image evaluation.

* **Greedy/deterministic sampling vs iterative denoising**: Diffusion requires 20-50 iterative denoising steps (slow). Active research on 1-2 step models (distillation, consistency models) trades quality for speed.

* **Text-to-image vs text-to-video complexity**: Video adds temporal dimension (motion, physics, object permanence). Computational cost orders of magnitude higher. Video still active research area, not fully production-ready.

### Practical Considerations

* **Optimal sampling steps: 20-50 for production**: Below 20 steps degrades quality noticeably. Above 50 steps gives diminishing returns (minor quality gain, much slower). Production systems typically use 30-50 steps.

* **Text rendering dramatically improved**: Historically a major challenge for diffusion models. GPT Image 1 and Imagen 4 now excel at text rendering in generated images. Diagram generation (logical structure + text) still difficult.

* **Open model gap**: Best models are closed-source (GPT Image 1, Imagen 4). First open model (Flux) ranks #3. Open-source models lag 6-12 months behind commercial leaders.

* **Safety guardrails are mandatory**: Both input (prompt) and output (generated content) guardrails required. Model can generate inappropriate content even from innocent prompts.

* **Prompt enhancement improves results**: Fixing spelling, adding detail, and paraphrasing user prompts significantly improves generation quality. Often use LLM (GPT-4) as prompt enhancer.

* **Pre-computing latents saves compute**: For latent diffusion, pre-compute and cache VAE latents of training data. Avoids redundant encoding during training (massive speedup).

* **Video generation extremely expensive**: Training costs tens to hundreds of millions of dollars. Inference also expensive (multiple forward passes, super-resolution stages). Requires careful cost management and quotas.

* **Combining image and video data helps**: Video datasets much smaller than image datasets. Training on both (treating images as 1-frame videos) improves video model quality by teaching spatial concepts.

* **Multiple aspect ratios now standard**: Modern models (DALI 3, Imagen 3, Sora) support portrait, square, landscape, and custom aspect ratios. Requires training on varied aspect ratios or cropping strategy.

### Common Pitfalls and Misconceptions

* **Thinking VAEs are dead**: VAEs not used for generation anymore, but critical as compression in LDM. Misunderstanding leads to overlooking their importance.

* **Training diffusion at full video resolution**: Prohibitively expensive. Always use latent space training with VAE compression (500x reduction).

* **Ignoring image data for video models**: Using only video data wastes opportunity to learn spatial concepts. Combining image and video data improves quality and convergence.

* **Not pre-computing latents**: Re-encoding images/videos to latents every training step is redundant. Pre-compute once, cache, and reuse (massive speedup).

* **Poor caption quality**: "Garbage in, garbage out." Low-quality captions lead to poor text-image alignment. Invest in caption generation/enhancement with VLMs.

* **Skipping inappropriate content filtering**: Leads to safety issues and potential legal problems. Filter during data preparation AND generation output.

* **Using wrong number of sampling steps**: Too few (< 20) degrades quality. Too many (> 50) wastes compute. Find optimal range (20-50) for your use case.

* **Assuming video generation is production-ready**: Still active research. Physics inconsistencies, object permanence issues, motion artifacts common. Not suitable for all use cases yet.

* **Overlooking evaluation metrics**: Relying only on human evaluation doesn't scale. Use FID (quality), CLIP Score (alignment), and human studies together.

* **Training without distributed setup**: Video models too large for single GPU. Requires multi-GPU, multi-node training. Plan infrastructure early.

## Quick Recall / Implementation Checklist

* [ ] Understand four generative techniques: VAE (compression), GAN (deprecated), Autoregressive (slow), Diffusion (standard)
* [ ] Remember VAEs still critical as compression networks in latent diffusion models
* [ ] Use DiT (Diffusion Transformer) architecture, not UNET, for new models (better scalability)
* [ ] Prepare high-quality data: Fix captions with VLMs, filter inappropriate content, deduplicate, aesthetic filtering
* [ ] Train in latent space for video (512x compression via VAE encoder), never pixel space
* [ ] Pre-compute VAE latents for training data (cache and reuse - massive speedup)
* [ ] Combine image and video data for video models (spatial + temporal learning)
* [ ] Use 20-50 sampling steps in production (optimal speed-quality trade-off)
* [ ] Implement cascaded generation: Base model → VAE decoder → Spatial super-res → Temporal super-res
* [ ] Add safety guardrails: Input (prompt checking) and output (harmful content detection)
* [ ] Enhance user prompts with LLM (fix spelling, add detail, paraphrase)
* [ ] Evaluate with FID (image quality), CLIP Score (text-image alignment), and human studies
* [ ] Remember text-to-video still active research (physics issues, object permanence challenges)
* [ ] Plan for extreme compute requirements: Multi-GPU, multi-node training, expensive inference
* [ ] Support multiple aspect ratios in modern systems (portrait, square, landscape)
* [ ] Consider cost management: Quotas, rate limits, caching for production deployment
* [ ] Use classifier-free guidance to control text-image alignment strength (typical range: 5-15)
* [ ] Implement batching and queue management for handling burst traffic
* [ ] Monitor for mode collapse (lack of diversity) and quality degradation over time
* [ ] Stay updated: Field evolving rapidly, new models and techniques every few months

---

## References

### Papers

* [**"Auto-encoding Variational Bayes"**](https://arxiv.org/abs/1312.6114) - Kingma & Welling (2013)
* [**"Generative Adversarial Networks"**](https://arxiv.org/abs/1406.2661) - Ian Goodfellow et al. (2014)
* [**"Conditional Image Generation with Pixel CNN Decoders"**](https://arxiv.org/abs/1606.05328) - Google DeepMind (2016)
* [**"Image Transformer"**](https://arxiv.org/abs/1802.05751) - Google Brain (2018)
* [**"Denoising Diffusion Probabilistic Models"**](https://arxiv.org/abs/2006.11239) - UC Berkeley (2020)
* [**"Zero Shot Text-to-Image Generation" (DALI 1)**](https://arxiv.org/abs/2102.12092) - OpenAI (2021)
* [**"Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALI 2)**](https://arxiv.org/abs/2204.06125) - OpenAI (2022)
* [**"Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen)**](https://arxiv.org/abs/2205.11487) - Google (2022)
* [**"High-Resolution Image Synthesis with Latent Diffusion Models"**](https://arxiv.org/abs/2112.10752) - Stability AI (2022)
* [**"Scalable Diffusion Models with Transformers"**](https://arxiv.org/abs/2212.09748) - DiT Paper (2023)
* [**"Video Diffusion Models"**](https://arxiv.org/abs/2204.03458) - Google (2022)
* [**"Video Generation Models as World Simulators" (Sora)**](https://openai.com/research/video-generation-models-as-world-simulators) - OpenAI (2024)
* [**"Convolutional Networks for Biomedical Image Segmentation" (UNET)**](https://arxiv.org/abs/1505.04597) - Ronneberger et al. (2015)
* [**"Denoising Diffusion Implicit Models" (DDIM)**](https://arxiv.org/abs/2010.02502) - Song et al. (2020)
* [**"A Style-Based Generator Architecture for Generative Adversarial Networks" (StyleGAN)**](https://arxiv.org/abs/1812.04948) - NVIDIA (2018)

### Datasets

* [**LAION-400M**](https://laion.ai/blog/laion-400-open-dataset/) - 400 million image-text pairs (2021)
* [**LAION-5B**](https://laion.ai/blog/laion-5b/) - 5 billion image-text pairs (2022)
* [**MNIST**](http://yann.lecun.com/exdb/mnist/) - Handwritten digits dataset

### Tools & Platforms

* [**ChatGPT**](https://chat.openai.com/) - OpenAI's chatbot with DALI/GPT Image integration
* [**Gemini**](https://gemini.google.com/) - Google's chatbot with Imagen integration
* [**Flux (Black Forest Labs)**](https://blackforestlabs.ai/) - Open text-to-image model (200 free credits)
* [**Stable Diffusion**](https://stability.ai/stable-diffusion) - Open-source latent diffusion model
* [**Midjourney**](https://www.midjourney.com/) - Commercial text-to-image service
* [**This Person Does Not Exist**](https://thispersondoesnotexist.com/) - GAN-powered face generation demo
* [**Artificial Analysis**](https://artificialanalysis.ai/) - Leaderboards for text-to-image and text-to-video models
* [**file.ai**](https://file.ai/) - Model comparison platform
* [**OpenAI Sora**](https://openai.com/sora) - Text-to-video model

### Model Providers

* [**OpenAI**](https://openai.com/) - DALI, GPT Image, Sora
* [**Google**](https://deepmind.google/) - Imagen, Vio
* [**Stability AI**](https://stability.ai/) - Stable Diffusion
* [**Black Forest Labs**](https://blackforestlabs.ai/) - Flux
* [**ByteDance**](https://www.bytedance.com/) - Seedream, Seedance
* [**Alibaba**](https://www.alibabacloud.com/) - VAN
* [**NVIDIA**](https://www.nvidia.com/en-us/research/) - StyleGAN

### Evaluation Metrics

* [**Inception Score**](https://arxiv.org/abs/1606.03498) - Salimans et al. (2016)
* [**Fréchet Inception Distance (FID)**](https://arxiv.org/abs/1706.08500) - Heusel et al. (2017)
* [**CLIP**](https://arxiv.org/abs/2103.00020) - OpenAI (2021) - Used for CLIP Similarity Score

### Community & Open Source

* [**Open Sora**](https://github.com/hpcaitech/Open-Sora) - Community reproduction of Sora
* [**Hugging Face Diffusers**](https://huggingface.co/docs/diffusers/index) - Diffusion model library
* [**Stability AI GitHub**](https://github.com/Stability-AI/stablediffusion) - Stable Diffusion implementation
