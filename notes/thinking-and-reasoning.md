# Thinking and Reasoning LLMs

## Table of Contents

* [Overview](#overview)
* [What Are Reasoning Models?](#what-are-reasoning-models)
  * [Core Concept](#core-concept)
  * [Popular Reasoning Models](#popular-reasoning-models)
  * [Reasoning Model Behavior](#reasoning-model-behavior)
* [Inference-Time Scaling](#inference-time-scaling)
  * [Trade-off](#trade-off)
* [Chain-of-Thought (CoT)](#chain-of-thought-cot)
  * [Visualization](#visualization)
  * [Few-Shot CoT](#few-shot-cot)
  * [Zero-Shot CoT](#zero-shot-cot)
* [Self-Consistency](#self-consistency)
  * [Method](#method)
  * [Combining with CoT](#combining-with-cot)
  * [Selecting the Best Response](#selecting-the-best-response)
  * [Critical Distinction: Verifiable vs Unverifiable Tasks](#critical-distinction-verifiable-vs-unverifiable-tasks)
* [Tree-of-Thoughts (ToT)](#tree-of-thoughts-tot)
  * [Method](#method-1)
  * [Advantage over Self-Consistency](#advantage-over-self-consistency)
  * [Search Algorithms](#search-algorithms)
* [Sequential Revision](#sequential-revision)
  * [Method](#method-2)
  * [When to Use](#when-to-use)
  * [Combining Approaches](#combining-approaches)
* [Training with Reasoning](#training-with-reasoning)
  * [Comparison](#comparison)
* [CoT Training](#cot-training)
  * [STaR (Self-Taught Reasoner)](#star-self-taught-reasoner)
  * [Iterative Process](#iterative-process)
  * [Special Tokens](#special-tokens)
  * [Training Procedure](#training-procedure)
* [RL with Reward Models](#rl-with-reward-models)
  * [Method](#method-3)
* [Outcome Reward Model (ORM)](#outcome-reward-model-orm)
  * [Training Data Format](#training-data-format)
  * [Data Collection](#data-collection)
  * [Training](#training)
  * [Limitation](#limitation)
* [Process Reward Model (PRM)](#process-reward-model-prm)
  * [Training Data Format](#training-data-format-1)
  * [Data Collection Challenge](#data-collection-challenge)
  * [Comparison](#comparison-1)
  * [Dual Use of PRM](#dual-use-of-prm)
* [Self-Correction](#self-correction)
  * [Two Dimensions](#two-dimensions)
  * [Intrinsic Self-Correction (Inference)](#intrinsic-self-correction-inference)
  * [Extrinsic Self-Correction (Inference)](#extrinsic-self-correction-inference)
  * [Training for Self-Correction](#training-for-self-correction)
  * [Self-Correction in Production](#self-correction-in-production)
* [Internalizing Search](#internalizing-search)
  * [System 1 vs System 2 Reasoning](#system-1-vs-system-2-reasoning)
  * [Meta Chain-of-Thought (Meta-CoT)](#meta-chain-of-thought-meta-cot)
  * [Data Format](#data-format)
  * [Training Process](#training-process)
  * [Emergent Behaviors](#emergent-behaviors)
  * [Evidence in Production Models](#evidence-in-production-models)
* [Deep Research](#deep-research)
  * [Architecture](#architecture)
  * [Reasoning LLM Role](#reasoning-llm-role)
  * [Recent Update (July 2025)](#recent-update-july-2025)
  * [User Experience](#user-experience)
* [How Production Systems Combine Techniques](#how-production-systems-combine-techniques)
* [Production Deployment Considerations](#production-deployment-considerations)
  * [Cost Management](#cost-management)
  * [Request Routing](#request-routing)
  * [Performance Optimization](#performance-optimization)
  * [System Architecture](#system-architecture)
* [Key Insights](#key-insights)
  * [Fundamental Trade-offs](#fundamental-trade-offs)
  * [Task Categorization](#task-categorization)
  * [Reward Model Hierarchy](#reward-model-hierarchy)
  * [Search Internalization](#search-internalization)
  * [Self-Correction Reality](#self-correction-reality)
  * [Production Systems](#production-systems)
  * [Sequential vs Parallel - Decision Guide](#sequential-vs-parallel---decision-guide)
* [Quick Recall / Implementation Checklist](#quick-recall--implementation-checklist)
  * [Inference-Time Scaling](#inference-time-scaling-1)
  * [Training Approaches](#training-approaches)
  * [Reward Models](#reward-models)
  * [Search and Exploration](#search-and-exploration)
  * [Self-Correction](#self-correction-1)
  * [Production Deployment](#production-deployment)
  * [Model Selection](#model-selection)
  * [Data Quality](#data-quality)
  * [Debugging](#debugging)
  * [System Architecture](#system-architecture-1)
  * [Resources](#resources)

---

**TL;DR**: Reasoning models (o3, DeepSeek-R1, Gemini 2.5 Pro) add explicit thinking phases before responding. Inference-time scaling: CoT (few-shot/zero-shot), self-consistency (parallel generation + voting), ToT (tree search), sequential revision. Training-time: STaR (self-taught), RL with reward models (ORM judges outcomes, PRM judges steps - PRM > ORM). Self-correction works extrinsically (tools/code execution) but struggles intrinsically without external feedback. Search internalization (Meta-CoT) teaches models when to use System 1 (fast) vs System 2 (slow reasoning). Production: route simple queries to fast models, complex to reasoning models; cache patterns; monitor costs. Critical: verifiable tasks enable automated evaluation; unverifiable require reward models.

---

## Overview

Reasoning-focused LLMs achieve stronger problem-solving through specialized inference-time and training-time techniques. This lecture notes explores how models like OpenAI's o-series and DeepSeek-R1 generate multi-step logical chains before responding. Traditional LLMs generate responses in a single forward pass. **Reasoning models** insert an explicit reasoning phase between input and output, allowing multi-step logical chains and exploration of alternative paths before producing final answers.

**Key insight**: Spending more compute at inference time (generating more tokens for reasoning) or training time (teaching models to reason) improves accuracy on complex tasks.

---

## What Are Reasoning Models?

### Core Concept

Reasoning models are LLMs with an added **thinking phase** before responding:

![Diagram 1](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6ICi_NyCklgFXV07BafooNTE4vy8zLx0hYCMxOLUWLAiJ7Ckc3RIRn5pekaJgiGysAtc2AhZ2BUu7AcRdgYLu0W7ZeYl5ig45hWXpxZBZFwgMmC2K4QNALaUKu8=)

**Key characteristics**:
* Generate intermediate reasoning traces (thoughts)
* Multi-step logical chains
* Multi-path exploration of different approaches
* Defer final answer until reasoning is complete

### Popular Reasoning Models

**Model Landscape (as of 2025)**:

| Model | Provider | Type | Notes |
|-------|----------|------|-------|
| **o3, o3-pro, o4-mini** | OpenAI | Reasoning | Best reasoning performance, slower |
| **GPT-4.5, GPT-4o** | OpenAI | Standard | Fast, not specialized for reasoning |
| **Gemini 2.5 Pro** | Google | Reasoning | Top-ranked on LMSys leaderboard |
| **Grok 4** | xAI | Reasoning | Competitive with major labs |
| **Kimi k2** | Moonshot | Standard | Non-thinking, ranks #5 globally |
| **DeepSeek-R1** | DeepSeek | Reasoning | Open-source (MIT), shows actual reasoning traces |

**Leaderboard observation**: Top 6 models include both reasoning and non-reasoning models, but reasoning models dominate complex benchmarks.

### Reasoning Model Behavior

**Example comparison** (counting 1 to 10):

**GPT-4o (standard)**: Directly outputs answer in ~1 second

```

Counting from 1 to 10: 1, 2, 3...10
That took about 10 seconds in real time.

```

**o4-mini (reasoning)**: Thinks for 9 seconds first

```

<Thinking for 9 seconds...>
User wants me to count and measure time.
Challenge: I'm ChatGPT, can't track real time.
Solution: Write code to capture timing.
<Generates and runs timing code>

Output: Counted 1 to 10 in ~0.00012 seconds

```

**DeepSeek-R1 (open reasoning)**: Shows actual raw reasoning traces
* Considers multiple approaches
* Handles ambiguity in prompt
* Decides on implementation strategy
* Shows complete thought process (not summarized)

---

## Inference-Time Scaling

Inference-time scaling improves reasoning **without retraining** by generating more tokens during inference.

| Technique | Complexity | Cost | Accuracy Gain | Exploration | Best For | Limitation |
|-----------|------------|------|---------------|-------------|----------|------------|
| **Zero-shot CoT** | Lowest | ~2x tokens | +20-40% | None | Quick reasoning boost | Simple problems only |
| **Few-shot CoT** | Low | ~3-5x tokens | +30-50% | None | Format-specific tasks | Requires good examples |
| **Self-Consistency** | Medium | ~10-30x tokens | +40-60% | Parallel paths | Verifiable tasks | Expensive at scale |
| **Tree-of-Thoughts** | High | ~50-100x tokens | +50-70% | Tree search | Complex problems | Very expensive |
| **Sequential Revision** | Medium | ~5-15x tokens | +30-50% | Iterative | Tasks with feedback | Requires external validation |

### Reasoning Technique Decision Tree

![Reasoning Technique Decision Tree](https://kroki.io/mermaid/svg/eNp1UkFu2zAQvPcVe-slglP0WiSo7Tp2k6hArRZoBR8ocS0tTJEqSdkRpP69K0ZqnATRhYA4M5yd2cKKuoRk-Q74-5zGiBIy9B4tWBTOaNLF9Q6i6Arm3byRBfqZEh513kJutPNWkPbXfwN_PuD6hIrSQxawPSy6RLgDY6ta4QP59hl2NfzLFP6HL7uNAz8wjmhpT4IvmREoi0DZ0qDUw5f0N1oTudJ4WJjkU2ZnV0JKuEP_niVK0gdwHmvI2nDuzjTuUVJTnZnqYZWu8PRcrbbmSBIBH8SAc7tHG8sg8QvdxZnHQKiEL2e5kezuplub0_TAOPIjMTYX0OgXzJzD9nTEmalRR6glyh7W6Q-HsMU_DWpPQsF3PJIjowPlRL5ka9yU5qs9F5eJ_DB6vHlrzE26RbWPFlwduVBjEPtwGX28hFpYoRQqcNPAT1o_0baTUg9f08QiRmYfJaVpuG8XZBwKm5cgVGEs26umyNZhgW67sF5VozzV4-B2HAmsabR0Y1K3U8Q93KVvBUA8egiNN3VPGisG7c7osenh_nWrxsLLCEaXm-Ay7raelALNHJHnDb-Bo634qfmc56OcLQ27ynmc3Q8FF8ZIQD1E08O3dIm1Mi2Eyl49_g9keyxX)

### Trade-off

**Cost**: More tokens → higher latency + compute cost
**Benefit**: Better accuracy on complex reasoning tasks

> **Key Insight**: Inference-time scaling improves reasoning without retraining by generating more tokens (CoT, self-consistency, ToT). Trade-off: higher cost and latency for better accuracy on complex tasks. Production systems route simple queries to fast models, complex queries to reasoning models that spend compute on thinking before responding.

![Diagram 2](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaM-8tNSi1LzkVAWn0pT01JJYBV1dOwWnaN_8olSFkPzs1LziWLBSJ7CEc7RTaklJapFCUGpicX5eZl46RNYZLOsS7ZGZngGUdUxOLi1KTK5E1uoKk3TOLy6JBQCw1ygQ)

---

## Chain-of-Thought (CoT)

**Core idea**: Generate intermediate reasoning steps before the final answer.

### Visualization

![Diagram 3](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALHaM-8gtKSWAVdXTsFp-iQjPzS9IwSBcNYsKwTWNgZLmwEEXYGC7vAhf0gwi5gYddo_9ISkJFgseKSypxUBUeFtMycHCvlVMM007Q0JAlXqERaWppJqiEAxCEq4Q==)

### Few-Shot CoT

Show examples with reasoning steps in the prompt:

**Example**:

```

Q: There are 2 boxes with 3 balls each. How many balls total?
A: 2 × 3 = 6, so 6.

Q: 3 red bags with 5 apples each, 2 blue bags with 7 apples each. Total apples?
A:

```

Model learns the pattern:

```

3 × 5 = 15 (red bags)
2 × 7 = 14 (blue bags)
15 + 14 = 29
So, 29.

```

### Zero-Shot CoT

Add a simple trigger phrase: **"Let's think step by step."**

**Paper**: *Large Language Models are Zero-Shot Reasoners* (2022)

**How it works**: Models trained on internet data have seen this phrase paired with step-by-step explanations, so it activates reasoning behavior.

**Example**:

```

Q: A lily pad doubles daily. Covers pond on day 48. Half-covered on which day?
A: Let's think step by step.

Day 47 → half covered
Day 48 → doubles to full coverage
Answer: Day 47

```

**Effectiveness comparison**:
* 0-shot: Often fails
* Few-shot (no CoT): Still fails
* Few-shot CoT: Works
* **Zero-shot CoT**: Works (no examples needed!)

---

## Self-Consistency

**Also called**: Parallel sampling, Best-of-N sampling

> **Understanding the Terminology**: Best-of-N is the foundational concept (generate N responses, pick best). Self-consistency is Best-of-N with specific scoring methods (majority voting or reward model). Parallel sampling is another name for the same approach.

![Diagram 4](https://kroki.io/mermaid/svg/eNpNyjEOgCAMheHdU_QCDupuIiJuyuBGGIghSlQkWO-vgaWd_ryvWzRhh4UX8F-nZLyvgBrKsgWmRuttNOhuD5XOH0l6KjUVTqWhMlCZsrAkQkm3HsDsg3nu85yakx5Ii9SjEs6bE-YXw4v6A0VxNPA=)

### Method

1. Sample N responses independently (with temperature > 0 for diversity)
2. Score or vote to select the best
3. Return highest-scoring response

### Combining with CoT

**Self-Consistency + CoT** = Each branch does full reasoning before voting

![Diagram 5](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaM-8gtKSWAVdXTsFJ8No5_wQhYDEkgwFw1iIPETCCCFhhCJhjJDwg0g4GYJlnA2jHfOKy1OLrBSMLKEyRhAZI4SMBVTGGCJjjKHHGWKaS3VYfklqLUQIYgzE_c7GSBwXMNs12jcxK78os6QSbAwAmPE5bw==)

**Paper**: *Self-Consistency Improves Chain of Thought Reasoning* (Google, 2022)

**Key finding**: Accuracy keeps improving as N increases (more samples = better results)

### Selecting the Best Response

**Two approaches**:

#### 1. Majority Voting

* Parse final answers from each generation
* Count frequency of each answer
* Pick most common

**When it works**: Verifiable tasks (math, logic) with parseable final answers

**Example**:

```

Gen 1: "so, 29"    → 29
Gen 2: "so, 28"    → 28
Gen 3: "so, 29"    → 29
Gen 4: "so, 29"    → 29
Vote: 29 (3 votes) wins

```

#### 2. Reward Model

Train a model to score (prompt, response) pairs:

![Diagram 6](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6ICi_NyCEgVthaDU4oL8vOJUBcNYBV1dO4Wg6KDU8sSiFAXf_JTUnFiwcicsyo2gysEKnLEo8ENWEARmBxtGByfnFwHtslIw0DMyi0WWM4LKGYHkLE1R5Iyhcn4gOUNjiFywEVjSLTogMzlbwSMzPSO1uCQWAJrYPmA=)

**When it works**: Unverifiable tasks (writing, creative tasks) requiring human judgment

**Training data**: Human-annotated (prompt, response, score) tuples

### Critical Distinction: Verifiable vs Unverifiable Tasks

> ** Key Insight**: This distinction determines which techniques work and drives major design decisions in reasoning systems.

**Verifiable Tasks** (math, coding, logic):
* **Characteristic**: Objective correct/incorrect answer exists
* **Auto-evaluation**: Parse answer and check against ground truth
* **Enables**:
  * Automatic ORM training data generation
  * Majority voting for best response selection
  * Auto-filtering of training data (keep only correct)
* **Examples**: "What is 23 × 17?", "Write function to reverse string", "Solve for x: 2x + 5 = 13"

**Unverifiable Tasks** (writing, creativity, analysis):
* **Characteristic**: Subjective quality, no single correct answer
* **Requires**: Human judgment and preference labeling
* **Enables**:
  * Reward model training (expensive)
  * Human preference data collection
  * Qualitative scoring
* **Examples**: "Write a compelling story opening", "Explain quantum physics simply", "Draft a persuasive email"

**Practical implications**:
* Verifiable tasks → Cheaper to train, easier to evaluate, use majority voting
* Unverifiable tasks → Expensive human labeling, must use reward models
* Production systems often separate routing based on task type

---

## Tree-of-Thoughts (ToT)

Treats decoding as a **search problem** with branching and pruning.

![Diagram 7](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaM-8gtKSWAVdXTsFp-iQjPzS9IwSBUM9w1iIPFjCGUnCCCLhBJZwgUsYwXRAJFyRJKA6nMESbkgSxhAJF7CEe7R_aQnIKWCx4pLKnFSglrTMnBwr5bS0ZCBAknBDkQAA2ME5BQ==)

### Method

1. Generate K candidate thoughts at each step
2. **Evaluate** each thought's promise (using heuristic or model)
3. **Prune** low-scoring branches
4. **Expand** high-scoring branches
5. Continue until reaching final answer

### Advantage over Self-Consistency

**Self-Consistency**: Generate N complete paths, score at end

**ToT**: Prune bad paths early → more efficient exploration

### Search Algorithms

**Common approaches**:

| Algorithm | Description | Key Feature |
|-----------|-------------|-------------|
| **Beam Search** | Keep top-k paths at each step | Fixed beam width |
| **Best-First** | Always expand highest-scoring node | Greedy |
| **Look-Ahead** | Roll out paths temporarily, backtrack if bad | Exploration with reset |
| **Monte Carlo Tree Search** | Balance exploration/exploitation | Used in AlphaGo |

**Paper**: *Scaling LLM Test-Time Compute Optimally* (DeepMind, 2024)

---

## Sequential Revision

**Also called**: Sequential sampling

Instead of parallel exploration, **iteratively refine** a single response.

![Diagram 8](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6ICi_NyCklgFXV07BadoHx_fWLCEE1jAOdq_tKSgtETBECLqDBZ1ASmzUghKLcssToVIuIAlXGHKjSCirmBRN0zlbmAJd5hy71gAWZYm1w==)

### Method

```python
output = generate(prompt)
for i in range(k_revisions):
    output = generate(f"Improve this: {output}")
return output

```

### When to Use

> ** Important Decision Rule**: Choose between sequential and parallel based on problem difficulty.

**Paper findings**:
* **Sequential**: Better for easy/medium questions (model usually starts in right direction)
* **Parallel**: Better for hard questions (need to explore multiple approaches)

**Why this works**:
* Easy/medium: LLM gets directionally correct on first try, just needs refinement
* Hard: LLM may pick wrong approach initially, need multiple attempts to find right direction

**Practical routing**:

```

If (problem_difficulty == "easy" or "medium"):
    Use sequential revision (k=2-3)
Else:
    Use parallel sampling (N=5-10)

```

### Combining Approaches

Production systems often combine:
* Sequential revision within each branch
* Parallel sampling across branches
* Reward model to select best final output

---

## Training with Reasoning

Instead of inference-time tricks, **train models to reason natively**.

### Comparison

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Inference-time** | Prompting, sampling | No retraining needed | Costs scale with each query |
| **Training-time** | Fine-tuning with reasoning data | Amortized cost, more reliable | Requires data + compute |

**Key insight**: Training-time methods can be combined with inference-time scaling for maximum performance.

---

## CoT Training

### STaR (Self-Taught Reasoner)

**Paper**: *STaR: Bootstrapping Reasoning With Reasoning* (2022)

**Problem**: How to get CoT training data?

![Diagram 9](https://kroki.io/mermaid/svg/eNpNzz0OwjAMBeCdU_gCuQKobfqDBExdUNQhaqw2Ek0qx9CBcndCigQeMvh9kl8G0vMIrdxBnEzlOiCcTucOhNhDripcRBg9Q-FbmMlPM1s3dEnnyRSqRoekGSE-1jt9w7CBIgH5rGxcgnZhQYLeE2HPh1ci8kPWK4YVSpUZA-zBaI4tuPsDF79CpaQNvSYDnmC07gvKdKRW8QgKvrutforqFDXqOMXmDzS_pNm-9wZye0l2)

### Iterative Process

1. **Generate rationales**: Use few-shot CoT prompting on questions
2. **Filter**: Keep only generations with correct final answers
3. **Fine-tune**: Train LLM on (question, rationale, answer) triples
4. **Repeat**: Use improved model to generate better rationales

**Optional**: For wrong answers, provide hints and regenerate

### Special Tokens

Define tags to separate reasoning from final answer:

**Common formats**:

```

<think>
Reasoning steps here...
</think>
Final answer here.

```

Or:

```

<scratchpad>
Working through the problem...
</scratchpad>
<answer>29</answer>

```

**DeepSeek-R1 tokenizer check**:

```

<think> → Token ID: 128798
</think> → Token ID: 128799

```

### Training Procedure

Once data is collected:

1. Apply template (add special tokens)
2. Standard next-token prediction training
3. Model learns to:
   * Generate `<think>` tag
   * Produce reasoning steps
   * Close with `</think>`
   * Output final answer

**Result**: Model natively generates reasoning without prompting

---

## RL with Reward Models

Improve reasoning quality through reinforcement learning.

![Diagram 10](https://kroki.io/mermaid/svg/eNpNj00KgzAQhfc9xVzARXuAgv8KaoNtV4ML0aCCmpAIpbevnWkxs3p831vMG0yrR3hEJ9jPR6HmqXtDUZQNeN4VAkzlKk27SajASKvVaqVtqB1QI8T6h-Hs8ujgF5fHB6-Yh8STnb9a00OpejmzidhQjp2cUE7x3inz_yYllmFdgD8PykzbuIAQN7YZ2Ryfuv9u4ZmsclL-B3yoRb0=)

### Method

1. **Rollout**: Generate multiple responses for each prompt
2. **Score**: Use reward model to evaluate each response
3. **Update**: Use RL (e.g., PPO) to increase probability of high-scoring responses
4. **Iterate**: Repeat until convergence

**Goal**: Train LLM to generate high-scoring responses in first attempt (reduce need for sampling)

---

## Outcome Reward Model (ORM)

**Scores only the final answer**, ignoring intermediate steps.

### Training Data Format

```

Input: [question] + [reasoning steps] + [final answer]
Label: 1 (correct) or 0 (incorrect)

```

### Data Collection

**Verifiable tasks** (math, coding):
* Generate responses with LLM
* Parse final answer
* Compare to ground truth
* Auto-label as 1/0

**Unverifiable tasks** (writing):
* Human annotators score quality
* Or use user feedback from production

### Training

```

Loss = CrossEntropy(predicted_score, true_label)

```

**Architecture**: Copy LLM, modify output layer to produce single scalar score

### Limitation

**Only looks at final answer**. Cannot detect:
* Incorrect reasoning steps that coincidentally reach correct answer
* Where in the chain of thought errors occur

---

## Process Reward Model (PRM)

**Scores each intermediate step** in the reasoning chain.

**Paper**: *Let's Verify Step by Step* (OpenAI, 2023)

### Training Data Format

```

Input: [question] + [thought_1] + [thought_2] + ... + [thought_n] + [answer]
Label: [score_1, score_2, ..., score_n, score_answer]

```

### Data Collection Challenge

**Problem**: Cannot automatically label intermediate steps (even for verifiable tasks)

**Solutions**:
1. **Manual annotation**: Humans score each step (expensive, doesn't scale)
2. **Monte Carlo Tree Search**: Use search algorithms to infer step quality (DeepMind paper)

### Comparison

| Aspect | Outcome Reward Model (ORM) | Process Reward Model (PRM) |
|--------|---------------------------|----------------------------|
| **Judges** | Final answer only | Every reasoning step |
| **Granularity** | Coarse (binary correct/incorrect) | Fine-grained (per step) |
| **Data Collection** | Easy (automated for verifiable) | Hard (requires human labeling or MCTS) |
| **Error Detection** | Late (at end) | Early (catches errors mid-reasoning) |
| **Search Integration** | Weak (only scores final) | Strong (prunes bad branches) |
| **Accuracy** | Good | Better (10-15% improvement) |
| **Training Cost** | Lower (data collection) | Higher (data collection) |
| **Inference Use** | Score complete responses | Guide ToT/beam search |
| **Production Adoption** | ✅ Common (simpler to implement) | ✅ Preferred when feasible |

### Dual Use of PRM

**Use Case 1: Training**
* Use PRM scores as rewards in RL
* Train policy to generate better reasoning chains

**Use Case 2: Inference-Time Search**
* During ToT or beam search, use PRM to score partial paths
* Prune low-scoring branches early
* Expand promising branches

**Example**:

```

Path A: score_1=0.9, score_2=0.8 → Continue
Path B: score_1=0.3 → Prune immediately

```

---

## Self-Correction

**Goal**: Model detects and fixes errors in its own output.

### Two Dimensions

**1. Compute source**:
* **Inference-time**: Prompt engineering to trigger revision
* **Training-time**: Fine-tune model to revise natively

**2. Feedback source**:
* **Intrinsic**: Model self-corrects without external signals
* **Extrinsic**: Uses reward model, verifier, or tool feedback

### Intrinsic Self-Correction (Inference)

![Diagram 11](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6ICi_NyCklgFXV07BadoHx_fWLCEE1jAOdq_tKSgtETBECLqDBZ1iVaC6LJSUHctS8wpTSxJVUjMS1FIy6xQSC0qyi8qVleCaHCBmItkpCvMSCOIClewqBvxRrphGOkOM9JYwS0zLzEnFgBKvz1z)

**Method**: Sequential revision via prompting

**Limitation**: No verification, just assumes last output is best

### Extrinsic Self-Correction (Inference)

Same process, but use **reward model to score** all outputs:

```

Score(Output 1) = 0.4
Score(Output 2) = 0.7  ← Pick this
Score(Output 3) = 0.6

```

**External feedback sources** (comprehensive list):
* **Reward Models**: ORM or PRM for quality scoring
* **Code Execution**: Run code and check for runtime errors
* **Unit Tests**: Pass/fail signals from test suites
* **Tool Outputs**: Results from function calls, API responses
* **Error Logs**: Stack traces, error messages from execution
* **Search Quality**: Relevance of retrieved information
* **Verification**: Mathematical proof checkers, logical validators

### Training for Self-Correction

> ** Critical Warning**: Simple SFT for self-correction does NOT work reliably.

**Supervised Fine-Tuning (SFT)** approach:

**Collect revision data**:

```

Format: [question] + [incorrect answer_1] + [incorrect answer_2] + ... + [correct answer]

```

**Fine-tune** LLM on revision trajectories

**Paper**: *Training Language Models to Self-Correct via RL* (DeepMind, 2024)

**Key finding**: Simple SFT for self-correction doesn't work well (not better than self-consistency)

**Proposed solution**: Specialized RL training with carefully designed reward penalties for revision process

> **Important**: Research from DeepMind shows that intrinsic self-correction via SFT is **NOT** more effective than simple self-consistency (parallel sampling). Only specialized RL methods (like SCORR) make self-correction reliable. Don't invest effort in SFT-based self-correction approaches without verification.

### Self-Correction in Production

**Practical approach** (extrinsic):
1. Model generates initial response
2. If it includes tool calls (e.g., code execution), run tools
3. Feed tool outputs (logs, errors) back to model
4. Model revises based on feedback
5. Repeat until success or max iterations

---

## Internalizing Search

**Goal**: Train models to perform search **within their reasoning**, not just via external algorithms.

### System 1 vs System 2 Reasoning

**System 1** (simple questions):
* Single reasoning path
* Directly go from question → steps → answer
* Fast, minimal exploration

**System 2** (complex questions):
* Multiple paths tried
* Backtracking when hitting dead ends
* Exploration of alternatives
* Slow, extensive search

### Meta Chain-of-Thought (Meta-CoT)

**Paper**: *Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought* (2025)

**Key idea**: Train on data with **sequences of CoT attempts**, not just single successful chains.

### Data Format

```

Question: [complex problem]

z_1: <think>First approach...</think> [realizes it won't work]
z_2: <think>Second approach...</think> [realizes it won't work]
...
z_k: <think>Better approach...</think> [realizes it's promising]
z_final: <think>Detailed solution...</think>
Answer: [final answer]

```

**Terminology**:
* **Latent thoughts** (z_1, z_2, ...): Exploratory reasoning chains that may be abandoned
* **Final CoT**: The successful reasoning chain leading to answer

![Diagram 12](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALH6MDS1OKSzPy8WAVdXTsFp2jHkpLU3IISBcNYsAInsLBztUtqYopCal5KLVjUGSzqEu2UmJxdUgQkIIpdwMKucDOMIMKuYGE3NDPcIGYgaXSHazSGaHQHC3tUBxTl52YWZ-alK0K0eoDFPaOd8_NKMvNKUyGqPcGiXtHB-TmlYB8BAK42PnA=)

### Training Process

Similar to STaR bootstrapping:

1. **Generate diverse solutions**: Use inference-time search (ToT, beam search) to collect attempts
2. **Label trajectories**: Mark which paths were abandoned vs. successful
3. **Fine-tune**: Train LLM on sequences including failed attempts
4. **Iterate**: Use improved model to generate richer training data

### Emergent Behaviors

> ** Key Insight: Adaptive Compute** - After Meta-CoT training, models automatically adjust reasoning depth based on question difficulty without manual configuration.

**After training**:
* Model generates latent thoughts (backtracking visible in output)
* Simpler questions → fewer latent thoughts (fast answer)
* Complex questions → many latent thoughts (extensive search)

**Adaptive compute**: Model automatically adjusts reasoning depth based on question difficulty

**How it emerges**:
* Training data has varying latent thought counts (few for simple, many for complex)
* Model learns correlation between difficulty and search depth
* At inference, model allocates compute proportionally to perceived difficulty
* No manual routing or configuration needed

**Example from ChatGPT o3**:

```

<Thinking indicator (no text shown)>  ← Inference-time search against verifier
<Visible latent thoughts>              ← Model-generated reasoning with backtracking
<Final answer>

```

**Inference time**: 3 minutes 32 seconds for complex question

### Evidence in Production Models

**Meta-CoT paper analysis** of o1, DeepSeek-R1, Gemini 2 Flash Thinking:

**Backtracking patterns found**:

```

DeepSeek-R1 output:
"Let me define this...
Wait. Actually, since this is a polynomial of degree 2...
[backtracks to earlier point]"

```

**OpenAI o1 inference time** correlates with question difficulty:
* Simple questions: ~10 seconds
* Complex questions: ~120+ seconds

---

## Deep Research

Combines reasoning models with multi-agent search.

**Announced**: OpenAI (February 2025)

**Definition**: "Agentic capability that conducts multi-step research on the Internet for complex tasks."

**Key feature**: Accomplishes in 10 minutes what takes humans hours.

### Architecture

![Diagram 13](https://kroki.io/mermaid/svg/eNpVzj0LwjAQBuDdX3G7dNDRQeh3hVbE1il0CHKkgSYpl4jorzckCG2m93jeIyeILxMMxQ78S9nDIsGNjFrcCElyhozFCe74JumQxtDMAuas45oLv5IK1C5SHqhgPXJ6TlHgsLZya8e1VVu7RiuC1WwwZraw98dwa7TUAtq283OHytAndsvYDbla5Trk5n_yCfqXUpzkF-NeE_zCKqn57H9YDLnxB0TfTx8=)

**Components**:

1. **Prompt Rewriter**: Fix grammar, clarify intent
2. **Manager Agent**: Coordinates parallel search agents
3. **Search Agents**: Each has:
   * Access to web search tools
   * **Reasoning LLM** (not standard LLM)
   * Memory of findings
4. **Summarizer**: Aggregates all findings into coherent report

### Reasoning LLM Role

**Old approach** (React pattern):
* Standard LLM + prompt engineering to elicit reasoning
* Manually specify thought → action → observation loop

**New approach** (Deep Research):
* **Reasoning LLM** handles thinking natively
* No need for complex prompting
* Model automatically:
  * Breaks down complex queries
  * Explores multiple search strategies
  * Backtracks when hitting dead ends
  * Synthesizes information

### Recent Update (July 2025)

**Visual browser access**: Deep Research can now:
* Navigate web pages visually
* Extract information from rendered pages
* Handle JavaScript-heavy sites

### User Experience

1. User enters research query
2. Model asks **clarifying questions**
3. After clarification, begins research:
   * Shows sources being accessed
   * Progress updates visible
4. Typical duration: 10-30 minutes
5. Returns comprehensive report with citations

---

## How Production Systems Combine Techniques

> ** Key Insight**: Real reasoning models don't use just one technique. They combine ALL approaches for maximum effectiveness.

**Example: ChatGPT o3 Multi-Phase Pipeline**

**Phase 1: Hidden "Thinking" (not shown to user)**
* Inference-time search against verifiers
* Beam search or ToT with PRM pruning
* Generates multiple high-level approaches
* Selects most promising paths
* **Duration**: Variable based on complexity

**Phase 2: Visible Reasoning Traces**
* Model-generated internalized search (Meta-CoT style)
* Shows latent thoughts with backtracking
* User sees: "Wait, actually..." and "Let me reconsider..."
* Adaptive compute: Simple questions finish quickly
* **Duration**: 10 seconds to 3+ minutes

**Phase 3: Tool Integration**
* Code execution, web search, calculations
* Extrinsic feedback from tool results
* Self-correction based on execution errors
* Iterative refinement with external signals

**Phase 4: Final Selection**
* Multiple candidate answers generated
* Scored with reward model (possibly PRM)
* Highest-scoring response selected
* Presented to user with confidence

**Techniques combined**:
* Inference-time search (Phase 1)
* Trained internalized search (Phase 2)
* Extrinsic self-correction (Phase 3)
* Reward model selection (Phase 4)
* Adaptive compute (throughout)

---

## Production Deployment Considerations

> ** Practical Guidance**: Reasoning models fundamentally change deployment economics and system design.

### Cost Management

**Token Economics**:
* Reasoning models generate **10-100x more tokens** per response than standard models
* Each reasoning token incurs full inference cost
* Example: Simple math question
  * Standard LLM: ~50 tokens ($0.0001)
  * Reasoning LLM: ~2,000 tokens ($0.004) - 40x cost

**Strategies**:
* Monitor token costs per query type and user
* Cache reasoning traces for frequently asked questions
* Consider pre-computing answers for common queries
* Implement budget limits per user/session

### Request Routing

**Complexity-Based Routing**:

```

Simple queries (facts, definitions, basic math):
  → GPT-4o, Gemini Flash (fast, $0.0001/query)

Medium queries (analysis, multi-step problems):
  → o4-mini, smaller reasoning models (balanced, $0.001/query)

Complex queries (research, advanced math, novel problems):
  → o3, o3-pro (best quality, $0.01-0.10/query)

```

**Detection methods**:
* Explicit user selection (UI dropdown)
* Keyword heuristics ("prove", "research", "complex")
* Classifier model (train on query difficulty)
* Default to fast, escalate if confidence low

### Performance Optimization

**Latency Management**:
* Set **maximum thinking time** based on use case:
  * Chat interface: 30-60 seconds max
  * Research tasks: 5-10 minutes acceptable
  * Background jobs: No limit
* Implement timeouts to prevent infinite loops
* Stream intermediate thoughts for user feedback

**Quality Monitoring**:
* Track reasoning token count vs. accuracy
* Monitor self-correction iterations (should stabilize)
* Watch for reward model score distributions
* Detect if model generates too many/few latent thoughts

### System Architecture

**Parallel Agent Systems**:
* Spawn multiple reasoning agents for research tasks
* Each agent explores different information sources
* Manager agent aggregates findings
* Trade latency for throughput (parallel > sequential)

**Caching Strategies**:
* Cache full reasoning traces (not just final answers)
* Index by semantic similarity, not exact match
* Reuse partial reasoning for related queries
* Update cache with higher-quality responses

**Fallback Logic**:

```python
def query_handler(prompt):
    complexity = detect_complexity(prompt)

    if complexity == "simple":
        return fast_llm(prompt)

    if complexity == "medium":
        response = reasoning_llm(prompt, max_time=30)
        if response.confidence < 0.8:

            # Escalate to better model
            return advanced_reasoning_llm(prompt, max_time=60)
        return response

    # Complex: use full reasoning pipeline
    return deep_research(prompt)

```

---

## Key Insights

### Fundamental Trade-offs

**Inference-time compute**:
*  No retraining needed
*  Flexible (adjust N on the fly)
*  Costs scale with every query
*  High latency

**Training-time compute**:
*  One-time training cost
*  Fast inference (amortized)
*  Requires high-quality data
*  Less flexible after training

### Task Categorization

**Verifiable tasks** (math, coding, logic):
* Easy to auto-evaluate (parse answer, run code)
* ORM training data is automatable
* Majority voting works well
* Example: "What is 23 × 17?"

**Unverifiable tasks** (writing, creativity):
* Require human judgment
* Need expensive human labeling
* Must use reward model for scoring
* Example: "Write a compelling story opening."

### Reward Model Hierarchy

**Effectiveness**: PRM > ORM > Heuristics

**Data cost**: PRM >> ORM > Heuristics

**Inference use**: PRM enables search pruning, ORM only final scoring

### Search Internalization

**Traditional**: Search algorithm external to model (ToT, beam search)

**Modern**: Model trained to search internally (Meta-CoT, o-series)

**Benefit**: Adaptive compute (simple questions fast, complex questions get more time)

### Self-Correction Reality

> ** Critical Finding**: Intrinsic self-correction via SFT does NOT reliably work.

**Important finding**: Intrinsic self-correction via SFT **does not reliably work**

**What works**:
* Extrinsic self-correction (with verifier feedback)
* Specialized RL training (not just SFT)
* Tool-in-the-loop (code execution, unit tests)

### Production Systems

Real reasoning models combine **all techniques**:
* Inference-time search against verifiers (initial pruning)
* Trained internalized search (visible reasoning)
* Self-correction with external feedback
* Multi-agent parallelization

**Example**: ChatGPT o3
1. Thinking phase (no output) → Inference-time search
2. Visible reasoning → Internalized search
3. Tool calls → Extrinsic feedback
4. Final answer

### Sequential vs Parallel - Decision Guide

**When to use sequential**:
* Problem appears straightforward
* Model likely to start in right direction
* Budget constrained (sequential cheaper)
* Low latency requirement

**When to use parallel**:
* Problem is novel or complex
* Multiple valid approaches exist
* Accuracy critical (worth extra cost)
* Can afford higher latency

**Hybrid approach** (production best practice):
* Start with sequential (k=1-2 revisions)
* If confidence low, switch to parallel (N=3-5)
* Use reward model to detect low confidence

---

## Quick Recall / Implementation Checklist

### Inference-Time Scaling

* [ ] **Zero-shot CoT**: Add "Let's think step by step" to prompts (free improvement)
* [ ] **Few-shot CoT**: Show 1-3 examples with reasoning steps
* [ ] **Self-consistency**: Sample N=5-10 responses, use majority vote or reward model
* [ ] **Sequential revision**: Better for easy questions (k=2-3 iterations)
* [ ] **Parallel sampling**: Better for hard questions (explore diverse approaches)
* [ ] Use **temperature > 0.7** for self-consistency (need diversity)
* [ ] Parse final answers carefully (use structured output or regex)

### Training Approaches

* [ ] **STaR for CoT**: Bootstrap rationales from few-shot prompting → fine-tune
* [ ] Define **special tokens** (`<think>`, `</think>`) for reasoning/answer separation
* [ ] **ORM training**: Collect (prompt, reasoning, answer, label) for verifiable tasks
* [ ] **PRM training**: Human-label each reasoning step (expensive but effective)
* [ ] Use RL (PPO) only with carefully designed rewards (SFT alone insufficient for self-correction)

### Reward Models

* [ ] **ORM**: Use for final answer scoring (easier to train)
* [ ] **PRM**: Use for search guidance and step-level feedback (better but costly)
* [ ] Reward models can serve dual purpose: training (RL) and inference (search)
* [ ] For verifiable tasks, automatic labeling is possible
* [ ] For unverifiable tasks, collect human preference data

### Search and Exploration

* [ ] **Beam search**: Keep top-k paths at each step (k=2-3 typical)
* [ ] **Prune early**: Use PRM to cut bad branches before full generation
* [ ] **Look-ahead**: Roll out short paths to evaluate, backtrack if poor
* [ ] **Adaptive depth**: Let model decide reasoning length based on difficulty

### Self-Correction

* [ ] **Don't rely on intrinsic SFT** (proven ineffective)
* [ ] **Use external feedback**: Reward model scores, code execution, unit tests
* [ ] **Tool-in-the-loop**: For coding tasks, run and feed errors back
* [ ] Comprehensive feedback sources: RM, code execution, unit tests, tool outputs, error logs
* [ ] Limit revision iterations (k=2-3) to avoid degradation
* [ ] Track which revision performed best (don't assume last is best)

### Production Deployment

* [ ] Combine inference-time and training-time methods
* [ ] Use reasoning LLMs for agentic systems (no React prompting needed)
* [ ] Implement **multi-agent parallelization** for research tasks
* [ ] Monitor **cost per query** (reasoning tokens add up 10-100x)
* [ ] Cache reasoning traces for similar queries
* [ ] Set **max thinking time** based on use case urgency (chat: 30s, research: 10min)
* [ ] Fall back to standard LLM for simple queries (routing logic)
* [ ] Implement complexity detection for request routing
* [ ] Track token costs per query type and user
* [ ] Set budget limits per user/session

### Model Selection

* [ ] **o-series**: Best reasoning, high cost, high latency
* [ ] **DeepSeek-R1**: Open-source alternative, shows raw traces
* [ ] **GPT-4o/Gemini**: Fast standard models for simple tasks
* [ ] Route requests based on detected complexity
* [ ] Test smaller reasoning models (Phi-4, Qwen) for cost optimization
* [ ] Use dropdown UI for explicit user complexity selection

### Data Quality

* [ ] For CoT training: Filter for correct final answers before adding to dataset
* [ ] For Meta-CoT: Include unsuccessful attempts (latent thoughts) in data
* [ ] For self-correction: Collect revision trajectories (not just final)
* [ ] Iterate bootstrapping: Use improved model to generate better data
* [ ] Balance simple vs. complex examples (models learn to match difficulty)

### Debugging

* [ ] If CoT fails: Check if special tokens are in vocabulary
* [ ] If self-consistency doesn't help: Increase temperature or N
* [ ] If search is ineffective: Verify PRM is scoring correctly
* [ ] If revisions degrade: Check for reward hacking or insufficient feedback
* [ ] Monitor latent thought count (should vary with question difficulty)
* [ ] Check reasoning token count vs. accuracy correlation

### System Architecture

* [ ] Implement parallel agent spawning for research tasks
* [ ] Build caching layer for reasoning traces (semantic similarity)
* [ ] Create fallback pipeline (fast → medium → advanced)
* [ ] Monitor self-correction iterations (should stabilize at k=2-3)
* [ ] Implement timeout logic to prevent infinite reasoning loops
* [ ] Stream intermediate thoughts for real-time user feedback

### Resources

**Key Papers**:
* [**Zero-shot CoT:** "Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916)
* [**Self-consistency:** "Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171)
* [**STaR:** "Bootstrapping Reasoning With Reasoning"](https://arxiv.org/abs/2203.14465)
* [**PRM:** "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050)
* [**Scaling inference:** "Scaling LLM Test-Time Compute Optimally"](https://arxiv.org/abs/2408.03314)
* [**Self-correction RL:** "Training Language Models to Self-Correct via RL" (DeepMind SCORR)](https://arxiv.org/abs/2409.12917)
* [**Meta-CoT:** "Towards System 2 Reasoning in LLMs"](https://arxiv.org/abs/2501.04682)

**Leaderboards**:
* [**LMSys Arena**](https://lmarena.ai/leaderboard/text) - Track reasoning model rankings

**Open Models**:
* [**DeepSeek-R1**](https://github.com/deepseek-ai/DeepSeek-R1) - MIT license, shows actual reasoning traces
* [**Kimi k2**](https://github.com/MoonshotAI/Kimi-k2) - Modified MIT, competitive performance
