# Adapting LLMs and Retrieval-Augmented Generation (RAG)

## Table of Contents

* [Overview](#overview)
* [Core Concepts](#core-concepts)
  * [Why LLM Adaptation is Needed](#why-llm-adaptation-is-needed)
* [Fine-Tuning](#fine-tuning)
  * [Overview](#overview-1)
  * [Two Fine-Tuning Approaches](#two-fine-tuning-approaches)
  * [Practical Implementation with Hugging Face PEFT](#practical-implementation-with-hugging-face-peft)
* [Prompt Engineering](#prompt-engineering)
  * [Overview](#overview-2)
  * [Prompt Engineering Techniques](#prompt-engineering-techniques)
  * [Applying Prompt Engineering to Retail Store Use Case](#applying-prompt-engineering-to-retail-store-use-case)
  * [Limitation of Prompt Engineering](#limitation-of-prompt-engineering)
* [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  * [Overview and Motivation](#overview-and-motivation)
  * [RAG Components](#rag-components)
  * [Retrieval Component](#retrieval-component)
  * [Generation Component](#generation-component)
  * [RAG Evaluation](#rag-evaluation)
  * [Complete RAG System Design](#complete-rag-system-design)
* [Key Insights](#key-insights)
  * [Fundamental Principles](#fundamental-principles)
  * [Important Distinctions and Comparisons](#important-distinctions-and-comparisons)
  * [Practical Considerations](#practical-considerations)
  * [Common Pitfalls and Misconceptions](#common-pitfalls-and-misconceptions)
* [Quick Recall / Implementation Checklist](#quick-recall--implementation-checklist)
* [References](#references)

---

**TL;DR**: Adapt LLMs to domain-specific needs through fine-tuning (PEFT/LoRa for 99%+ parameter reduction), prompt engineering (few-shot, zero-shot, chain-of-thought), or RAG (retrieval + generation). RAG dominates for large knowledge bases: parse documents (AI-based), chunk intelligently (1000 chars, 20% overlap), embed with quality models (OpenAI text-embedding-3), index with FAISS (ANN search), retrieve top-K (5-10), and generate with guardrails. Evaluate across context relevance, faithfulness, and answer correctness. Vector search > keyword search; PEFT > full fine-tuning; hybrid approaches often optimal.

---

## Overview

This lecture notes provides a comprehensive guide to adapting general-purpose Large Language Models (LLMs) for specialized, domain-specific applications. The primary focus is building a customer support chatbot for an e-commerce retail store. The lecture covers three major adaptation techniques: fine-tuning, prompt engineering, and Retrieval-Augmented Generation (RAG). Special emphasis is placed on RAG systems, including detailed coverage of document processing, embedding models, vector search algorithms, and complete system design with practical implementation considerations.

## Core Concepts

### Adaptation Technique Decision Tree

![Adaptation Technique Decision Tree](https://kroki.io/mermaid/svg/eNplk8tO60AMhvfnKbw5q1KxroRA0BvQUCFOWaAqCzfjpgNziWYmBU7Du-NMEjWF7Eb2_9n-7eQOix2sJn-Av-v1kkhAsIACiwBJ8lA_hNUozVUKw-El3By6nGyHJqeLjTu_1FaQgg3tcC-tg4KcRkMmqM-rr0i-qbXVC_mzY1ajr2B8-KdRKRAY0FOIQA4rdDlBJJ9AlvYMXksfuEcBb8a-KxI1ZnJYdA_YMAi8_E-sjNJxlCZHZAXT9TMnPU5nq_PEPmEsOxr9HUCBDjUFcuBIlFmQ1qQ9yEmzFczWY2u8FJy-LTmwlYaGoTTS5N0k2jqKAsisUtQSI3ISkTMZPEjDYRPoo3FgS-9sfFZqttFXMG9sd4RqGKQmKAsmkm-tmfTmezuxodmPVIqLerDbGsq82zj90_U8xh0FJ2mPCgaQkyGHvR7n3e4quGs8c1bzdUxNzrOS6yaVJlOloFignqaIaWmPUa_OB2ZnFdx3DYB1Uf6b2ta_jXe3aAzIarN9YFN-NN6dVWvI4th00laaraKkLjnoryntCZa2gocTa_a8L96hJ3TZjoX8T6TfMfAOEw==)

### Why LLM Adaptation is Needed

**General-Purpose LLMs vs. Domain-Specific Needs**:
* General-purpose LLMs excel at common tasks:
  * Math questions ("What is 2+2?")
  * Brainstorming requests ("Help me write an email to my manager")
  * Coding assistance ("My code is not running, read it and fix the bug")
* These models perform well because training data includes similar examples

**The Hallucination Problem**:
* Domain-specific queries cause general-purpose LLMs to hallucinate
* Example: Asking ChatGPT "What is your refund policy?" for a retail store
  * ChatGPT responds about OpenAI's refund policy (incorrect context)
  * Lacks knowledge of the specific retail store's policies
* Without domain-specific information, LLMs generate inaccurate or irrelevant responses

**Problem Statement**: Adapt a general-purpose LLM to accurately answer questions in a specific domain using additional documents (internal knowledge base, PDFs, HTMLs, Wikis, etc.)

**Three Main Adaptation Techniques**:
1. **Fine-Tuning**: Continue training the LLM on domain documents
2. **Prompt Engineering**: Craft prompts to guide LLM behavior
3. **Retrieval-Augmented Generation (RAG)**: Retrieve relevant documents and provide them as context

| Technique | Training Required? | Cost | Update Speed | Knowledge Location | Best For | Limitations |
|-----------|-------------------|------|--------------|-------------------|----------|-------------|
| **Fine-Tuning** | ✅ Yes (PEFT) | Medium | Slow (retrain needed) | Model weights | Consistent behavior changes | Cannot easily update knowledge |
| **Prompt Engineering** | ❌ No | Very Low | Instant | Prompt context | Quick iteration, guiding format | Context window limits |
| **RAG** | ❌ No (for base) | Low-Medium | Fast (re-index) | External database | Large/dynamic knowledge bases | Retrieval quality dependency |

> **Key Insight**: Three complementary adaptation techniques serve different needs. Fine-tuning modifies model weights permanently (best for consistent behavioral changes), prompt engineering guides without training (flexible, immediate), and RAG separates knowledge from model (easy updates, handles large knowledge bases). Combined approaches are often optimal - RAFT combines RAG + fine-tuning.

## Fine-Tuning

### Overview

**Definition**: Continue training a general-purpose LLM on domain-specific document database

**Process**:

![Diagram 1](https://kroki.io/mermaid/svg/eNodjEEKwjAUBfee4l8geAEp2AZFiCDVXcjiN32tgfYnpM1CT69ktsPMnDm9yfQH-nO2VwgyL_QoOcUNZMzdkVINtfYSBOpVJMhMUU5DPjY6-rJCdtK888AbXN20tejsM8EHXsIXYx1V2VWpbQ8ePzTFTDeZkCEe7gcnFyvG)

**How it Works**:
* Feed domain documents (PDFs, HTMLs, knowledge base) to training algorithm
* Update model weights through continued training
* Output: Specialized LLM with tuned weights
* Responses come directly from compressed knowledge in learned weights

**Example**:
* Question: "What is the return policy?"
* Fine-tuned response: "You can return items for any reason within 30 days of your purchase"
* Answer derived from learned weights without explicit document access at inference time

### Two Fine-Tuning Approaches

#### Option 1: Updating All Parameters

**Method**: Allow optimizer to tune all weights in the LLM during training

**How it Works**:
* LLM architecture: Multiple transformer blocks
* Each block contains attention layers and MLP layers
* Each linear layer has weight matrices
* Optimizer updates every weight across all layers

**Limitation**: **Computationally very expensive**
* Modern LLMs have billions of parameters
* Training requires substantial computational resources
* Not practical for most organizations

#### Option 2: Parameter-Efficient Fine-Tuning (PEFT)

**Definition**: Adapt LLM by updating only a subset of parameters

**Advantages**:
* Significantly reduces computational cost
* Maintains most of the original model frozen
* Learns new behavior with minimal parameter updates

**Popular PEFT Techniques**:
* **Adapters**
* **LoRa** (Low-Rank Adaptation)
* Prompt tuning
* Activation scalers
* Bias-only
* Sparse weight deltas

##### Adapters

**Concept**: Inject new trainable layers into frozen LLM architecture

**Architecture**:

![Diagram 2](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaM-8gtKSWAVdXTsFp2i3ovyq1DwFn8TK1CIFw1iwEiewnHO0Y0piQQlQGCxpk1SkbxdSlJiZl5iUkwpR6AxW6IJqiBFEzgUs50rYEFewQjdUQ4whcm5gOXfChriDFXpE-5eWgDwHAHaXRwk=)

**Implementation**:
* Original LLM layers remain frozen (gray in diagram, weights unchanged)
* New adapter layers added between existing layers (red in diagram, trainable)
* Only adapter layers updated during training
* Original model knowledge preserved while learning new task-specific behavior

**Original Paper**: "Parameter-Efficient Transfer Learning for NLP" (2019)
* Injects adapter layers inside transformer architecture
* After attention and MLP layers
* Only adapters learned during fine-tuning

##### LoRa (Low-Rank Adaptation)

**Concept**: Add low-rank matrices to linear layers while freezing original weights

**How it Works**:

For each linear layer with weight matrix W (size d_out × d_in):
* Keep W frozen
* Add two low-rank matrices: B and A
* Original computation: y = Wx
* LoRa computation: y = Wx + BAx

**Architecture**:

![Diagram 3](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALHaM-8gtIShYpYBV1dOwWnaLei_KrUPIWAxJIMK5ukIn278EPbK2IhasFKnKN98oMSkRQ4HdruCFfjBFbjEu2YkpJZkpmfBxF1hoiC2S5gtmu0f2kJ0OJYAOANKTQ=)

**Key Properties**:
* Original weights W remain unchanged
* Low-rank matrices B and A are trainable
* Output of both paths have same shape and can be added
* Targets all linear layers in the model

**Advantages over Adapters**:
* Faster at inference time
* No additional latency from extra layers
* Can merge LoRa weights with original weights after training

**Original Paper**: "LoRA: Low-Rank Adaptation of Large Language Models"

### Practical Implementation with Hugging Face PEFT

**Library**: Hugging Face PEFT (Parameter-Efficient Fine-Tuning)

**Code Example**:

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-3B-Instruct")

# Model has 3.085 billion parameters

# Create LoRa configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",

    # Additional configuration parameters...
)

# Wrap model with PEFT
peft_model = get_peft_model(model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()

# Total parameters: 3.09B

# Trainable parameters: 3.15M (0.1% of total)

```

**Key Observations**:
* Only 0.1% of original parameters are trainable with LoRa
* Massive reduction in training computational requirements
* Easy to use with just a few lines of code
* After training, save and load the LoRa-adapted model for inference

**Inference Example**:

```python
from peft import PeftModel

# Load LoRa-adapted model
model = PeftModel.from_pretrained(base_model, "path/to/lora/model")

# Generate response
input_text = "What is the refund policy?"
output = model.generate(input_text)

# Response aligned with fine-tuning data

```

## Prompt Engineering

### Overview

**Definition**: Craft or design prompts to extract desired output from the LLM

**Standard Workflow**:

![Diagram 4](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6NDi1CKFwNLUospYBV1dOwWnaB8f31iwpBNYwDk6KLW4ID-vODUWAPNUEM4=)

**Prompt Engineering Workflow**:

![Diagram 5](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6NDi1CKFwNLUospYBV1dOwWn6ICi_NyCEgXXvPTMvNTUosy8dJukIn07n_z0zORYsC4nsErnaNe8jMS85NQUBYgWiKQzWNIl2sfHFyLgAhZwjQ5KLS7IzytOjQUAFPAmQg==)

**Key Difference**: Transform user's raw query before passing to LLM

### Prompt Engineering Techniques

#### Few-Shot Prompting

**Concept**: Show examples to the LLM in the prompt

**Purpose**: Guide generation format by demonstrating desired behavior

**Base Model Example** (without few-shot):

```

Input: "How can I learn machine learning?"
Output: "A few years ago, I was wondering the same thing. I had just started my PhD..."
(Irrelevant continuation, not answering the question)

```

**With Few-Shot Prompting**:

```

Q: What is a good place to eat?
A: Berkeley has lots of good restaurants.

Q: What is the process of leasing a car?
A: You can go to a dealership and talk with a salesman.

Q: What is the capital of France?
A: Paris.

Q: How can I learn ML?
A: (Model generates answer following the format)

```

**Result**: Model follows Q&A format and provides direct answer

**Post-Trained Model Example**:

Without few-shot:

```

Input: "John has 4 books and buys 3 more. How many books does he have?"
Output: "Easy one. John has 4 books initially and buys 3 more. So 4 plus 3 equals 7. John now has 7 books."
(Verbose, explains reasoning)

```

With few-shot:

```

Q: You have 3 apples and get 2 more, how many apples do you have?
A: 5

Q: John has 4 books and buys 3 more. How many books does he have?
A: 7

```

**Result**: Model generates concise answer in exact format shown

**Use Cases**:
* Control output format
* Guide response style (concise vs detailed)
* Ensure consistency across responses
* Teach specific answer structures

#### Zero-Shot Prompting

**Concept**: Guide generation without showing examples

**Method**: Provide format hints or instructions directly in the prompt

**Base Model Example**:

Without zero-shot:

```

Input: "How can I learn ML?"
Output: "Where do I start? I want to learn ML. Where should I start?"
(Asking more questions instead of answering)

```

With zero-shot (adding Q: and A:):

```

Input: "Q: How can I learn ML?
A:"
Output: "You can learn ML by reading books, watching videos, taking online courses, or attending boot camps."
(Provides answer)

```

**Key Observation**: Simple format markers (Q: and A:) guide base model to answer instead of continuing with questions

**Structured Output Example**:

Without structure:

```

"Extract the candidate's years of experience from this resume."
(Unpredictable output format)

```

With structured zero-shot:

```

"Extract the candidate's years of experience from this resume.
Return only JSON like: {"years_experience": <number>}"

```

**Result**: LLM outputs structured JSON format as specified

**Common Applications**:
* JSON output formatting
* Specific answer templates
* Controlled response structure
* Format enforcement without examples

#### Chain of Thought (COT) Prompting

**Definition**: Encourage LLM to think step-by-step through reasoning process

**Origin**: Google Brain paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (January 2023)

**Problem**: LLMs fail on difficult reasoning tasks without intermediate steps

**Standard Few-Shot Example** (fails):

```

Q: Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. How many tennis balls does he have?
A: 11

Q: The cafeteria had 23 apples. If they use 20 to make lunch and bought 6 more, how many apples do they have?
A: 27  (INCORRECT - should be 9)

```

**Chain of Thought Few-Shot Example** (succeeds):

```

Q: Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. How many tennis balls does he have?
A: Started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they use 20 to make lunch and bought 6 more, how many apples do they have?
A: The cafeteria had 23 apples originally. They used 20 to make lunch, so they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9.  (CORRECT)

```

**Impact**:
* PaLM 540B model: 18% solve rate → 57% solve rate (just by adding reasoning traces)
* Same model, no additional training
* Only difference: showing reasoning steps in examples

**Zero-Shot Chain of Thought**:

**Innovation**: "Let's think step by step" (Google DeepMind paper, 19 days after original COT paper)

**Method**: Add single phrase to prompt without examples

Without zero-shot COT:

```

Q: What is 15 divided by 3 plus 2?
A: 8  (Incorrect, no reasoning shown)

```

With zero-shot COT:

```

Q: What is 15 divided by 3 plus 2? Let's think step by step.
A: First, we divide 15 by 3, which equals 5. Then we add 2 to get 7. The answer is 7.

```

**Key Insight**: Single instruction phrase triggers reasoning behavior
* No examples needed
* Model breaks task into smaller steps
* Dramatically improves accuracy on reasoning tasks

**Foundation for Future Models**: Chain of thought became basis for training reasoning models (O1, O3) covered in later weeks

#### Role-Specific Prompting

**Concept**: Assign a role to the LLM to improve response quality

**Standard Prompt**:

```

"How can I reduce my tax burden?"

```

**Role-Specific Prompt**:

```

"You are an expert tax adviser.

How can I reduce my tax burden?"

```

**Result**: LLM generates responses aligned with assigned expertise

**Common Roles**:
* Expert tax adviser
* Helpful customer support agent
* Professional software engineer
* Medical professional
* Legal consultant

**Why it Works**: Guides LLM to activate domain-specific knowledge and response style

#### System Prompt vs User Prompt

**Two-Component Architecture**:

**System Prompt**:
* Hidden instructions provided to LLM
* Not visible to end user
* Result of prompt engineering efforts
* Specifies:
  * Role assignment
  * Output format instructions
  * Few-shot examples
  * Chain of thought guidance
  * Any other behavioral rules

**User Prompt**:
* Visible query typed by user
* Actual question or request
* Combined with system prompt before LLM processing

**Combined Prompt Structure**:

![Diagram 6](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6ODK4pLUXIWAovzcghKbpCJ9O4_MlJTUPAXPvOKSotLkksz8vOJYBV1dOwXnaOf83KTMvNSUWLBmp-jQ4tQiZK1gfmBpalElVAdYnTOY7RLt4-ML0egCFnCNDkotLgCanhoLAMCXLRc=)

**Example**:

System Prompt:

```

You are a helpful assistant for TechStore customer support.

Your task is to deliver concise and accurate responses.
Always cite the source of your information.

Example format:
Q: What is the shipping time?
A: Standard shipping takes 5-7 business days. [Source: Shipping Policy]

User Query:

```

User Prompt:

```

What is your refund policy?

```

**User Experience**: User only types query, system prompt applied automatically

**Implementation**: All major chatbot services (ChatGPT, Claude, Gemini) provide system prompt configuration

### Applying Prompt Engineering to Retail Store Use Case

**Goal**: Adapt general-purpose LLM for retail store customer support

**Approach**: Include internal documents in prompt

**System Prompt Design**:

```

Read the following documents and respond to the question:

[Document 1: Refund Policy]
Customers can return items for any reason within 30 days of purchase...

[Document 2: Shipping Information]
Standard shipping typically takes 14 days...

[Document 3: Customer Service Contact]
For additional questions, contact us at...

User Query: What is your refund policy?

```

**How it Works**:
* LLM receives both documents and question
* Has necessary context to answer accurately
* No hallucination because information is provided
* Response based on given documents

**Advantage**: No fine-tuning required, works with general-purpose LLM

### Limitation of Prompt Engineering

**Problem**: Context window constraints

**Context Window**: Maximum number of tokens LLM can process as input

**Scenario**: Large enterprise with extensive documentation
* Thousands of PDF pages
* Millions of internal documents
* Cannot fit all documents in single prompt

**Calculation Example**:
* 1,000 PDF documents
* Average 10 pages per PDF = 10,000 pages
* Exceeds typical context windows (even extended ones)

**Issues**:
1. **Length Limit**: Physically impossible to include all documents
2. **Computational Cost**: Processing extremely long contexts is expensive
3. **Inefficiency**: Including irrelevant documents wastes tokens and compute

**Solution**: Need method to select only relevant documents (leads to RAG)

> **Key Insight**: RAG solves the context window limitation that makes prompt engineering impractical at scale. Cannot include millions of documents in prompts due to token limits and compute costs. Retrieval narrows huge databases to handful of relevant chunks (top-K=5-10), making domain adaptation practical for enterprises with extensive documentation.

## Retrieval-Augmented Generation (RAG)

### Overview and Motivation

**Problem with Prompt Engineering**: Cannot include entire document database due to context window limits

**RAG Solution**: Add retrieval component that selects only relevant documents

**Architecture Comparison**:

Prompt Engineering:

![Diagram 7](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6NDi1CKFwNLUokoFbZukIn07Rx8fBZf85NLc1LyS4lgFXV07BadoHx_fWLAGJ7CAc3RQanFBfl5xaiwApWIXQQ==)

RAG:

![Diagram 8](https://kroki.io/mermaid/svg/eNotjEEKhTAMRPeeIhfwCoJavxs3Cq6Ki3wJX0FbSavg7c1PndUw82Z-jMcC3ZCBqLRjIIb-JL4nyPMCKjtQ5JUu3CZFamv8fO7kIhiM-MVAL6l1pd7IapPNH3rpkOZG-8a25Igxrt6lvEy5-kb9Rz7C4Z38P46ULvo=)

**Key Innovation**: Retrieval filters document database to find relevant pieces

**How RAG Works**:
1. **Retrieval** examines user query and document database
2. Identifies which documents or document portions are relevant
3. Outputs k retrieved documents (retrieve_1, retrieve_2, ..., retrieve_k)
4. **Generation** (LLM) receives user query + retrieved documents only
5. LLM generates response based on relevant context

**Advantage**: Narrow down huge database to manageable context size

### RAG Components

**Two Main Components**:
1. **Retrieval**: Search problem to find relevant documents
2. **Generation**: LLM with potential enhancements

### Retrieval Component

#### Overview

**Purpose**: Search document database to find relevant text pieces

**Two Main Steps**:
1. **Build searchable index** (preprocessing, done once offline)
2. **Search from index** (runtime, done for each query)

**Index Building Process**:

![Diagram 9](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALHaJf85NLc1LwSBZfEksSkxOJUm6QifbsAF7diHQWPEF8fIOWZm5ieWhyroKtrp-AU7ZmXklqh4FSamZOSmZcOVu2Tn56ZHAs20Amsyjk6ODWxKDkjMSknVQGsIRYAsz8lTA==)

**Search Process**:

![Diagram 10](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6NDi1CKFwNLUospYBV1dOwWn6ODUxKJkoIr89MzkWLAqZ6hYYlJOqoJnXkpqBVQtWNYJzHaJDkotKcpMLUtNUXDOKM3LLo4FAJZmH9Y=)

#### Building Searchable Index: Three Steps

**Step 1: Document Parsing**
**Step 2: Document Chunking**
**Step 3: Indexing**

##### Step 1: Document Parsing

**Goal**: Convert documents (PDFs, HTMLs) to extracted text content

**Problem**: LLMs require text input, not PDF binary format

**Two Approaches**:

**1. Rule-Based Parsing** (less common)
* Uses predefined rules and heuristics
* Limited flexibility
* Not widely used in modern systems

**2. AI-Based Parsing** (standard approach)
* Uses machine learning models
* More accurate and flexible
* Industry standard

**Parsing Process**:

![Diagram 11](https://kroki.io/mermaid/svg/eNoljEEKgzAURPc9xT-A0n0pQk0UChZLm13IIuqvBtRI_AG9fWOc3TBvXu_0MoDgFwh5yDcvgdvWTziTgjTNIJeV3q0n4EjYkrGzimweVyafXUDNbwdmp8XOoay3e-OumTA0YgICN0qgNL13R9PNiKeARQGXBwDFRk5He_zW7AMvpMF26wnzCBfyS863FFQd1J4WT-oPl_o8qA==)

**Layout Detection Example**:
* Identify title regions
* Locate body text blocks
* Detect figures and images
* Find tables
* Each component marked with bounding box

**Text Extraction**:
* Apply OCR to text regions
* Extract text from each detected block
* Maintain reading order
* For images: Store coordinates for later processing

**Structured Output**:

```

[Text Block]
Content: "This is the introduction..."
Type: paragraph

[Text Block]
Content: "Section 1: Background"
Type: heading

[Image Block]
Coordinates: (x1, y1, x2, y2)
Type: figure

```

**Popular Libraries**:

**Doctr**:
* Automates document parsing
* Converts various formats to uniform structure
* Easy to use

**LayoutParser**:
* Unified toolkit for deep learning-based document image analysis
* Supports multiple document types:
  * PDFs
  * Magazine scans
  * Websites
  * Historical documents
* Detects text, image, and table regions
* Extracts content with OCR

**LayoutParser Example**:

```python
import layoutparser as lp

# Load detection model
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

# Detect layout
layout = model.detect(image)

# Extract text using OCR
ocr_agent = lp.TesseractAgent()
for block in layout:
    segment_image = (block.crop_image(image))
    text = ocr_agent.detect(segment_image)

```

**Key Takeaway**: Powerful libraries available; no need to implement from scratch

##### Step 2: Document Chunking

**Goal**: Break documents into smaller, manageable pieces (chunks)

**Why Chunking is Necessary**:
1. **Non-uniform document length**: Documents vary from short paragraphs to entire books
2. **Too broad**: Single document may cover multiple topics
3. **Context window limits**: Even after retrieval, entire book exceeds context limits

**Chunking Definition**: Breaking documents into smaller pieces, each called a "chunk"

**Advantages**:
* **Handle non-uniform lengths**: Create uniform chunk sizes
* **Improve precision**: Find specific relevant section within document
* **Overcome LLM limitations**: Each chunk fits within context window
* **Optimize computational resources**: Only include necessary content

**Chunking Algorithms**:

**1. Length-Based Chunking**:
* Split text at specified character/token length
* Simple and fast
* **Problem**: May split mid-sentence
* **Use case**: When sentence boundaries don't matter

**2. Regular Expression-Based Chunking**:
* Split on punctuation (periods, question marks, etc.)
* Respects sentence boundaries
* **Problem**: Lacks semantic understanding
  * May group unrelated sentences
  * May separate related content across chunks
* **Better than length-based** but not optimal

**3. Specialized Splitters** (recommended):
* Designed for specific formats (HTML, Markdown, code)
* Split at element boundaries:
  * Headers
  * List items
  * Code blocks
* **Works well** for structured documents
* **Example**: LangChain text splitters

**LangChain Text Splitters**:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Target chunk size
    chunk_overlap=200,    # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(document_text)

```

**Key Hyperparameters**:
* **chunk_size**: Typical 500-2000 characters (too small loses context; too large reduces precision)
* **chunk_overlap**: 10-20% of chunk_size (prevents cutting concepts mid-thought, ensures continuity)

##### Step 3: Indexing

**Goal**: Store chunks in data structure that enables efficient search

**Indexing Methods**:
1. Keyword/text-based indexing
2. Full-text indexing
3. **Knowledge graph-based indexing** (mentioned below, rarely used in practice)
4. **Vector-based indexing** (primary focus, most common)

#### Keyword/Text-Based Indexing

**Concept**: Match query terms with document content (partial or exact match)

**How it Works**:
* Build inverted index mapping terms to documents
* At query time: Find documents containing query terms
* Example tool: Elasticsearch

**Search Process**:

```

Query: "refund policy"
Index lookup: Find all chunks containing "refund" AND/OR "policy"
Return: Matching chunks

```

**Limitations**:
* **No semantic understanding**: Misses synonyms
* **Exact match focus**: "return policy" won't match "refund policy"
* **Context insensitive**: Can't capture meaning relationships

**Use Cases**:
* When exact keyword matching is sufficient
* Legacy systems
* Complement to vector search (hybrid approaches)

#### Knowledge Graph-Based Indexing

**Concept**: Represent documents/entities as graph nodes and relationships as edges.

**Why Rarely Used**: High complexity, difficult maintenance, limited library support. Not worth the effort for most RAG systems.

**When Useful**: Social networks, highly relational data (org charts), user embedding models leveraging network structure.

**Recommendation**: Use vector-based indexing unless you have specific relational needs.

> **Key Insight**: Vector-based indexing (embeddings + FAISS) is the modern RAG standard. Unlike keyword search (misses synonyms, context-insensitive), embeddings capture semantic meaning - "refund policy" and "return policy" are nearby vectors despite different words. This semantic understanding enables finding conceptually similar content regardless of exact phrasing.

#### Vector-Based Indexing (Modern Standard)

**Concept**: Encode text as vectors in high-dimensional space where semantic similarity = proximity

**Key Innovation**: Use pre-trained embedding models to map text to vectors

##### Embedding Models

**Definition**: ML model that maps input (text, images, video) to vector in high-dimensional embedding space

**Key Property**: **Semantically meaningful space**
* Similar meanings → nearby vectors
* Different meanings → distant vectors
* Distance measures semantic similarity

**Text Embeddings**:

![Diagram 12](https://kroki.io/mermaid/svg/eNodjcsKgzAURPf9ivmAmFr7AhGhlu7aTZFuJIu8sAX1yjWBfn5tZndgzkzPen7j_txgzaVr_TeUaAmD1zxhJPbQhmKoDG9rioyZyUUbFgFLU9A2IC4KWVajSTZuo_HOfaYeD3J-UGm5SY1r9_I2EJfIZbEXyHK5O4oVTmeRDqSUfzwU6gd8Rizq)

**Example Relationships**:
* "machine learning" and "ML" → nearby vectors (synonyms)
* "king" - "man" + "woman" = "queen" (semantic relationships)
* "dog" closer to "cat" than to "computer"

**Word2Vec** (foundational paper):
* First major demonstration of semantic embeddings
* Trained word-level embeddings
* Showed meaningful relationships:
  * king - man + woman ≈ queen
  * Synonyms cluster together
  * Analogies preserved in vector space

##### Building Index with Embeddings

**Process**:

![Diagram 13](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALH6JDUihIF54zSvOziWAVdXTsFJ4iQa25SakpKZl66gm9-SmpOLFi5E1iFczRYPUJJMUTWGSzrEu2Zl5JaoRCSmJSTGgsAKWIg8Q==)

**Index Table Structure**:

| Chunk ID | Text | Embedding |
|----------|------|-----------|
| 1 | "To learn more about our products, contact this number." | [0.23, -0.15, 0.67, ..., 0.42] |
| 2 | "Our refund policy allows returns within 30 days." | [0.15, 0.22, -0.31, ..., 0.55] |
| 3 | "Shipping typically takes 14 business days." | [-0.11, 0.33, 0.08, ..., -0.22] |

**Embedding Dimensions**: Typically 384-3072 dimensions
* OpenAI text-embedding-3-small: 1536 dimensions
* OpenAI text-embedding-3-large: 3072 dimensions
* Sentence-transformers: 384-768 dimensions

##### Popular Embedding Models

**Text Embedding Models**:

| Model | Provider | Output Dimension | Notes |
|-------|----------|-----------------|-------|
| text-embedding-3-large | OpenAI | 3072 | High performance, proprietary |
| text-embedding-3-small | OpenAI | 1536 | Faster, still effective |
| text-embedding-004 | Google (Gemini) | 768 | Google's offering |
| all-MiniLM-L6-v2 | Sentence-Transformers | 384 | Open source, fast |

**Code Example (OpenAI)**:

```python
import openai

text = "What is your refund policy?"
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

# embedding is a list of 1536 floating-point numbers

```

##### Embedding Layers vs Embedding Models

**Critical Distinction**: Don't confuse embedding layers (inside LLMs) with embedding models (standalone systems for RAG).

**Embedding Layer** (within LLMs):
* Simple lookup table: Matrix of size (vocabulary_size × hidden_dimension)
* Example: 50,000 vocab × 768 dims = lookup table with 50,000 rows
* Each token ID maps to one row: Token 2 → Row 2 → Vector [0.23, -0.15, ..., 0.42]
* Weights updated during LLM training
* Purpose: Convert token IDs to vectors for transformer processing

**Embedding Models** (standalone):
* Complex neural networks trained separately (often transformer-based)
* Specialized training: Contrastive learning, optimized for semantic similarity
* Output: Single vector representing entire input (full sentence/paragraph or image)
* Purpose: Retrieval, similarity search in RAG systems
* Examples: OpenAI text-embedding-3, CLIP, Qwen3 Embedding

**Key Relationship**:
* Every LLM contains embedding layers internally
* For RAG: Use standalone embedding models, not LLM's internal layers
* Larger models generally have larger embedding dimensions (CLIP: 768, Qwen3-8B: 4,096)

##### Handling Images in RAG

**Challenge**: Documents contain both text and images (diagrams, figures, tables)

**Two Approaches**:

**Option 1: Image Embedding Models**

Use models with shared embedding space for text and images

**CLIP (Contrastive Language-Image Pre-Training)** by OpenAI:
* Has both text encoder and image encoder
* Trained together on image-caption pairs
* Shared embedding space: Similar images and their descriptions have nearby embeddings

**Architecture**:

![Diagram 14](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaM_cxPTUWAVdXTsFJwhHwTUvOT8ltSgWrMI5OiS1ogSiwAXMRpV3Asu4RgdnJBalpii45ialpqRk5qUrBBckJqdC1LhA1AAAgugi1w==)

**Key Requirement**: Use same model for both text and images
* Text chunks: Process with CLIP text encoder
* Image chunks: Process with CLIP image encoder
* Results in unified index where text and images can be compared

**Implementation**:

```python
import clip

model, preprocess = clip.load("ViT-B/32")

# For text chunks
text_embedding = model.encode_text(clip.tokenize(["Customer support info"]))

# For image chunks
image = preprocess(Image.open("diagram.png"))
image_embedding = model.encode_image(image)

# Both embeddings in same space, can compute similarity

```

**Option 2: Image Captioning**

Convert images to text descriptions, then use text embeddings only. Captioning model generates text description of image, which is then embedded like any other text chunk.

**Advantages**: Can use any text embedding model; simpler infrastructure (one model)
**Disadvantage**: Caption may miss visual details; adds captioning latency

**How Retrieved Images Are Used** (Important):

In most RAG systems, retrieved images are **NOT** passed to the LLM:

**Standard Practice**:
* Images displayed in UI as visual citations alongside generated text
* Image captions (if using Option 2) passed to LLM as text
* LLM generates from text; images shown separately

**Why**: Most LLMs are text-only; image processing is expensive; images serve as supplementary visuals.

**Exception**: Multimodal LLMs (GPT-4V, Gemini Vision) can accept images but uncommon in production due to cost and complexity.

**Flow**: Retrieval finds text + images → Text to LLM → Generated text + images displayed in UI

**Final Index**: Separate indices for text and images (Option 1), or unified index with text chunks and image captions (Option 2).

#### Searching from Index at Runtime

**Goal**: Given user query, efficiently find most relevant chunks from index

**Process**:

![Diagram 15](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALHaM_cxPTUWAVdXTsFp2jnxIKSzPy8zLx0Bd_8lNScWLAiJ7Csc3RIakWJgktqcXJRJlgZRNYZLOsCkXXNTUpNSUHT7wJW4RoNl4wFABhYJyY=)

##### Step 1: Encode Query

**Use same text encoder** that built the index

```python
query = "What is your refund policy?"
query_embedding = encoder.encode(query)

# Results in vector in same embedding space as indexed chunks

```

**Critical**: Must use identical model for consistency

##### Step 2: Nearest Neighbor Search

**Problem**: Find top K chunks with embeddings closest to query embedding

**Definition**: Given query point q and N data points, find K points with minimum distance to q

**Distance Metrics**:
* **Euclidean distance**: √(Σ(pi - qi)²)
* **Cosine similarity**: (p·q) / (||p|| ||q||) (most common for embeddings)
* **Dot product**: p·q

**Cosine Similarity** (preferred for embeddings):
* Measures angle between vectors
* Range: -1 (opposite) to 1 (identical)
* Ignores magnitude, focuses on direction
* Works well for normalized embeddings

**Visual Example** (2D for illustration):

```

         Query (q)
            *
           /|\
          / | \
        /   |   \
      /     |     \
    *       *       *
   p1      p2      p3

Closest points to q: p2, p1
(Smallest cosine distance)

```

##### Exact Nearest Neighbor Search

**Algorithm**: Compare query embedding with every indexed embedding

```python
def exact_nearest_neighbor(query_embedding, index_embeddings, k=5):
    distances = []
    for idx, emb in enumerate(index_embeddings):
        distance = cosine_distance(query_embedding, emb)
        distances.append((distance, idx))

    distances.sort()  # Sort by distance
    return distances[:k]  # Return top K

```

**Characteristics**:
* **Guaranteed correct**: Returns exact top-K closest points
* **Simple**: Easy to implement and understand
* **Slow**: O(N) complexity where N = number of chunks

**Problem**: Not scalable
* With millions/billions of chunks, comparing all is too expensive
* Each query requires checking every embedding
* Real-time search requirements make this impractical

##### Approximate Nearest Neighbor (ANN) Search

**Goal**: Find approximately nearest neighbors much faster than exact search

**Trade-off**: Sacrifice small amount of accuracy for massive speed improvement

**Acceptable Approximation**:
* May not return absolute closest K points
* Returns K points that are very close to query
* For RAG: Close-enough documents work fine
* Speed improvement: 10x-1000x faster

**Three Main ANN Categories**:
1. Clustering-based
2. Tree-based
3. Locality-sensitive hashing

##### 1. Clustering-Based ANN

**Concept**: Pre-cluster data points, search only within relevant cluster(s)

**Preprocessing**:

![Diagram 16](https://kroki.io/mermaid/svg/eNoljEEKgzAURPc9xb9A6AWKoEkKpd1o7SpkEc0nkWoi31js7VuSWQ3zHuPIrB4e3Qn-qdVrQ4J2R_pqYKyCRvV4JJBhjBZJZ6vJhKusgVwGtHYKrkCeoVBPNDR6qGcXaUp-KVSqW7B4lG-RJ5H7VfVxZffLQOeqwxk_JiTgfg_vTf8ALkwvxg==)

**Cluster Representation**:
* Each cluster has centroid (representative point)
* All points assigned to nearest centroid

**Search Process**:

![Diagram 17](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALHaMecHAWXxJJEhYD8zLyS4lgFXV07Bado55zS4pLUosy8dAXHnPT8osySjFybpCJ9O29d39TEvOJYsHYnsGpnmGoFQ2RhF7iwEbKwK1zYGFnYDS5sgizsDhc2jQUA5PA0bg==)

**Example**:
* Query arrives
* Compare with 5 centroids (fast)
* Identify closest centroid (e.g., Cluster 3)
* Search only within Cluster 3 points
* Return top-K from cluster

**Trade-off**:
* **Speed**: Only search 20% of data (1 of 5 clusters)
* **Accuracy**: May miss points in nearby clusters
* Example: Top-3 might be in Cluster 3, but 4th-best in Cluster 2

**In Practice**: Often search top-N clusters (not just 1) for better accuracy

##### 2. Tree-Based ANN

**Concept**: Partition space hierarchically into tree structure

**Tree Building**:

![Diagram 18](https://kroki.io/mermaid/svg/eNotzLEOwiAUheHdp7gvQBxcTRMLuphYtU0cCENtr0KC0Nxehr69DeXM3_m_1E8WOrWDdSf9SEgL3KMLbECICmp9cWGEG_aEMx_ftK8kBqboRpM_dWZSS59mRoJDNi16HBiLkdko3a6VwcLLsXVha5VXE_yyWZXtWT-REwXo4iSumX4o_qB48wcUDjjO)

**How it Works**:
* Recursively split space (e.g., along dimension with highest variance)
* Each leaf node represents region containing subset of points
* Tree encodes spatial partitioning

**Search Process**:
* Traverse tree from root
* At each node, decide which branch query belongs to
* Reach leaf node (region)
* Search only points in that region

**Similarity to Clustering**: Both partition space into regions

**Difference**: Tree provides hierarchical structure with efficient traversal

##### 3. Locality-Sensitive Hashing (LSH)

**Concept**: Hash functions where similar inputs produce similar outputs

**Standard Hash Functions** (e.g., SHA):
* Small input change → completely different output
* Purpose: Distribution, not similarity preservation

**Locality-Sensitive Hash**:
* Similar inputs → similar hash values → same/nearby buckets
* Preprocessing: Hash all points into buckets
* Query: Hash query, search only within matching bucket
* **Advantage**: Constant-time bucket lookup, then search small bucket

##### ANN Algorithm Comparison

| Method | Preprocessing | Search Time | Accuracy | Use Case |
|--------|--------------|-------------|----------|----------|
| Exact | None | O(N) | 100% | Small datasets (<10K points) |
| Clustering | Build clusters | O(N/C) | ~95% | Medium datasets, balanced speed/accuracy |
| Tree | Build tree | O(log N) avg | ~90-95% | Structured data, hierarchical relationships |
| LSH | Compute hashes | O(1) + O(B) | ~85-95% | Very large datasets, extreme speed priority |

**Practical Recommendation**: Clustering-based or tree-based for most RAG applications

##### FAISS (Facebook AI Similarity Search)

**What it is**: Library for efficient similarity search and clustering of dense vectors

**Features**:
* Implements multiple ANN algorithms
* Optimized for CPU and GPU
* Supports billions of vectors
* Open source

**Supported Distance Metrics**:
* Euclidean distance (L2)
* Cosine similarity
* Inner product

**Basic Usage**:

```python
import faiss
import numpy as np

# Build index
dimension = 128  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add vectors to index
vectors = np.random.random((10000, dimension)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
k = 5  # Number of nearest neighbors
distances, indices = index.search(query, k)

print(f"Top {k} nearest neighbors: {indices}")
print(f"Distances: {distances}")

```

**Advanced Indexes**:
* IndexIVFFlat: Inverted file index (clustering-based)
* IndexIVFPQ: Product quantization for compression
* IndexHNSW: Hierarchical navigable small world graphs

**Key Takeaway**: Production-ready library; no need to implement ANN from scratch

##### Production Library Best Practices

**Standard Libraries for Production**:
* **FAISS** (Meta): Open-source, highly optimized, GPU support, industry standard
* **ScaNN** (Google): Google's efficient similarity search library

**LangChain vs Direct Library Usage**:

* **For prototyping**: LangChain wrappers are convenient and fast to implement
* **For production**: Use base libraries directly (FAISS, ScaNN)
  * Avoids wrapper complexity and version mismatches
  * More control, easier to optimize and debug
  * Direct access to all library features

**Observed Pattern**: Start with LangChain for prototyping, switch to direct library usage when productionizing.

### Generation Component

#### Basic Generation

**Core**: Generation component is simply a general-purpose LLM

**Process**:

![Diagram 19](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALH6KD8_BIrBbfSnByF4ILE5NRYBV1dOwWnaJ_UtBKF4NKkYrAgRDFYyjk6KDM9A13OCSznEh2Ump6Zn6dgiCzqChM1gog6g0XdYKLGyKLuMFGTWADTQS3y)

**Prompt Structure**:

```

Retrieved Context:
* "Our refund policy allows returns within 30 days..."
* "Items must be unused and in original packaging..."
* "Contact customer support at 555-0123..."

User Query: What is your refund policy?

Generate response:

```

**LLM Response**:

```

Our refund policy allows returns within 30 days of purchase. Items must be unused and in original packaging. For assistance, contact customer support at 555-0123.

```

#### Enhanced Generation with Prompt Engineering

**Goal**: Improve response quality using techniques from prompt engineering section

**Enhanced Architecture**:

![Diagram 20](https://kroki.io/mermaid/svg/eNpLL0osyFDwCeJSAALH6KDUkqLM1LLUFAXn_LyS1LySWAVdXTsFp2jn_NykzLzUWLA65-jQ4tQihcDS1KJKqAKwuBOY7RIdUJSfW1ACUesCFnON9vHxhQi4ggXcgFYVF-TnFafGAgDOxSSa)

**Prompt Engineering Components**:
* Role-specific prompting
* Output format instructions
* Chain of thought guidance
* Few-shot examples
* Citation requirements

**Example Enhanced Prompt**:

```

You are a helpful customer support assistant for TechStore.

Your task is to deliver concise and accurate responses based on the provided context.
Always cite the source of your information.

If there is a reasoning process to generate the response, think step by step and put your steps in bullet points.

Context from documents:
1. Our refund policy allows returns within 30 days of purchase. [Source: Refund Policy Document]
2. Items must be unused and in original packaging. [Source: Refund Policy Document]
3. Refunds processed within 5-7 business days. [Source: Refund Processing Guide]

User Question: What is your refund policy?

Response:

```

**LLM Enhanced Response**:

```

Our refund policy:
* Returns accepted within 30 days of purchase
* Items must be unused and in original packaging
* Refunds processed within 5-7 business days

[Source: Refund Policy Document, Refund Processing Guide]

```

#### RAFT: Retrieval-Augmented Fine-Tuning

**Problem**: Retrieved documents may include irrelevant content
* Retrieval quality not always perfect
* Embedding models may return some irrelevant chunks
* General-purpose LLM treats all retrieved content equally

**Solution**: Fine-tune LLM to distinguish relevant from irrelevant retrieved documents

**RAFT Concept**: Train LLM to:
* Focus on relevant documents in retrieved content
* Ignore or minimize influence of irrelevant documents
* Generate responses primarily from helpful sources

**Training Process**:

![Diagram 21](https://kroki.io/mermaid/svg/eNotjTsKwzAMhveeQhfIFQJ5OJMDbaCT8BBa4XiIbGS30NvHyNX08_0PednTAXa7Qb0BNyoS6EtvmCIX4uKg63oY8S7xTAUM-8BEEtg7rUz4zCTw-JD8_lnlo-oZDR87v-peG2ilWU2D1q4NGAVLfZ9T5EzuAp64LCA=)

**Training Data Format**:
* Prompt: Question
* Context: Mix of relevant and irrelevant documents (labeled)
* Target: Response based only on relevant documents

**Example Training Sample**:

```

Question: What is the refund policy?

Context:
[RELEVANT] Our refund policy allows returns within 30 days...
[IRRELEVANT] Our company was founded in 2010...
[RELEVANT] Refunds are processed within 5-7 business days...
[IRRELEVANT] We offer a wide range of products...

Target Response: Our refund policy allows returns within 30 days. Refunds are processed within 5-7 business days. [Based on relevant documents only]

```

**Outcome**: LLM learns pattern of focusing on relevant markers/content

**Paper**: "RAFT: Adapting Language Model to Domain Specific RAG" (2024)

**Benefits**:
* Robust to imperfect retrieval
* Better handling of noisy context
* Improved response accuracy

**Trade-off**: Requires fine-tuning infrastructure and labeled training data

### RAG Evaluation

**Challenge**: Multiple components means multiple failure points

**Components to Evaluate**:
* Retrieval quality
* Generation accuracy
* Faithfulness to context
* Overall system performance

#### Evaluation Aspects

![Diagram 22](https://kroki.io/mermaid/svg/eNo9jkEKwkAMRfeeIhcovYAUbEdFaF1Id0MXqY11oGZKJhW8vdOp-FeBz3v5o-D8hNbsIOZgKyFUglbQseMRDCp2kGUFlLbGniYw_r68iDXse8kLDHCjid7Iml9EfmeXbGXiKntyTFm7MEFdN1tVpcrYxg9RWRMKB1CflJch2t3j8xevL8PGmcQdbUmqJHAmJkF1nhNpKMwurr96F1ZcxUXB1H0BJQRJZQ==)

**Three Key Evaluation Dimensions**:

##### 1. Context Relevance

**What it Measures**: Quality of retrieval component

**Question**: Are retrieved documents relevant to the query?

**Evaluation Approach**: Ranking metrics

**Common Metrics**:

**Hit Rate (Recall@K)**:
* Definition: Fraction of queries where at least one relevant document appears in top-K results
* Formula: (Number of queries with relevant doc in top-K) / (Total queries)
* Example: If 80 out of 100 queries have relevant doc in top-5, Hit Rate@5 = 0.80

**Mean Reciprocal Rank (MRR)**:
* Definition: Average of reciprocal ranks of first relevant document
* Formula: (1/N) Σ (1 / rank of first relevant doc)
* Example: First relevant at rank 1 = score 1.0, at rank 2 = 0.5, at rank 3 = 0.33

**NDCG (Normalized Discounted Cumulative Gain)**:
* Considers relevance scores and ranking positions
* Rewards placing highly relevant docs higher in results
* Normalized to 0-1 range

**Precision@K**:
* Definition: Fraction of top-K results that are relevant
* Formula: (Number of relevant docs in top-K) / K
* Example: 3 relevant out of top-5 results = Precision@5 = 0.60

**Evaluation Process**:
1. Create test set of queries with known relevant documents
2. Run retrieval
3. Compare retrieved documents with ground truth
4. Calculate metrics

##### 2. Faithfulness

**What it Measures**: Whether generated response aligns with provided context

**Question**: Is the response factually grounded in retrieved documents (vs hallucinated)?

**Challenge**: Subjective evaluation

**Evaluation Approaches**:

**Human Evaluation**:
* Domain experts review responses
* Check if claims are supported by context
* Label as faithful/unfaithful

**Automated Fact-Checking**:
* Use NLI (Natural Language Inference) models
* Check if response is entailed by context
* Consistency checking between response and sources

**Entailment Checking Example**:

```

Context: "Returns accepted within 30 days."
Response: "You can return items within 30 days."
Evaluation: Entailed → Faithful

Context: "Returns accepted within 30 days."
Response: "You can return items within 60 days."
Evaluation: Contradicted → Not Faithful

```

**Example Metrics**:
* Faithfulness score: Percentage of response claims supported by context
* Hallucination rate: Percentage of response containing unsupported claims

##### 3. Answer Correctness/Relevance

**What it Measures**: Whether response correctly answers the query

**Question**: Is the generated response accurate and relevant to the user's question?

**Evaluation Approaches**:

**Ground Truth Comparison**:
* Human-written correct answers
* Compare generated response with reference

**Metrics** (from text generation evaluation):

**BLEU (Bilingual Evaluation Understudy)**:
* Measures n-gram overlap between generated and reference text
* Range: 0-1 (higher better)
* Originally for machine translation

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
* Measures recall of n-grams
* ROUGE-1: unigram overlap
* ROUGE-2: bigram overlap
* ROUGE-L: longest common subsequence

**BERTScore**:
* Uses BERT embeddings to compute semantic similarity
* Better captures meaning than exact token matching
* More robust to paraphrasing

**Example**:

```

Query: "What is your refund policy?"

Reference: "We accept returns within 30 days. Items must be unused."

Generated: "Our policy allows returns for up to one month. Products should be in original condition."

BERTScore: 0.92 (high semantic similarity despite different wording)
BLEU: 0.15 (low due to different word choices)

```

**Semantic Similarity** (preferred for RAG):
* Embed both reference and generated response
* Compute cosine similarity of embeddings
* Captures meaning better than token overlap

#### Evaluation Best Practices

**Multi-Faceted Approach**: Use multiple metrics covering different aspects
* Context relevance: Validate retrieval
* Faithfulness: Ensure no hallucination
* Answer correctness: Verify usefulness

**Test Set Creation**:
* Diverse query types
* Known relevant documents
* Expert-written reference answers
* Cover edge cases (ambiguous queries, multi-hop reasoning)

**Continuous Monitoring**: Track metrics over time as system evolves

### Complete RAG System Design

#### System Architecture

![Diagram 23](https://kroki.io/mermaid/svg/eNplkM0KgzAMgO8-RV_AVxione7nNNkteCgum4USJa1ue_uVZjJkOYXvyx95sJkGddWZilHAZUZ-dyrPd6qEFgNbXIzrki0TrlaMN1WNFPAVRFdJa2iQkE2wIwkvhKdcp3wPtSXjVIt-GsljlyVZw3deFC6upR7lkkbOUovfbjxAbWwY7rMj9F5qj7BO_Ss_QUH-iRwpM_bh13TeNMkPPocLVqU=)

#### Component Details

##### 1. Input Guardrails

**Purpose**: Filter unsafe or inappropriate user queries before processing

**What it Checks**:
* Violence or harmful content requests
* Illegal activities
* Jailbreak attempts
* Inappropriate content
* Policy violations

**Implementation**: ML classifier trained on unsafe query examples

**Decision**:
* **Safe query**: Proceed to next component
* **Unsafe query**: Return rejection message, skip LLM processing

**Benefits**:
* Protects system from misuse
* Saves compute on inappropriate requests
* Ensures compliance

##### 2. Query Rewrite/Query Expansion

**Purpose**: Enhance user query for better retrieval

**Common Issues in Raw Queries**:
* Typos and misspellings
* Missing punctuation
* Grammatical errors
* Ambiguous phrasing
* Incomplete information

**Query Rewrite**: Correct and clarify query

```

Original: "hw long shiping take"
Rewritten: "How long does shipping take?"

```

**Query Expansion**: Add related terms

```

Original: "refund policy"
Expanded: "refund policy return exchange money back guarantee"

```

**Implementation**:
* Rule-based corrections (spell check)
* ML-based query understanding
* Synonym expansion
* LLM-based rewriting

**Benefits**:
* Better retrieval accuracy
* Handles user input variations
* Improves search effectiveness

##### 3. Retrieval (Covered Earlier)

**Steps**:
1. Encode query with text encoder
2. Run ANN search against indices
3. Return top-K relevant chunks (text and/or images)

##### 4. Prompt Engineering (Covered Earlier)

**Combines**:
* Retrieved chunks
* Enhanced user query
* System instructions
* Output format specifications
* Few-shot examples (if applicable)

##### 5. LLM Generation

**Options**:
* General-purpose LLM (GPT-4, Claude, Llama 3, etc.)
* RAFT fine-tuned LLM (for better handling of irrelevant context)

##### 6. Output Guardrails

**Purpose**: Ensure generated response is safe to show user

**What it Checks**:
* Harmful or dangerous content in response
* Biased or discriminatory language
* Leaked sensitive information (PII)
* Factual inaccuracies (if detectable)
* Policy violations

**Implementation**: ML classifier on generated text

**Decision**:
* **Safe response**: Display to user
* **Unsafe response**: Block and show rejection message

**Why Needed**: Even with safe input, LLM may occasionally generate unsafe output

**Trade-off**: May occasionally block legitimate responses (false positives)

#### Key System Properties

**Modularity**: Each component can be improved independently
* Better embedding model → better retrieval
* Better LLM → better generation
* Better guardrails → better safety

**Scalability**:
* Index built offline (can handle large databases)
* Only relevant chunks retrieved (efficient at runtime)
* ANN search scales to billions of documents

**Maintainability**:
* Easy to add new documents (re-index)
* Easy to remove outdated documents
* No model retraining required for content updates

**Live Data Integration** (covered in next week):
* RAG can incorporate real-time information
* Search web at query time
* Add live results to retrieved context
* LLM always has access to current information

#### Practical Considerations

**Index Updates**:
* Periodic re-indexing of document database
* Incremental updates for new documents
* Delete outdated content

**Hyperparameter Tuning**:
* Top-K retrieved chunks (typical: 3-10)
* Chunk size and overlap
* ANN algorithm parameters
* LLM temperature and sampling

**Cost Optimization**:
* Balance retrieval quality vs speed
* Cache frequent queries
* Use smaller embedding models for cost-sensitive applications

**Monitoring**:
* Track retrieval quality metrics
* Monitor generation faithfulness
* Log query patterns
* Identify failure modes

## Key Insights

### Fundamental Principles

* **Three complementary adaptation techniques**: Fine-tuning modifies model weights, prompt engineering guides behavior without training, and RAG provides external knowledge. Each has distinct use cases and can be combined for optimal results.

* **Parameter-efficient fine-tuning enables practical adaptation**: Methods like LoRa and adapters reduce trainable parameters by 99%+, making fine-tuning accessible without massive computational resources. Only 0.1% trainable parameters can achieve comparable results to full fine-tuning.

* **Prompt engineering is surprisingly powerful**: Simple additions like "Let's think step by step" can dramatically improve reasoning performance (18% → 57% on complex tasks) without any model training. Chain of thought prompting became the foundation for modern reasoning models.

* **RAG solves the context window limitation**: Cannot include entire document databases in prompts due to token limits. Retrieval narrows millions of documents to handful of relevant chunks, making domain adaptation practical for large knowledge bases.

* **Semantic embeddings are core to modern RAG**: Vector representations capture meaning rather than keywords. Similar meanings have nearby embeddings regardless of exact word choice, enabling sophisticated semantic search.

### Important Distinctions and Comparisons

* **Fine-tuning vs prompt engineering vs RAG**:
  * **Fine-tuning**: Permanent model changes, requires training infrastructure, best for consistent behavioral changes
  * **Prompt engineering**: No training needed, flexible and immediate, limited by context window, best for guiding existing capabilities
  * **RAG**: Separates knowledge from model, easy to update, handles large knowledge bases, best for dynamic knowledge that changes frequently
  * **Combined approach often optimal**: RAFT (fine-tuning + RAG), prompt engineering enhances both fine-tuning and RAG

* **Full parameter fine-tuning vs parameter-efficient fine-tuning**:
  * **Full parameter**: Updates all billions of parameters, computationally prohibitive for most organizations, theoretically optimal
  * **PEFT (Adapters, LoRa)**: Updates <1% of parameters, 100x faster and cheaper, achieves 95%+ of full fine-tuning performance
  * **Practical reality**: PEFT is production standard; full fine-tuning reserved for well-funded research labs

* **Few-shot vs zero-shot prompting**:
  * **Few-shot**: Show 2-5 examples of desired behavior, more reliable, uses more tokens, best for specific formats or complex tasks
  * **Zero-shot**: Provide instructions without examples, more flexible, uses fewer tokens, best for straightforward tasks or when examples are difficult to create
  * **When to use each**: Few-shot for format control and consistency, zero-shot for one-off queries and simple tasks

* **Chain of thought (COT) few-shot vs zero-shot**:
  * **Few-shot COT**: Show examples with reasoning steps, more reliable for complex reasoning, requires careful example crafting
  * **Zero-shot COT**: Just add "Let's think step by step", surprisingly effective, no examples needed, works across diverse tasks
  * **Both dramatically improve reasoning**: Choose based on task complexity and whether good examples are available

* **Rule-based vs AI-based document parsing**:
  * **Rule-based**: Fixed heuristics, limited flexibility, fails on varied formats, legacy approach
  * **AI-based** (LayoutParser, Doctr): Uses ML models, handles diverse layouts, extracts text from images via OCR, industry standard
  * **Modern practice**: Always use AI-based parsing for production RAG systems

* **Length-based vs regex-based vs specialized chunking**:
  * **Length-based**: Simple, fast, but may split mid-sentence, loses semantic coherence
  * **Regex-based** (punctuation): Respects sentence boundaries, but misses topic transitions, may group unrelated content
  * **Specialized splitters** (LangChain, HTML/Markdown aware): Understands document structure, splits at semantic boundaries, supports overlap for continuity
  * **Recommendation**: Use specialized splitters for production; they handle edge cases better

* **Keyword-based vs vector-based indexing**:
  * **Keyword-based** (Elasticsearch): Exact/partial term matching, fast for specific queries, misses synonyms and semantic relations
  * **Vector-based** (embeddings + FAISS): Captures semantic meaning, finds conceptually similar content, handles paraphrasing, works across languages
  * **Modern RAG standard**: Vector-based is primary method; keyword-based used for hybrid search or exact match requirements

* **Exact nearest neighbor vs approximate nearest neighbor (ANN)**:
  * **Exact**: Guarantees finding true top-K, O(N) complexity, too slow for production (millions of chunks)
  * **ANN** (clustering, tree-based, LSH): Finds approximate top-K, 10-1000x faster, 90-95%+ accuracy sufficient for RAG
  * **Trade-off is favorable**: Slight accuracy loss for massive speed gain; users don't notice difference in relevance

* **Image embedding vs image captioning for multimodal RAG**:
  * **Image embedding** (CLIP): Direct embedding of images, shared space with text, preserves visual details, requires multimodal model
  * **Image captioning**: Convert to text first, use any text embedding model, may lose visual nuances, adds captioning step
  * **Choice depends**: Use image embeddings if visual details matter; use captioning for simpler infrastructure

* **General-purpose LLM vs RAFT fine-tuned LLM for generation**:
  * **General-purpose**: Works out-of-box, treats all retrieved docs equally, may be confused by irrelevant docs
  * **RAFT fine-tuned**: Learned to focus on relevant docs, ignores distractors, more robust to imperfect retrieval
  * **When to use RAFT**: When retrieval quality is inconsistent or noisy; requires fine-tuning infrastructure

### Practical Considerations

* **Focus on concepts, not specific libraries**: Field evolves rapidly; library expertise becomes obsolete (TensorFlow vs PyTorch). Learn foundational concepts deeply, stay pragmatically flexible with tools.

* **Prototype vs production tools differ**: Example - Ollama (easy experiments) vs VLLM (production scale). Understand concepts to make switching simple.

* **Lightweight LLMs for simple tasks**: Use smaller models (<1-3B params) for subtasks like classification/routing. Examples: GPT-2, Qwen-2.5-1.5B, Phi-3-mini. Saves costs without compromising core functionality.

* **Companies build custom RAG but leverage standard components**: Use existing LLMs, FAISS/ScaNN for search, high-quality embedding models. Custom integration logic and company-specific prompting.

* **Production team structure**: Often separate teams for evaluation, security, optimization, and deployment. Large companies (Google, Meta) build everything internally; startups rely on open-source.

* **Chunk size/overlap critical**: 500-2000 chars with 10-20% overlap. Tune empirically.

* **Same embedding model for index and search**: Model mismatch breaks semantic similarity.

* **Top-K trades precision/recall**: Start K=5, tune based on evaluation.

* **Query rewrite improves retrieval**: Correct typos, expand synonyms. Lightweight, high impact.

* **Input/output guardrails essential**: Protect against malicious queries and harmful generation.

* **RAG enables easy updates**: Re-index for new docs; no retraining needed.

* **Embedding dimensions trade quality/speed**: Larger (1536-3072) more nuanced but slower; smaller (384-768) faster.

* **ANN algorithm choice matters**: IVF (balanced), HNSW (high accuracy), LSH (extreme scale).

* **Hybrid search improves robustness**: Combine vector (semantic) + keyword (exact terminology).

* **Multi-dimensional evaluation required**: Context relevance, faithfulness, answer correctness.

### Common Pitfalls and Misconceptions

* **Prompt engineering ≠ RAG replacement at scale**: Works for small knowledge bases; RAG required for thousands of documents. Context windows and compute costs prohibit including entire databases in prompts.

* **More retrieved chunks ≠ better results**: Beyond K=5-10, additional chunks add noise. Find optimal K through evaluation.

* **Document chunking is mandatory**: Whole documents (books) overwhelm context windows and reduce retrieval precision.

* **Embedding quality > ANN algorithm sophistication**: Invest in high-quality embeddings first (OpenAI text-embedding-3-large). Algorithm choice matters less than embedding model quality.

* **RAFT not always necessary**: General-purpose LLMs work well with good retrieval. Only pursue RAFT if evaluation shows retrieval noise hurts generation.

* **Semantic search can miss exact terminology**: Hybrid keyword+vector search catches both semantic matches and specific jargon/proper nouns.

* **Context window limits are hard constraints**: Monitor total tokens (query + chunks + system prompt) to avoid failures.

* **Always use ANN at scale**: 90% ANN accuracy is sufficient; speed gain is 100x+ vs exact search.

## Quick Recall / Implementation Checklist

* [ ] **Choose adaptation technique based on use case**: Fine-tuning for consistent behavior changes, prompt engineering for quick iterations, RAG for large/dynamic knowledge bases. Combination often optimal.

* [ ] **Start with PEFT (LoRa) if fine-tuning**: Full parameter updates are impractical for most organizations. Hugging Face PEFT library provides easy implementation.

* [ ] **Use zero-shot COT for reasoning tasks**: Simply add "Let's think step by step" to prompts before committing to complex few-shot examples.

* [ ] **Design system prompt carefully**: Include role specification, output format, few-shot examples, and COT guidance as appropriate. System prompt applies to all queries, so invest time optimizing.

* [ ] **For RAG, always use AI-based document parsing**: Libraries like LayoutParser and Doctr handle diverse formats. Rule-based parsing fails on real-world document variety.

* [ ] **Chunk documents with specialized splitters**: LangChain text splitters or similar libraries. Start with chunk_size=1000 and chunk_overlap=200, then tune based on evaluation.

* [ ] **Use high-quality embedding models**: OpenAI text-embedding-3-large or similar state-of-art models. Embedding quality dominates retrieval performance.

* [ ] **Implement vector-based indexing with FAISS**: Standard for production RAG. Start with IndexIVFFlat (clustering-based ANN) for balanced speed/accuracy.

* [ ] **Keep same embedding model for indexing and search**: Model mismatch breaks semantic similarity. Lock model version and use consistently.

* [ ] **Tune top-K retrieved chunks empirically**: Start with K=5. Increase if missing relevant info, decrease if context too noisy. Evaluate with real queries.

* [ ] **Add query rewrite/expansion before retrieval**: Correct typos, expand with synonyms. Simple preprocessing step with significant impact on retrieval quality.

* [ ] **For multimodal RAG, use CLIP or captions**: CLIP preserves visual details; captioning simplifies infrastructure.

* [ ] **Consider RAFT if retrieval noisy**: Only if evaluation shows LLM confused by irrelevant docs.

* [ ] **Implement input and output guardrails**: Safety classifiers on both user queries and LLM responses. Essential for production deployment.

* [ ] **Evaluate across multiple dimensions**: Context relevance (retrieval), faithfulness (no hallucination), answer correctness (end-to-end). No single metric sufficient.

* [ ] **Use hybrid keyword+vector search for robustness**: Combine semantic matching (vector) with exact term matching (keyword) to catch all relevant docs.

* [ ] **Monitor and log query patterns**: Identify frequent queries, failure modes, edge cases for iterative improvement.

* [ ] **Plan for index updates**: Design workflow for periodic/incremental re-indexing without downtime.

* [ ] **Test end-to-end before production**: Build prototype, evaluate on representative queries, iterate.

* [ ] **Leverage existing libraries**: FAISS, LangChain, Hugging Face. Don't reinvent well-solved components.

* [ ] **Start simple, then optimize**: Basic RAG works well. Add complexity (RAFT, hybrid search) only if evaluation shows need.

* [ ] **Document hyperparameters**: Track chunk_size, top-K, embedding model, ANN algorithm for reproducibility.

---

## References

### Papers

* [**Chain-of-Thought Prompting:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) - Google Brain (January 2023)

* [**Zero-shot CoT:** "Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916) - Google DeepMind (May 2022)

* [**RAFT:** "Adapting Language Model to Domain Specific RAG"](https://arxiv.org/abs/2403.10131) - UC Berkeley (2024)

* [**LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) - Microsoft (2021)

* [**Parameter-Efficient Transfer Learning:** "Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751) - Google (2019)

* [**Word2Vec:** "Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781) - Google (2013)

### Tools & Libraries

* [**Hugging Face PEFT**](https://huggingface.co/docs/peft/) - Parameter-Efficient Fine-Tuning library

* [**FAISS**](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search

* [**LangChain**](https://python.langchain.com/) - Framework for LLM applications

* [**LangChain Text Splitters**](https://python.langchain.com/docs/how_to/split_text/) - Document chunking utilities

### Vector Databases

* [**Pinecone**](https://www.pinecone.io/) - Managed vector database

* [**Weaviate**](https://weaviate.io/) - Open-source vector database

* [**Chroma**](https://www.trychroma.com/) - AI-native embedding database

* [**Qdrant**](https://qdrant.tech/) - Vector similarity search engine

* [**Milvus**](https://milvus.io/) - Open-source vector database

* [**pgvector**](https://github.com/pgvector/pgvector) - PostgreSQL extension for vector similarity search

### Embedding Models

* [**Sentence Transformers**](https://www.sbert.net/) - Pre-trained sentence embedding models

* [**OpenAI Embeddings**](https://platform.openai.com/docs/guides/embeddings) - text-embedding-3-small, text-embedding-3-large

* [**Cohere Embeddings**](https://cohere.com/embeddings) - Multilingual embedding models
