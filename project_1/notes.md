# LLM & PyTorch Study Guide – Concepts, Analogies, and Code Anchors

## 1. Python Foundations – the hidden scaffolding

* **set() as vocabulary extractor**: instantly removes duplicates; mental model for building a vocab list.
* **char ↔ ID bidict**: the **one-to-one map** that lets us jump between human-readable and model-readable.
* **zip()**: pairs elements **position-wise** – identical to how attention pairs Query-Key positions.
* **Tensors vs lists**: tensors live on GPU, support autograd; lists are CPU-only, no gradients.
* **Shape intuition**: `(batch, seq, dim)` – every tensor tells a **story of scale**.

## 2. Tokenizers – the Lego factory

* **Sub-word tokens**: balance vocabulary size vs. sequence length (GPT-2 ~50 k tokens).
* **AutoTokenizer** is a **factory pattern**: give it ANY checkpoint → returns the correct fast/slow class.

______

Python

```
tok = AutoTokenizer.from_pretrained("gpt2")   # works for BERT, T5, Qwen, …
```

* **Fast vs Slow**: fast = Rust backend, ~10× speed-up; slow = pure Python, fallback.
* **Special tokens**: pad, eos, unk – added automatically by tokenizer if `add_special_tokens=True`.
* **tiktoken**: OpenAI’s standalone BPE; no HF config needed; ultra-fast for huge corpora.
* **“Ġ” prefix**: indicates a sub-word that **starts after a space** (GPT-2 BPE).
* **Encoding pipeline**:\
  raw text → tokenizer → input\_ids (ints) → model → logits → softmax → next-token probs

## 3. PyTorch nn.Module – clay for universal approximation

* **nn.Parameter**: a **leaf** in the autograd graph; anything not wrapped won’t get gradients.
* **Custom Linear layer anatomy**:

______

Python

```
y = x @ W.T + b        # (B, in) @ (out, in).T → (B, out)
```

* **Parameter count**: `in_f × out_f + out_f` – memorize this for quick sanity checks.
* **Layer compatibility**: output channels of layer _i_ must equal input channels of layer _i+1_.
* **Universal Approximation Theorem**: one hidden layer + non-linearity → can approximate **any** continuous function on a compact set → the **magic** behind “stack layers and it just works”.

## 4. Transformer building blocks – attention as a classroom

* **Embedding tables**:

  * wte (token) – vocabulary semantics
  * wpe (position) – order semantics\
    Added element-wise before the first block.

* **Self-attention classroom analogy**:

  * Query = student raising hand “Who can help me?”
  * Key = classmates shouting “I know this!”
  * Value = the actual explanation they give\
    Softmax turns raised-hand strength into **attention weights**.

* **Multi-Head**: several independent classrooms in parallel; each learns different relationship types.

* **Residual highway**: `output = f(x) + x` – keeps gradient super-highway open (no vanishing).

* **Feed-Forward block**: same MLP reused at every position; no cross-talk → **position-wise**.

* **LayerNorm**: zero-mean, unit-variance → stabilizes training; placed **before** attention & MLP (Pre-Norm).

* **GPT-2 scale**: 12 blocks, 12 heads, 768 dim, 50257 vocab → ~117 M parameters.

## 5. Inference – from text to probabilities

* **torch.no\_grad()**: context-manager that **turns off autograd bookkeeping** → memory ↓, speed ↑.

* **logits**: raw scores; can be **positive or negative**; softmax squashes them to probabilities.

* **Softmax temperature**:

  * high T → flatter → more randomness
  * low T → sharper → more greedy

* **Top-k**: keep only k highest logits, re-normalize, sample.

* **Top-p (nucleus)**: dynamic k – smallest set whose cumulative prob ≥ p.

* **Greedy decode**: always pick argmax; deterministic; can loop repetitive text.

* **Shape mantra**:\
  `(batch, seq_len, vocab)` → slice `[:, -1, :]` for **next-token** logits.

## 6. Generation helpers – one function, three strategies

______

Python

```
def generate(model, tokenizer, prompt, strategy="greedy", temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = strategy != "greedy"
    top_k = 50 if strategy == "top_k" else None
    top_p = 0.9 if strategy == "top_p" else None
    out = model.generate(**inputs, max_new_tokens=128, do_sample=do_sample,
                         temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

* **Completion-only** (GPT-2) vs **instruction-tuned** (Qwen):

  * Completion continues the prompt.
  * Instruction follows directions → prepend “### Instruction: …” for best results.

## 7. Interactive playground – widgets in 30 s

* **ipywidgets pattern**:
  1. Build widgets → 2. Collect values → 3. Button callback → 4. Output widget to capture prints.

* **Key widgets**:

  * Textarea (prompt)
  * Dropdown (model, strategy)
  * FloatSlider (temperature)
  * Button (trigger)
  * Output (display Markdown without print clutter)

* **Display trick**: wrap everything in `VBox([...])` for vertical layout.

## 8. Attention Mask (quick peek)

* **Causal mask**: lower-triangular matrix → model can’t peek at future tokens.
* **Boolean tensor** `True` = **ignore**; HF does this internally when `model.generate(...)`
* Shape: `(seq, seq)` or broadcasted `(B, 1, seq, seq)` for multi-head.

______

Python

```
mask = torch.tril(torch.ones(seq, seq)) == 0   # 1 = ignore
```

***

## 9. Beam Search – width-controlled exploration

* Keeps **k best partial sequences** at each step (k = beam\_width).
* **Deterministic**, higher quality than greedy; slower.
* HF flag: `num_beams=4` (turns off sampling).

______

Python

```
out = model.generate(**inputs, num_beams=4, early_stopping=True)
```

***

## 10. Device hygiene – keep tensors together

* `.to(device)` is **idempotent**; call once on model **and** on every new tensor.
* **MPS (Apple GPU)** behaves like CUDA; fallback order: cuda → mps → cpu.
* **One-liner check**:

______

Python

```
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

***

## 11. Training vs Inference – the gradient switch

* **Training**: gradients ON (default), loss.backward(), optimizer.step().
* **Inference**: wrap with `torch.no_grad()` to freeze weights & save memory.
* **Fine-tuning mini-loop (pseudo)**:

______

Python

```
for b in loader:
    opt.zero_grad(); loss = model(**b).loss; loss.backward(); opt.step()
```

***

## 12. Temperature & Repetition Penalty – two knobs

* **Temperature** ∈ (0, ∞) rescales logits **before** softmax.
* **Repetition penalty** shrinks probability of **already generated tokens**; HF: `repetition_penalty=1.2`.
* Use both together for creative but non-repetitive text.

***

## 13. EOS & PAD – silent string terminators

* **EOS token id** tells generate() when to stop (HF adds it automatically).
* **PAD token** needed for batches of unequal length; GPT-2 tokenizer has **no default pad**, so we often set:

______

Python

```
tokenizer.pad_token = tokenizer.eos_token
```

***

## 14. Model Hub aliases – what you met

* **gpt2** ↔ 117 M params, 12 layers, 768 dim
* **Qwen/Qwen3-0.6B** ↔ 0.6 B params, instruction-tuned, supports system prompts

***

## 15. Widget pro-tip – refresh without rerun

* Call `output_widget.clear_output()` inside callback to **replace** old text instead of appending.

