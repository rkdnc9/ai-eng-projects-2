# Adapting LLMs and Retrieval-Augmented Generation (RAG)

## Overview

This lecture notes provides a comprehensive guide to adapting general-purpose Large Language Models (LLMs) for specialized, domain-specific applications. The primary focus is building a customer support chatbot for an e-commerce retail store. The lecture covers three major adaptation techniques: fine-tuning, prompt engineering, and Retrieval-Augmented Generation (RAG). Special emphasis is placed on RAG systems, including detailed coverage of document processing, embedding models, vector search algorithms, and complete system design with practical implementation considerations.

## Core Concepts

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

## Fine-Tuning

### Overview

**Definition**: Continue training a general-purpose LLM on domain-specific document database

**Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 952.875px; background-color: transparent;" viewBox="0 0 952.875 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M218.141,47L222.307,47C226.474,47,234.807,47,242.474,47C250.141,47,257.141,47,260.641,47L264.141,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjE4LjE0MDYyNSwieSI6NDd9LHsieCI6MjQzLjE0MDYyNSwieSI6NDd9LHsieCI6MjY4LjE0MDYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M470.609,47L474.776,47C478.943,47,487.276,47,494.943,47C502.609,47,509.609,47,513.109,47L516.609,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDcwLjYwOTM3NSwieSI6NDd9LHsieCI6NDk1LjYwOTM3NSwieSI6NDd9LHsieCI6NTIwLjYwOTM3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M693.734,47L697.901,47C702.068,47,710.401,47,718.068,47C725.734,47,732.734,47,736.234,47L739.734,47" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NjkzLjczNDM3NSwieSI6NDd9LHsieCI6NzE4LjczNDM3NSwieSI6NDd9LHsieCI6NzQzLjczNDM3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(113.0703125, 47)"><rect class="basic label-container" style="" x="-105.0703125" y="-27" width="210.140625" height="54"/><g class="label" style="" transform="translate(-75.0703125, -12)"><rect/><foreignObject width="150.140625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>General Purpose LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(369.375, 47)"><rect class="basic label-container" style="" x="-101.234375" y="-39" width="202.46875" height="78"/><g class="label" style="" transform="translate(-71.234375, -24)"><rect/><foreignObject width="142.46875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Fine-Tuning on<br />Document Database</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(607.171875, 47)"><rect class="basic label-container" style="" x="-86.5625" y="-27" width="173.125" height="54"/><g class="label" style="" transform="translate(-56.5625, -12)"><rect/><foreignObject width="113.125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Specialized LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(844.3046875, 47)"><rect class="basic label-container" style="" x="-100.5703125" y="-27" width="201.140625" height="54"/><g class="label" style="" transform="translate(-70.5703125, -12)"><rect/><foreignObject width="141.140625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Ready for Inference</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 181.641px; background-color: transparent;" viewBox="0 0 181.640625 870" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M90.82,62L90.82,66.167C90.82,70.333,90.82,78.667,90.82,86.333C90.82,94,90.82,101,90.82,104.5L90.82,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6NjJ9LHsieCI6OTAuODIwMzEyNSwieSI6ODd9LHsieCI6OTAuODIwMzEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,166L90.82,170.167C90.82,174.333,90.82,182.667,90.82,190.333C90.82,198,90.82,205,90.82,208.5L90.82,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6MTY2fSx7IngiOjkwLjgyMDMxMjUsInkiOjE5MX0seyJ4Ijo5MC44MjAzMTI1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,294L90.82,298.167C90.82,302.333,90.82,310.667,90.82,318.333C90.82,326,90.82,333,90.82,336.5L90.82,340" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6Mjk0fSx7IngiOjkwLjgyMDMxMjUsInkiOjMxOX0seyJ4Ijo5MC44MjAzMTI1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,398L90.82,402.167C90.82,406.333,90.82,414.667,90.82,422.333C90.82,430,90.82,437,90.82,440.5L90.82,444" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6Mzk4fSx7IngiOjkwLjgyMDMxMjUsInkiOjQyM30seyJ4Ijo5MC44MjAzMTI1LCJ5Ijo0NDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,526L90.82,530.167C90.82,534.333,90.82,542.667,90.82,550.333C90.82,558,90.82,565,90.82,568.5L90.82,572" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6NTI2fSx7IngiOjkwLjgyMDMxMjUsInkiOjU1MX0seyJ4Ijo5MC44MjAzMTI1LCJ5Ijo1NzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,630L90.82,634.167C90.82,638.333,90.82,646.667,90.82,654.333C90.82,662,90.82,669,90.82,672.5L90.82,676" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6NjMwfSx7IngiOjkwLjgyMDMxMjUsInkiOjY1NX0seyJ4Ijo5MC44MjAzMTI1LCJ5Ijo2ODB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.82,758L90.82,762.167C90.82,766.333,90.82,774.667,90.82,782.333C90.82,790,90.82,797,90.82,800.5L90.82,804" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6OTAuODIwMzEyNSwieSI6NzU4fSx7IngiOjkwLjgyMDMxMjUsInkiOjc4M30seyJ4Ijo5MC44MjAzMTI1LCJ5Ijo4MDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(90.8203125, 35)"><rect class="basic label-container" style="" x="-48.6015625" y="-27" width="97.203125" height="54"/><g class="label" style="" transform="translate(-18.6015625, -12)"><rect/><foreignObject width="37.203125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(90.8203125, 139)"><rect class="basic label-container" style="" x="-82.8203125" y="-27" width="165.640625" height="54"/><g class="label" style="" transform="translate(-52.8203125, -12)"><rect/><foreignObject width="105.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Frozen Layer 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(90.8203125, 255)"><rect class="basic label-container" style="" x="-80.5625" y="-39" width="161.125" height="78"/><g class="label" style="" transform="translate(-50.5625, -24)"><rect/><foreignObject width="101.125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Adapter Layer<br />Trainable</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(90.8203125, 371)"><rect class="basic label-container" style="" x="-82.8203125" y="-27" width="165.640625" height="54"/><g class="label" style="" transform="translate(-52.8203125, -12)"><rect/><foreignObject width="105.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Frozen Layer 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(90.8203125, 487)"><rect class="basic label-container" style="" x="-80.5625" y="-39" width="161.125" height="78"/><g class="label" style="" transform="translate(-50.5625, -24)"><rect/><foreignObject width="101.125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Adapter Layer<br />Trainable</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(90.8203125, 603)"><rect class="basic label-container" style="" x="-82.8203125" y="-27" width="165.640625" height="54"/><g class="label" style="" transform="translate(-52.8203125, -12)"><rect/><foreignObject width="105.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Frozen Layer 3</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(90.8203125, 719)"><rect class="basic label-container" style="" x="-80.5625" y="-39" width="161.125" height="78"/><g class="label" style="" transform="translate(-50.5625, -24)"><rect/><foreignObject width="101.125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Adapter Layer<br />Trainable</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(90.8203125, 835)"><rect class="basic label-container" style="" x="-54.9375" y="-27" width="109.875" height="54"/><g class="label" style="" transform="translate(-24.9375, -12)"><rect/><foreignObject width="49.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 656.594px; background-color: transparent;" viewBox="0 0 656.59375 222" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M96.772,84L104.482,77.833C112.192,71.667,127.612,59.333,138.821,53.167C150.031,47,157.031,47,160.531,47L164.031,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6OTYuNzcyMjE2Nzk2ODc1LCJ5Ijo4NH0seyJ4IjoxNDMuMDMxMjUsInkiOjQ3fSx7IngiOjE2OC4wMzEyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M96.772,138L104.482,144.167C112.192,150.333,127.612,162.667,139.976,168.833C152.341,175,161.651,175,166.306,175L170.961,175" id="L_A_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_C_0" data-points="W3sieCI6OTYuNzcyMjE2Nzk2ODc1LCJ5IjoxMzh9LHsieCI6MTQzLjAzMTI1LCJ5IjoxNzV9LHsieCI6MTc0Ljk2MDkzNzUsInkiOjE3NX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M318.656,47L322.823,47C326.99,47,335.323,47,347.15,52.766C358.977,58.532,374.298,70.063,381.959,75.829L389.619,81.595" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6MzE4LjY1NjI1LCJ5Ijo0N30seyJ4IjozNDMuNjU2MjUsInkiOjQ3fSx7IngiOjM5Mi44MTQ5NDE0MDYyNSwieSI6ODR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M311.727,175L317.048,175C322.37,175,333.013,175,345.995,169.234C358.977,163.468,374.298,151.937,381.959,146.171L389.619,140.405" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MzExLjcyNjU2MjUsInkiOjE3NX0seyJ4IjozNDMuNjU2MjUsInkiOjE3NX0seyJ4IjozOTIuODE0OTQxNDA2MjUsInkiOjEzOH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M488.719,111L492.885,111C497.052,111,505.385,111,513.052,111C520.719,111,527.719,111,531.219,111L534.719,111" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NDg4LjcxODc1LCJ5IjoxMTF9LHsieCI6NTEzLjcxODc1LCJ5IjoxMTF9LHsieCI6NTM4LjcxODc1LCJ5IjoxMTF9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(63.015625, 111)"><rect class="basic label-container" style="" x="-55.015625" y="-27" width="110.03125" height="54"/><g class="label" style="" transform="translate(-25.015625, -12)"><rect/><foreignObject width="50.03125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input x</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(243.34375, 47)"><rect class="basic label-container" style="" x="-75.3125" y="-39" width="150.625" height="78"/><g class="label" style="" transform="translate(-45.3125, -24)"><rect/><foreignObject width="90.625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Frozen Path:<br />W·x</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(243.34375, 175)"><rect class="basic label-container" style="" x="-68.3828125" y="-39" width="136.765625" height="78"/><g class="label" style="" transform="translate(-38.3828125, -24)"><rect/><foreignObject width="76.765625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LoRa Path:<br />B·A·x</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(428.6875, 111)"><rect class="basic label-container" style="" x="-60.03125" y="-27" width="120.0625" height="54"/><g class="label" style="" transform="translate(-30.03125, -12)"><rect/><foreignObject width="60.0625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Addition</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-9" transform="translate(593.65625, 111)"><rect class="basic label-container" style="" x="-54.9375" y="-27" width="109.875" height="54"/><g class="label" style="" transform="translate(-24.9375, -12)"><rect/><foreignObject width="49.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 467.891px; background-color: transparent;" viewBox="0 0 467.890625 70" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M147.016,35L151.182,35C155.349,35,163.682,35,171.349,35C179.016,35,186.016,35,189.516,35L193.016,35" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTQ3LjAxNTYyNSwieSI6MzV9LHsieCI6MTcyLjAxNTYyNSwieSI6MzV9LHsieCI6MTk3LjAxNTYyNSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M284.578,35L288.745,35C292.911,35,301.245,35,308.911,35C316.578,35,323.578,35,327.078,35L330.578,35" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6Mjg0LjU3ODEyNSwieSI6MzV9LHsieCI6MzA5LjU3ODEyNSwieSI6MzV9LHsieCI6MzM0LjU3ODEyNSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(77.5078125, 35)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(240.796875, 35)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(397.234375, 35)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Prompt Engineering Workflow**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 953px; background-color: transparent;" viewBox="0 0 953 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M147.016,47L151.182,47C155.349,47,163.682,47,171.349,47C179.016,47,186.016,47,189.516,47L193.016,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTQ3LjAxNTYyNSwieSI6NDd9LHsieCI6MTcyLjAxNTYyNSwieSI6NDd9LHsieCI6MTk3LjAxNTYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M397.016,47L401.182,47C405.349,47,413.682,47,421.349,47C429.016,47,436.016,47,439.516,47L443.016,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6Mzk3LjAxNTYyNSwieSI6NDd9LHsieCI6NDIyLjAxNTYyNSwieSI6NDd9LHsieCI6NDQ3LjAxNTYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M632.125,47L636.292,47C640.458,47,648.792,47,656.458,47C664.125,47,671.125,47,674.625,47L678.125,47" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NjMyLjEyNSwieSI6NDd9LHsieCI6NjU3LjEyNSwieSI6NDd9LHsieCI6NjgyLjEyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M769.688,47L773.854,47C778.021,47,786.354,47,794.021,47C801.688,47,808.688,47,812.188,47L815.688,47" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NzY5LjY4NzUsInkiOjQ3fSx7IngiOjc5NC42ODc1LCJ5Ijo0N30seyJ4Ijo4MTkuNjg3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(77.5078125, 47)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(297.015625, 47)"><rect class="basic label-container" style="" x="-100" y="-39" width="200" height="78"/><g class="label" style="" transform="translate(-70, -24)"><rect/><foreignObject width="140" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt Engineering<br />Logic</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(539.5703125, 47)"><rect class="basic label-container" style="" x="-92.5546875" y="-27" width="185.109375" height="54"/><g class="label" style="" transform="translate(-62.5546875, -12)"><rect/><foreignObject width="125.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Enhanced Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(725.90625, 47)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(882.34375, 47)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 708.938px; background-color: transparent;" viewBox="0 0 708.9375 222" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M206.766,47L210.932,47C215.099,47,223.432,47,235.789,52.782C248.145,58.564,264.525,70.129,272.714,75.911L280.904,81.693" id="L_A_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_C_0" data-points="W3sieCI6MjA2Ljc2NTYyNSwieSI6NDd9LHsieCI6MjMxLjc2NTYyNSwieSI6NDd9LHsieCI6Mjg0LjE3MTc1MjkyOTY4NzUsInkiOjg0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M181.453,175L189.839,175C198.224,175,214.995,175,231.57,169.218C248.145,163.436,264.525,151.871,272.714,146.089L280.904,140.307" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTgxLjQ1MzEyNSwieSI6MTc1fSx7IngiOjIzMS43NjU2MjUsInkiOjE3NX0seyJ4IjoyODQuMTcxNzUyOTI5Njg3NSwieSI6MTM4fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M388.063,111L392.229,111C396.396,111,404.729,111,412.396,111C420.063,111,427.063,111,430.563,111L434.063,111" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6Mzg4LjA2MjUsInkiOjExMX0seyJ4Ijo0MTMuMDYyNSwieSI6MTExfSx7IngiOjQzOC4wNjI1LCJ5IjoxMTF9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M525.625,111L529.792,111C533.958,111,542.292,111,549.958,111C557.625,111,564.625,111,568.125,111L571.625,111" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NTI1LjYyNSwieSI6MTExfSx7IngiOjU1MC42MjUsInkiOjExMX0seyJ4Ijo1NzUuNjI1LCJ5IjoxMTF9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(107.3828125, 47)"><rect class="basic label-container" style="" x="-99.3828125" y="-39" width="198.765625" height="78"/><g class="label" style="" transform="translate(-69.3828125, -24)"><rect/><foreignObject width="138.765625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>System Prompt<br />Hidden Instructions</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-1" transform="translate(322.4140625, 111)"><rect class="basic label-container" style="" x="-65.6484375" y="-27" width="131.296875" height="54"/><g class="label" style="" transform="translate(-35.6484375, -12)"><rect/><foreignObject width="71.296875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Combined</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-2" transform="translate(107.3828125, 175)"><rect class="basic label-container" style="" x="-74.0703125" y="-39" width="148.140625" height="78"/><g class="label" style="" transform="translate(-44.0703125, -24)"><rect/><foreignObject width="88.140625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Prompt<br />User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(481.84375, 111)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(638.28125, 111)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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

## Retrieval-Augmented Generation (RAG)

### Overview and Motivation

**Problem with Prompt Engineering**: Cannot include entire document database due to context window limits

**RAG Solution**: Add retrieval component that selects only relevant documents

**Architecture Comparison**:

Prompt Engineering:

<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 497.375px; background-color: transparent;" viewBox="0 0 497.375 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M176.5,47L180.667,47C184.833,47,193.167,47,200.833,47C208.5,47,215.5,47,219,47L222.5,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTc2LjUsInkiOjQ3fSx7IngiOjIwMS41LCJ5Ijo0N30seyJ4IjoyMjYuNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M314.063,47L318.229,47C322.396,47,330.729,47,338.396,47C346.063,47,353.063,47,356.563,47L360.063,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzE0LjA2MjUsInkiOjQ3fSx7IngiOjMzOS4wNjI1LCJ5Ijo0N30seyJ4IjozNjQuMDYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(92.25, 47)"><rect class="basic label-container" style="" x="-84.25" y="-39" width="168.5" height="78"/><g class="label" style="" transform="translate(-54.25, -24)"><rect/><foreignObject width="108.5" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query +<br />ALL Documents</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(270.28125, 47)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(426.71875, 47)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


RAG:

<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1013.48px; background-color: transparent;" viewBox="0 0 1013.484375 174" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M178.742,23.437L188.197,21.864C197.651,20.291,216.56,17.146,229.533,16.421C242.506,15.697,249.543,17.393,253.062,18.242L256.58,19.09" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTc4Ljc0MjE4NzUsInkiOjIzLjQzNjg3MzM3NTQxNzc1fSx7IngiOjIzNS40Njg3NSwieSI6MTR9LHsieCI6MjYwLjQ2ODc1LCJ5IjoyMC4wMjc0NDY0MDc3NDk1NzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M210.469,122.159L214.635,121.466C218.802,120.773,227.135,119.386,240.614,109.82C254.093,100.253,272.716,82.506,282.028,73.633L291.34,64.759" id="L_C_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_B_0" data-points="W3sieCI6MjEwLjQ2ODc1LCJ5IjoxMjIuMTU4OTMwNTYwNzEyOTZ9LHsieCI6MjM1LjQ2ODc1LCJ5IjoxMTh9LHsieCI6Mjk0LjIzNjA2OTI3NzEwODQsInkiOjYyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M384.672,35L388.839,35C393.005,35,401.339,35,409.005,35C416.672,35,423.672,35,427.172,35L430.672,35" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6Mzg0LjY3MTg3NSwieSI6MzV9LHsieCI6NDA5LjY3MTg3NSwieSI6MzV9LHsieCI6NDM0LjY3MTg3NSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M640.297,35L644.464,35C648.63,35,656.964,35,664.634,35.369C672.304,35.738,679.312,36.476,682.815,36.845L686.319,37.214" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NjQwLjI5Njg3NSwieSI6MzV9LHsieCI6NjY1LjI5Njg3NSwieSI6MzV9LHsieCI6NjkwLjI5Njg3NSwieSI6MzcuNjMzMzExMzg5MDcxNzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M136.721,62L153.179,78.167C169.637,94.333,202.553,126.667,233.528,142.833C264.503,159,293.536,159,322.57,159C351.604,159,380.638,159,416.457,159C452.276,159,494.88,159,537.484,159C580.089,159,622.693,159,655.644,145.012C688.594,131.025,711.892,103.049,723.541,89.061L735.189,75.074" id="L_A_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_E_0" data-points="W3sieCI6MTM2LjcyMDg5MjEzNzA5Njc3LCJ5Ijo2Mn0seyJ4IjoyMzUuNDY4NzUsInkiOjE1OX0seyJ4IjozMjIuNTcwMzEyNSwieSI6MTU5fSx7IngiOjQwOS42NzE4NzUsInkiOjE1OX0seyJ4Ijo1MzcuNDg0Mzc1LCJ5IjoxNTl9LHsieCI6NjY1LjI5Njg3NSwieSI6MTU5fSx7IngiOjczNy43NDkxNzc2MzE1NzksInkiOjcyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M830.172,45L834.339,45C838.505,45,846.839,45,854.505,45C862.172,45,869.172,45,872.672,45L876.172,45" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6ODMwLjE3MTg3NSwieSI6NDV9LHsieCI6ODU1LjE3MTg3NSwieSI6NDV9LHsieCI6ODgwLjE3MTg3NSwieSI6NDV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(109.234375, 35)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(322.5703125, 35)"><rect class="basic label-container" style="" x="-62.1015625" y="-27" width="124.203125" height="54"/><g class="label" style="" transform="translate(-32.1015625, -12)"><rect/><foreignObject width="64.203125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieval</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-2" transform="translate(109.234375, 139)"><rect class="basic label-container" style="" x="-101.234375" y="-27" width="202.46875" height="54"/><g class="label" style="" transform="translate(-71.234375, -12)"><rect/><foreignObject width="142.46875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Document Database</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(537.484375, 35)"><rect class="basic label-container" style="" x="-102.8125" y="-27" width="205.625" height="54"/><g class="label" style="" transform="translate(-72.8125, -12)"><rect/><foreignObject width="145.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Relevant Documents</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(760.234375, 45)"><rect class="basic label-container" style="" x="-69.9375" y="-27" width="139.875" height="54"/><g class="label" style="" transform="translate(-39.9375, -12)"><rect/><foreignObject width="79.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generation</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-11" transform="translate(942.828125, 45)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 669.219px; background-color: transparent;" viewBox="0 0 669.21875 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M218.063,47L222.229,47C226.396,47,234.729,47,242.396,47C250.063,47,257.063,47,260.563,47L264.063,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjE4LjA2MjUsInkiOjQ3fSx7IngiOjI0My4wNjI1LCJ5Ijo0N30seyJ4IjoyNjguMDYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M429.063,47L433.229,47C437.396,47,445.729,47,453.396,47C461.063,47,468.063,47,471.563,47L475.063,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDI5LjA2MjUsInkiOjQ3fSx7IngiOjQ1NC4wNjI1LCJ5Ijo0N30seyJ4Ijo0NzkuMDYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(113.03125, 47)"><rect class="basic label-container" style="" x="-105.03125" y="-39" width="210.0625" height="78"/><g class="label" style="" transform="translate(-75.03125, -24)"><rect/><foreignObject width="150.0625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Document Database<br />PDFs, HTMLs, Images</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(348.5625, 47)"><rect class="basic label-container" style="" x="-80.5" y="-39" width="161" height="78"/><g class="label" style="" transform="translate(-50.5, -24)"><rect/><foreignObject width="101" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Index Building<br />Logic</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(570.140625, 47)"><rect class="basic label-container" style="" x="-91.078125" y="-27" width="182.15625" height="54"/><g class="label" style="" transform="translate(-61.078125, -12)"><rect/><foreignObject width="122.15625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Searchable Index</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Search Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 631.781px; background-color: transparent;" viewBox="0 0 631.78125 174" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M168.586,35L176.348,35C184.109,35,199.633,35,214.805,38.859C229.978,42.718,244.8,50.435,252.211,54.294L259.621,58.153" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTY4LjU4NTkzNzUsInkiOjM1fSx7IngiOjIxNS4xNTYyNSwieSI6MzV9LHsieCI6MjYzLjE2OTMyMDkxMzQ2MTU1LCJ5Ijo2MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M190.156,139L194.323,139C198.49,139,206.823,139,218.4,135.141C229.978,131.282,244.8,123.565,252.211,119.706L259.621,115.847" id="L_C_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_B_0" data-points="W3sieCI6MTkwLjE1NjI1LCJ5IjoxMzl9LHsieCI6MjE1LjE1NjI1LCJ5IjoxMzl9LHsieCI6MjYzLjE2OTMyMDkxMzQ2MTU1LCJ5IjoxMTR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M389.891,87L394.057,87C398.224,87,406.557,87,414.224,87C421.891,87,428.891,87,432.391,87L435.891,87" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6Mzg5Ljg5MDYyNSwieSI6ODd9LHsieCI6NDE0Ljg5MDYyNSwieSI6ODd9LHsieCI6NDM5Ljg5MDYyNSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(99.078125, 35)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(315.0234375, 87)"><rect class="basic label-container" style="" x="-74.8671875" y="-27" width="149.734375" height="54"/><g class="label" style="" transform="translate(-44.8671875, -12)"><rect/><foreignObject width="89.734375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Search Logic</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-2" transform="translate(99.078125, 139)"><rect class="basic label-container" style="" x="-91.078125" y="-27" width="182.15625" height="54"/><g class="label" style="" transform="translate(-61.078125, -12)"><rect/><foreignObject width="122.15625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Searchable Index</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(531.8359375, 87)"><rect class="basic label-container" style="" x="-91.9453125" y="-27" width="183.890625" height="54"/><g class="label" style="" transform="translate(-61.9453125, -12)"><rect/><foreignObject width="123.890625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieved Chunks</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 253.609px; background-color: transparent;" viewBox="0 0 253.609375 534" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M126.805,62L126.805,66.167C126.805,70.333,126.805,78.667,126.805,86.333C126.805,94,126.805,101,126.805,104.5L126.805,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTI2LjgwNDY4NzUsInkiOjYyfSx7IngiOjEyNi44MDQ2ODc1LCJ5Ijo4N30seyJ4IjoxMjYuODA0Njg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M126.805,166L126.805,170.167C126.805,174.333,126.805,182.667,126.805,190.333C126.805,198,126.805,205,126.805,208.5L126.805,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTI2LjgwNDY4NzUsInkiOjE2Nn0seyJ4IjoxMjYuODA0Njg3NSwieSI6MTkxfSx7IngiOjEyNi44MDQ2ODc1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M126.805,294L126.805,298.167C126.805,302.333,126.805,310.667,126.805,318.333C126.805,326,126.805,333,126.805,336.5L126.805,340" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTI2LjgwNDY4NzUsInkiOjI5NH0seyJ4IjoxMjYuODA0Njg3NSwieSI6MzE5fSx7IngiOjEyNi44MDQ2ODc1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M126.805,422L126.805,426.167C126.805,430.333,126.805,438.667,126.805,446.333C126.805,454,126.805,461,126.805,464.5L126.805,468" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTI2LjgwNDY4NzUsInkiOjQyMn0seyJ4IjoxMjYuODA0Njg3NSwieSI6NDQ3fSx7IngiOjEyNi44MDQ2ODc1LCJ5Ijo0NzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(126.8046875, 35)"><rect class="basic label-container" style="" x="-82.0546875" y="-27" width="164.109375" height="54"/><g class="label" style="" transform="translate(-52.0546875, -12)"><rect/><foreignObject width="104.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>PDF Document</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(126.8046875, 139)"><rect class="basic label-container" style="" x="-91.328125" y="-27" width="182.65625" height="54"/><g class="label" style="" transform="translate(-61.328125, -12)"><rect/><foreignObject width="122.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Layout Detection</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(126.8046875, 255)"><rect class="basic label-container" style="" x="-118.8046875" y="-39" width="237.609375" height="78"/><g class="label" style="" transform="translate(-88.8046875, -24)"><rect/><foreignObject width="177.609375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Identify Components:<br />Title, Text, Figure, Table</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(126.8046875, 383)"><rect class="basic label-container" style="" x="-84.4609375" y="-39" width="168.921875" height="78"/><g class="label" style="" transform="translate(-54.4609375, -24)"><rect/><foreignObject width="108.921875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Extraction<br />OCR Methods</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(126.8046875, 499)"><rect class="basic label-container" style="" x="-95.28125" y="-27" width="190.5625" height="54"/><g class="label" style="" transform="translate(-65.28125, -12)"><rect/><foreignObject width="130.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Structured Output</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 825.875px; background-color: transparent;" viewBox="0 0 825.875 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M253.75,47L257.917,47C262.083,47,270.417,47,278.083,47C285.75,47,292.75,47,296.25,47L299.75,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjUzLjc1LCJ5Ijo0N30seyJ4IjoyNzguNzUsInkiOjQ3fSx7IngiOjMwMy43NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M524.734,47L528.901,47C533.068,47,541.401,47,549.068,47C556.734,47,563.734,47,567.234,47L570.734,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NTI0LjczNDM3NSwieSI6NDd9LHsieCI6NTQ5LjczNDM3NSwieSI6NDd9LHsieCI6NTc0LjczNDM3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(130.875, 47)"><rect class="basic label-container" style="" x="-122.875" y="-39" width="245.75" height="78"/><g class="label" style="" transform="translate(-92.875, -24)"><rect/><foreignObject width="185.75" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text: To learn more about<br />our products, contact us</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(414.2421875, 47)"><rect class="basic label-container" style="" x="-110.4921875" y="-27" width="220.984375" height="54"/><g class="label" style="" transform="translate(-80.4921875, -12)"><rect/><foreignObject width="160.984375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Embedding Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(696.3046875, 47)"><rect class="basic label-container" style="" x="-121.5703125" y="-39" width="243.140625" height="78"/><g class="label" style="" transform="translate(-91.5703125, -24)"><rect/><foreignObject width="183.140625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Vector: 0.23, -0.15, 0.67,<br />..., 0.42</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 236.984px; background-color: transparent;" viewBox="0 0 236.984375 382" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M118.492,62L118.492,66.167C118.492,70.333,118.492,78.667,118.492,86.333C118.492,94,118.492,101,118.492,104.5L118.492,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTE4LjQ5MjE4NzUsInkiOjYyfSx7IngiOjExOC40OTIxODc1LCJ5Ijo4N30seyJ4IjoxMTguNDkyMTg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M118.492,166L118.492,170.167C118.492,174.333,118.492,182.667,118.492,190.333C118.492,198,118.492,205,118.492,208.5L118.492,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTE4LjQ5MjE4NzUsInkiOjE2Nn0seyJ4IjoxMTguNDkyMTg3NSwieSI6MTkxfSx7IngiOjExOC40OTIxODc1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M118.492,270L118.492,274.167C118.492,278.333,118.492,286.667,118.492,294.333C118.492,302,118.492,309,118.492,312.5L118.492,316" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTE4LjQ5MjE4NzUsInkiOjI3MH0seyJ4IjoxMTguNDkyMTg3NSwieSI6Mjk1fSx7IngiOjExOC40OTIxODc1LCJ5IjozMjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(118.4921875, 35)"><rect class="basic label-container" style="" x="-72.7734375" y="-27" width="145.546875" height="54"/><g class="label" style="" transform="translate(-42.7734375, -12)"><rect/><foreignObject width="85.546875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Chunks</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(118.4921875, 139)"><rect class="basic label-container" style="" x="-110.4921875" y="-27" width="220.984375" height="54"/><g class="label" style="" transform="translate(-80.4921875, -12)"><rect/><foreignObject width="160.984375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Embedding Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(118.4921875, 243)"><rect class="basic label-container" style="" x="-96.9140625" y="-27" width="193.828125" height="54"/><g class="label" style="" transform="translate(-66.9140625, -12)"><rect/><foreignObject width="133.828125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Chunk Embeddings</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(118.4921875, 347)"><rect class="basic label-container" style="" x="-70.7265625" y="-27" width="141.453125" height="54"/><g class="label" style="" transform="translate(-40.7265625, -12)"><rect/><foreignObject width="81.453125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Index Table</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 384.297px; background-color: transparent;" viewBox="0 0 384.296875 278" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M90.703,62L90.703,66.167C90.703,70.333,90.703,78.667,90.703,86.333C90.703,94,90.703,101,90.703,104.5L90.703,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6OTAuNzAzMTI1LCJ5Ijo2Mn0seyJ4Ijo5MC43MDMxMjUsInkiOjg3fSx7IngiOjkwLjcwMzEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M299.852,62L299.852,66.167C299.852,70.333,299.852,78.667,299.852,86.333C299.852,94,299.852,101,299.852,104.5L299.852,108" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6Mjk5Ljg1MTU2MjUsInkiOjYyfSx7IngiOjI5OS44NTE1NjI1LCJ5Ijo4N30seyJ4IjoyOTkuODUxNTYyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M90.703,166L90.703,170.167C90.703,174.333,90.703,182.667,98.486,190.703C106.268,198.74,121.833,206.479,129.615,210.349L137.398,214.219" id="L_B_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_E_0" data-points="W3sieCI6OTAuNzAzMTI1LCJ5IjoxNjZ9LHsieCI6OTAuNzAzMTI1LCJ5IjoxOTF9LHsieCI6MTQwLjk3OTE5MTcwNjczMDc3LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M299.852,166L299.852,170.167C299.852,174.333,299.852,182.667,292.069,190.703C284.287,198.74,268.722,206.479,260.94,210.349L253.157,214.219" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6Mjk5Ljg1MTU2MjUsInkiOjE2Nn0seyJ4IjoyOTkuODUxNTYyNSwieSI6MTkxfSx7IngiOjI0OS41NzU0OTU3OTMyNjkyMywieSI6MjE2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(90.703125, 35)"><rect class="basic label-container" style="" x="-51.453125" y="-27" width="102.90625" height="54"/><g class="label" style="" transform="translate(-21.453125, -12)"><rect/><foreignObject width="42.90625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Image</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(90.703125, 139)"><rect class="basic label-container" style="" x="-82.703125" y="-27" width="165.40625" height="54"/><g class="label" style="" transform="translate(-52.703125, -12)"><rect/><foreignObject width="105.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Image Encoder</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-2" transform="translate(299.8515625, 35)"><rect class="basic label-container" style="" x="-45.1953125" y="-27" width="90.390625" height="54"/><g class="label" style="" transform="translate(-15.1953125, -12)"><rect/><foreignObject width="30.390625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-3" transform="translate(299.8515625, 139)"><rect class="basic label-container" style="" x="-76.4453125" y="-27" width="152.890625" height="54"/><g class="label" style="" transform="translate(-46.4453125, -12)"><rect/><foreignObject width="92.890625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Encoder</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-5" transform="translate(195.27734375, 243)"><rect class="basic label-container" style="" x="-119.3359375" y="-27" width="238.671875" height="54"/><g class="label" style="" transform="translate(-89.3359375, -12)"><rect/><foreignObject width="178.671875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Shared Embedding Space</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1052.69px; background-color: transparent;" viewBox="0 0 1052.6875 174" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M147.016,35L151.182,35C155.349,35,163.682,35,171.349,35C179.016,35,186.016,35,189.516,35L193.016,35" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTQ3LjAxNTYyNSwieSI6MzV9LHsieCI6MTcyLjAxNTYyNSwieSI6MzV9LHsieCI6MTk3LjAxNTYyNSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M349.906,35L354.073,35C358.24,35,366.573,35,374.24,35C381.906,35,388.906,35,392.406,35L395.906,35" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzQ5LjkwNjI1LCJ5IjozNX0seyJ4IjozNzQuOTA2MjUsInkiOjM1fSx7IngiOjM5OS45MDYyNSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M585.781,35L589.948,35C594.115,35,602.448,35,615.284,38.894C628.121,42.787,645.461,50.574,654.13,54.468L662.8,58.361" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTg1Ljc4MTI1LCJ5IjozNX0seyJ4Ijo2MTAuNzgxMjUsInkiOjM1fSx7IngiOjY2Ni40NDkwNjg1MDk2MTU0LCJ5Ijo2MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M542.273,139L553.691,139C565.109,139,587.945,139,608.033,135.106C628.121,131.213,645.461,123.426,654.13,119.532L662.8,115.639" id="L_E_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_D_0" data-points="W3sieCI6NTQyLjI3MzQzNzUsInkiOjEzOX0seyJ4Ijo2MTAuNzgxMjUsInkiOjEzOX0seyJ4Ijo2NjYuNDQ5MDY4NTA5NjE1NCwieSI6MTE0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M817.359,87L821.526,87C825.693,87,834.026,87,841.693,87C849.359,87,856.359,87,859.859,87L863.359,87" id="L_D_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_F_0" data-points="W3sieCI6ODE3LjM1OTM3NSwieSI6ODd9LHsieCI6ODQyLjM1OTM3NSwieSI6ODd9LHsieCI6ODY3LjM1OTM3NSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(77.5078125, 35)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(273.4609375, 35)"><rect class="basic label-container" style="" x="-76.4453125" y="-27" width="152.890625" height="54"/><g class="label" style="" transform="translate(-46.4453125, -12)"><rect/><foreignObject width="92.890625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Encoder</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(492.84375, 35)"><rect class="basic label-container" style="" x="-92.9375" y="-27" width="185.875" height="54"/><g class="label" style="" transform="translate(-62.9375, -12)"><rect/><foreignObject width="125.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query Embedding</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(726.5703125, 87)"><rect class="basic label-container" style="" x="-90.7890625" y="-27" width="181.578125" height="54"/><g class="label" style="" transform="translate(-60.7890625, -12)"><rect/><foreignObject width="121.578125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Search Algorithm</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-6" transform="translate(492.84375, 139)"><rect class="basic label-container" style="" x="-49.4296875" y="-27" width="98.859375" height="54"/><g class="label" style="" transform="translate(-19.4296875, -12)"><rect/><foreignObject width="38.859375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Index</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(956.0234375, 87)"><rect class="basic label-container" style="" x="-88.6640625" y="-39" width="177.328125" height="78"/><g class="label" style="" transform="translate(-58.6640625, -24)"><rect/><foreignObject width="117.328125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Top-K<br />Relevant Chunks</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 836.078px; background-color: transparent;" viewBox="0 0 836.078125 302" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M418.039,62L418.039,66.167C418.039,70.333,418.039,78.667,418.039,86.333C418.039,94,418.039,101,418.039,104.5L418.039,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NDE4LjAzOTA2MjUsInkiOjYyfSx7IngiOjQxOC4wMzkwNjI1LCJ5Ijo4N30seyJ4Ijo0MTguMDM5MDYyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M315.039,169.941L274.201,177.451C233.362,184.961,151.685,199.98,110.846,210.99C70.008,222,70.008,229,70.008,232.5L70.008,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzE1LjAzOTA2MjUsInkiOjE2OS45NDA4Mjc4NzEwNjA0M30seyJ4Ijo3MC4wMDc4MTI1LCJ5IjoyMTV9LHsieCI6NzAuMDA3ODEyNSwieSI6MjQwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M315.039,188.882L303.203,193.235C291.367,197.588,267.695,206.294,255.859,214.147C244.023,222,244.023,229,244.023,232.5L244.023,236" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6MzE1LjAzOTA2MjUsInkiOjE4OC44ODE2NTU3NDIxMjA4N30seyJ4IjoyNDQuMDIzNDM3NSwieSI6MjE1fSx7IngiOjI0NC4wMjM0Mzc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M418.039,190L418.039,194.167C418.039,198.333,418.039,206.667,418.039,214.333C418.039,222,418.039,229,418.039,232.5L418.039,236" id="L_B_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_E_0" data-points="W3sieCI6NDE4LjAzOTA2MjUsInkiOjE5MH0seyJ4Ijo0MTguMDM5MDYyNSwieSI6MjE1fSx7IngiOjQxOC4wMzkwNjI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M521.039,188.882L532.875,193.235C544.711,197.588,568.383,206.294,580.219,214.147C592.055,222,592.055,229,592.055,232.5L592.055,236" id="L_B_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_F_0" data-points="W3sieCI6NTIxLjAzOTA2MjUsInkiOjE4OC44ODE2NTU3NDIxMjA4N30seyJ4Ijo1OTIuMDU0Njg3NSwieSI6MjE1fSx7IngiOjU5Mi4wNTQ2ODc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M521.039,169.941L561.878,177.451C602.716,184.961,684.393,199.98,725.232,210.99C766.07,222,766.07,229,766.07,232.5L766.07,236" id="L_B_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_G_0" data-points="W3sieCI6NTIxLjAzOTA2MjUsInkiOjE2OS45NDA4Mjc4NzEwNjA0M30seyJ4Ijo3NjYuMDcwMzEyNSwieSI6MjE1fSx7IngiOjc2Ni4wNzAzMTI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(418.0390625, 35)"><rect class="basic label-container" style="" x="-82.1875" y="-27" width="164.375" height="54"/><g class="label" style="" transform="translate(-52.1875, -12)"><rect/><foreignObject width="104.375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>All Data Points</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(418.0390625, 151)"><rect class="basic label-container" style="" x="-103" y="-39" width="206" height="78"/><g class="label" style="" transform="translate(-73, -24)"><rect/><foreignObject width="146" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Clustering Algorithm<br />K-Means</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(70.0078125, 267)"><rect class="basic label-container" style="" x="-62.0078125" y="-27" width="124.015625" height="54"/><g class="label" style="" transform="translate(-32.0078125, -12)"><rect/><foreignObject width="64.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(244.0234375, 267)"><rect class="basic label-container" style="" x="-62.0078125" y="-27" width="124.015625" height="54"/><g class="label" style="" transform="translate(-32.0078125, -12)"><rect/><foreignObject width="64.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(418.0390625, 267)"><rect class="basic label-container" style="" x="-62.0078125" y="-27" width="124.015625" height="54"/><g class="label" style="" transform="translate(-32.0078125, -12)"><rect/><foreignObject width="64.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 3</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(592.0546875, 267)"><rect class="basic label-container" style="" x="-62.0078125" y="-27" width="124.015625" height="54"/><g class="label" style="" transform="translate(-32.0078125, -12)"><rect/><foreignObject width="64.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 4</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(766.0703125, 267)"><rect class="basic label-container" style="" x="-62.0078125" y="-27" width="124.015625" height="54"/><g class="label" style="" transform="translate(-32.0078125, -12)"><rect/><foreignObject width="64.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 5</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Cluster Representation**:
* Each cluster has centroid (representative point)
* All points assigned to nearest centroid

**Search Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 174.906px; background-color: transparent;" viewBox="0 0 174.90625 582" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M87.453,62L87.453,66.167C87.453,70.333,87.453,78.667,87.453,86.333C87.453,94,87.453,101,87.453,104.5L87.453,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6ODcuNDUzMTI1LCJ5Ijo2Mn0seyJ4Ijo4Ny40NTMxMjUsInkiOjg3fSx7IngiOjg3LjQ1MzEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M87.453,190L87.453,194.167C87.453,198.333,87.453,206.667,87.453,214.333C87.453,222,87.453,229,87.453,232.5L87.453,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6ODcuNDUzMTI1LCJ5IjoxOTB9LHsieCI6ODcuNDUzMTI1LCJ5IjoyMTV9LHsieCI6ODcuNDUzMTI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M87.453,318L87.453,322.167C87.453,326.333,87.453,334.667,87.453,342.333C87.453,350,87.453,357,87.453,360.5L87.453,364" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6ODcuNDUzMTI1LCJ5IjozMTh9LHsieCI6ODcuNDUzMTI1LCJ5IjozNDN9LHsieCI6ODcuNDUzMTI1LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M87.453,446L87.453,450.167C87.453,454.333,87.453,462.667,87.453,470.333C87.453,478,87.453,485,87.453,488.5L87.453,492" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6ODcuNDUzMTI1LCJ5Ijo0NDZ9LHsieCI6ODcuNDUzMTI1LCJ5Ijo0NzF9LHsieCI6ODcuNDUzMTI1LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(87.453125, 35)"><rect class="basic label-container" style="" x="-71.8125" y="-27" width="143.625" height="54"/><g class="label" style="" transform="translate(-41.8125, -12)"><rect/><foreignObject width="83.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query Point</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(87.453125, 151)"><rect class="basic label-container" style="" x="-75.2734375" y="-39" width="150.546875" height="78"/><g class="label" style="" transform="translate(-45.2734375, -24)"><rect/><foreignObject width="90.546875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Find Nearest<br />Centroid</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(87.453125, 279)"><rect class="basic label-container" style="" x="-62.0078125" y="-39" width="124.015625" height="78"/><g class="label" style="" transform="translate(-32.0078125, -24)"><rect/><foreignObject width="64.015625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Cluster 3<br />Selected</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(87.453125, 407)"><rect class="basic label-container" style="" x="-79.453125" y="-39" width="158.90625" height="78"/><g class="label" style="" transform="translate(-49.453125, -24)"><rect/><foreignObject width="98.90625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Search Within<br />Cluster Only</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(87.453125, 535)"><rect class="basic label-container" style="" x="-75.9296875" y="-39" width="151.859375" height="78"/><g class="label" style="" transform="translate(-45.9296875, -24)"><rect/><foreignObject width="91.859375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Return Top-K<br />from Cluster</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 648.125px; background-color: transparent;" viewBox="0 0 648.125 278" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M236.094,61.824L222.333,66.02C208.573,70.216,181.052,78.608,167.292,86.304C153.531,94,153.531,101,153.531,104.5L153.531,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjM2LjA5Mzc1LCJ5Ijo2MS44MjQyNjI0MTUyNDY0N30seyJ4IjoxNTMuNTMxMjUsInkiOjg3fSx7IngiOjE1My41MzEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M412.031,61.824L425.792,66.02C439.552,70.216,467.073,78.608,480.833,86.304C494.594,94,494.594,101,494.594,104.5L494.594,108" id="L_A_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_C_0" data-points="W3sieCI6NDEyLjAzMTI1LCJ5Ijo2MS44MjQyNjI0MTUyNDY0N30seyJ4Ijo0OTQuNTkzNzUsInkiOjg3fSx7IngiOjQ5NC41OTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.259,166L102.427,170.167C95.594,174.333,81.93,182.667,75.098,190.333C68.266,198,68.266,205,68.266,208.5L68.266,212" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6MTA5LjI1ODcxMzk0MjMwNzcsInkiOjE2Nn0seyJ4Ijo2OC4yNjU2MjUsInkiOjE5MX0seyJ4Ijo2OC4yNjU2MjUsInkiOjIxNn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M197.804,166L204.636,170.167C211.468,174.333,225.133,182.667,231.965,190.333C238.797,198,238.797,205,238.797,208.5L238.797,212" id="L_B_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_E_0" data-points="W3sieCI6MTk3LjgwMzc4NjA1NzY5MjMyLCJ5IjoxNjZ9LHsieCI6MjM4Ljc5Njg3NSwieSI6MTkxfSx7IngiOjIzOC43OTY4NzUsInkiOjIxNn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M450.321,166L443.489,170.167C436.657,174.333,422.992,182.667,416.16,190.333C409.328,198,409.328,205,409.328,208.5L409.328,212" id="L_C_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_F_0" data-points="W3sieCI6NDUwLjMyMTIxMzk0MjMwNzcsInkiOjE2Nn0seyJ4Ijo0MDkuMzI4MTI1LCJ5IjoxOTF9LHsieCI6NDA5LjMyODEyNSwieSI6MjE2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M538.866,166L545.698,170.167C552.531,174.333,566.195,182.667,573.027,190.333C579.859,198,579.859,205,579.859,208.5L579.859,212" id="L_C_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_G_0" data-points="W3sieCI6NTM4Ljg2NjI4NjA1NzY5MjMsInkiOjE2Nn0seyJ4Ijo1NzkuODU5Mzc1LCJ5IjoxOTF9LHsieCI6NTc5Ljg1OTM3NSwieSI6MjE2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(324.0625, 35)"><rect class="basic label-container" style="" x="-87.96875" y="-27" width="175.9375" height="54"/><g class="label" style="" transform="translate(-57.96875, -12)"><rect/><foreignObject width="115.9375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Root: Full Space</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(153.53125, 139)"><rect class="basic label-container" style="" x="-79.8515625" y="-27" width="159.703125" height="54"/><g class="label" style="" transform="translate(-49.8515625, -12)"><rect/><foreignObject width="99.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Left Subspace</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(494.59375, 139)"><rect class="basic label-container" style="" x="-83.8046875" y="-27" width="167.609375" height="54"/><g class="label" style="" transform="translate(-53.8046875, -12)"><rect/><foreignObject width="107.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Right Subspace</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(68.265625, 243)"><rect class="basic label-container" style="" x="-60.265625" y="-27" width="120.53125" height="54"/><g class="label" style="" transform="translate(-30.265625, -12)"><rect/><foreignObject width="60.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Region 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(238.796875, 243)"><rect class="basic label-container" style="" x="-60.265625" y="-27" width="120.53125" height="54"/><g class="label" style="" transform="translate(-30.265625, -12)"><rect/><foreignObject width="60.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Region 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(409.328125, 243)"><rect class="basic label-container" style="" x="-60.265625" y="-27" width="120.53125" height="54"/><g class="label" style="" transform="translate(-30.265625, -12)"><rect/><foreignObject width="60.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Region 3</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(579.859375, 243)"><rect class="basic label-container" style="" x="-60.265625" y="-27" width="120.53125" height="54"/><g class="label" style="" transform="translate(-30.265625, -12)"><rect/><foreignObject width="60.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Region 4</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 853.375px; background-color: transparent;" viewBox="0 0 853.375 174" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M198.594,35L202.76,35C206.927,35,215.26,35,225.763,38.822C236.265,42.645,248.937,50.289,255.273,54.111L261.609,57.934" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTk4LjU5Mzc1LCJ5IjozNX0seyJ4IjoyMjMuNTkzNzUsInkiOjM1fSx7IngiOjI2NS4wMzM4MDQwODY1Mzg0NSwieSI6NjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M172.805,139L181.27,139C189.734,139,206.664,139,221.465,135.178C236.265,131.355,248.937,123.711,255.273,119.889L261.609,116.066" id="L_C_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_B_0" data-points="W3sieCI6MTcyLjgwNDY4NzUsInkiOjEzOX0seyJ4IjoyMjMuNTkzNzUsInkiOjEzOX0seyJ4IjoyNjUuMDMzODA0MDg2NTM4NDUsInkiOjExNH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M370.984,87L375.151,87C379.318,87,387.651,87,395.318,87C402.984,87,409.984,87,413.484,87L416.984,87" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6MzcwLjk4NDM3NSwieSI6ODd9LHsieCI6Mzk1Ljk4NDM3NSwieSI6ODd9LHsieCI6NDIwLjk4NDM3NSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M532.5,87L536.667,87C540.833,87,549.167,87,556.833,87C564.5,87,571.5,87,575,87L578.5,87" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NTMyLjUsInkiOjg3fSx7IngiOjU1Ny41LCJ5Ijo4N30seyJ4Ijo1ODIuNSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M670.063,87L674.229,87C678.396,87,686.729,87,694.396,87C702.063,87,709.063,87,712.563,87L716.063,87" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6NjcwLjA2MjUsInkiOjg3fSx7IngiOjY5NS4wNjI1LCJ5Ijo4N30seyJ4Ijo3MjAuMDYyNSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(103.296875, 35)"><rect class="basic label-container" style="" x="-95.296875" y="-27" width="190.59375" height="54"/><g class="label" style="" transform="translate(-65.296875, -12)"><rect/><foreignObject width="130.59375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieved Content</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(309.7890625, 87)"><rect class="basic label-container" style="" x="-61.1953125" y="-27" width="122.390625" height="54"/><g class="label" style="" transform="translate(-31.1953125, -12)"><rect/><foreignObject width="62.390625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Combine</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-2" transform="translate(103.296875, 139)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(476.7421875, 87)"><rect class="basic label-container" style="" x="-55.7578125" y="-27" width="111.515625" height="54"/><g class="label" style="" transform="translate(-25.7578125, -12)"><rect/><foreignObject width="51.515625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(626.28125, 87)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(782.71875, 87)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Prompt Structure**:

```
Retrieved Context:
- "Our refund policy allows returns within 30 days..."
- "Items must be unused and in original packaging..."
- "Contact customer support at 555-0123..."

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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1004.58px; background-color: transparent;" viewBox="0 0 1004.578125 174" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M198.594,35L202.76,35C206.927,35,215.26,35,228.828,38.911C242.395,42.821,261.196,50.642,270.596,54.553L279.997,58.464" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTk4LjU5Mzc1LCJ5IjozNX0seyJ4IjoyMjMuNTkzNzUsInkiOjM1fSx7IngiOjI4My42ODk5MDM4NDYxNTM4LCJ5Ijo2MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M172.805,139L181.27,139C189.734,139,206.664,139,224.529,135.089C242.395,131.179,261.196,123.358,270.596,119.447L279.997,115.536" id="L_C_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_B_0" data-points="W3sieCI6MTcyLjgwNDY4NzUsInkiOjEzOX0seyJ4IjoyMjMuNTkzNzUsInkiOjEzOX0seyJ4IjoyODMuNjg5OTAzODQ2MTUzOCwieSI6MTE0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448.594,87L452.76,87C456.927,87,465.26,87,472.927,87C480.594,87,487.594,87,491.094,87L494.594,87" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6NDQ4LjU5Mzc1LCJ5Ijo4N30seyJ4Ijo0NzMuNTkzNzUsInkiOjg3fSx7IngiOjQ5OC41OTM3NSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M683.703,87L687.87,87C692.036,87,700.37,87,708.036,87C715.703,87,722.703,87,726.203,87L729.703,87" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NjgzLjcwMzEyNSwieSI6ODd9LHsieCI6NzA4LjcwMzEyNSwieSI6ODd9LHsieCI6NzMzLjcwMzEyNSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M821.266,87L825.432,87C829.599,87,837.932,87,845.599,87C853.266,87,860.266,87,863.766,87L867.266,87" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6ODIxLjI2NTYyNSwieSI6ODd9LHsieCI6ODQ2LjI2NTYyNSwieSI6ODd9LHsieCI6ODcxLjI2NTYyNSwieSI6ODd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(103.296875, 35)"><rect class="basic label-container" style="" x="-95.296875" y="-27" width="190.59375" height="54"/><g class="label" style="" transform="translate(-65.296875, -12)"><rect/><foreignObject width="130.59375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieved Content</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(348.59375, 87)"><rect class="basic label-container" style="" x="-100" y="-27" width="200" height="54"/><g class="label" style="" transform="translate(-70, -12)"><rect/><foreignObject width="140" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt Engineering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-2" transform="translate(103.296875, 139)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(591.1484375, 87)"><rect class="basic label-container" style="" x="-92.5546875" y="-27" width="185.109375" height="54"/><g class="label" style="" transform="translate(-62.5546875, -12)"><rect/><foreignObject width="125.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Enhanced Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(777.484375, 87)"><rect class="basic label-container" style="" x="-43.78125" y="-27" width="87.5625" height="54"/><g class="label" style="" transform="translate(-13.78125, -12)"><rect/><foreignObject width="27.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(933.921875, 87)"><rect class="basic label-container" style="" x="-62.65625" y="-27" width="125.3125" height="54"/><g class="label" style="" transform="translate(-32.65625, -12)"><rect/><foreignObject width="65.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 241.141px; background-color: transparent;" viewBox="0 0 241.140625 558" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M120.57,62L120.57,66.167C120.57,70.333,120.57,78.667,120.57,86.333C120.57,94,120.57,101,120.57,104.5L120.57,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTIwLjU3MDMxMjUsInkiOjYyfSx7IngiOjEyMC41NzAzMTI1LCJ5Ijo4N30seyJ4IjoxMjAuNTcwMzEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M120.57,190L120.57,194.167C120.57,198.333,120.57,206.667,120.57,214.333C120.57,222,120.57,229,120.57,232.5L120.57,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTIwLjU3MDMxMjUsInkiOjE5MH0seyJ4IjoxMjAuNTcwMzEyNSwieSI6MjE1fSx7IngiOjEyMC41NzAzMTI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M120.57,294L120.57,298.167C120.57,302.333,120.57,310.667,120.57,318.333C120.57,326,120.57,333,120.57,336.5L120.57,340" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTIwLjU3MDMxMjUsInkiOjI5NH0seyJ4IjoxMjAuNTcwMzEyNSwieSI6MzE5fSx7IngiOjEyMC41NzAzMTI1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M120.57,422L120.57,426.167C120.57,430.333,120.57,438.667,120.57,446.333C120.57,454,120.57,461,120.57,464.5L120.57,468" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTIwLjU3MDMxMjUsInkiOjQyMn0seyJ4IjoxMjAuNTcwMzEyNSwieSI6NDQ3fSx7IngiOjEyMC41NzAzMTI1LCJ5Ijo0NzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(120.5703125, 35)"><rect class="basic label-container" style="" x="-103.5546875" y="-27" width="207.109375" height="54"/><g class="label" style="" transform="translate(-73.5546875, -12)"><rect/><foreignObject width="147.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Create Training Data</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(120.5703125, 151)"><rect class="basic label-container" style="" x="-110.328125" y="-39" width="220.65625" height="78"/><g class="label" style="" transform="translate(-80.328125, -24)"><rect/><foreignObject width="160.65625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Label Documents<br />as Relevant/Irrelevant</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(120.5703125, 267)"><rect class="basic label-container" style="" x="-81.0546875" y="-27" width="162.109375" height="54"/><g class="label" style="" transform="translate(-51.0546875, -12)"><rect/><foreignObject width="102.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Fine-Tune LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(120.5703125, 383)"><rect class="basic label-container" style="" x="-110.078125" y="-39" width="220.15625" height="78"/><g class="label" style="" transform="translate(-80.078125, -24)"><rect/><foreignObject width="160.15625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Model Learns to<br />Identify Relevant Docs</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(120.5703125, 511)"><rect class="basic label-container" style="" x="-112.5703125" y="-39" width="225.140625" height="78"/><g class="label" style="" transform="translate(-82.5703125, -24)"><rect/><foreignObject width="165.140625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Better Generation<br />Despite Noisy Retrieval</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 951.043px; background-color: transparent;" viewBox="0 0 951.04296875 486" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M339.485,62L334.279,66.167C329.073,70.333,318.662,78.667,313.456,86.333C308.25,94,308.25,101,308.25,104.5L308.25,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MzM5LjQ4NDk3NTk2MTUzODQ1LCJ5Ijo2Mn0seyJ4IjozMDguMjUsInkiOjg3fSx7IngiOjMwOC4yNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M308.25,166L308.25,170.167C308.25,174.333,308.25,182.667,308.25,190.333C308.25,198,308.25,205,308.25,208.5L308.25,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzA4LjI1LCJ5IjoxNjZ9LHsieCI6MzA4LjI1LCJ5IjoxOTF9LHsieCI6MzA4LjI1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M308.25,270L308.25,274.167C308.25,278.333,308.25,286.667,312.935,294.583C317.621,302.5,326.991,310,331.677,313.75L336.362,317.5" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MzA4LjI1LCJ5IjoyNzB9LHsieCI6MzA4LjI1LCJ5IjoyOTV9LHsieCI6MzM5LjQ4NDk3NTk2MTUzODQ1LCJ5IjozMjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M406.953,62L412.158,66.167C417.364,70.333,427.776,78.667,432.982,91.5C438.188,104.333,438.188,121.667,438.188,139C438.188,156.333,438.188,173.667,438.188,191C438.188,208.333,438.188,225.667,438.188,243C438.188,260.333,438.188,277.667,433.502,290.083C428.817,302.5,419.446,310,414.761,313.75L410.075,317.5" id="L_A_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_D_0" data-points="W3sieCI6NDA2Ljk1MjUyNDAzODQ2MTU1LCJ5Ijo2Mn0seyJ4Ijo0MzguMTg3NSwieSI6ODd9LHsieCI6NDM4LjE4NzUsInkiOjEzOX0seyJ4Ijo0MzguMTg3NSwieSI6MTkxfSx7IngiOjQzOC4xODc1LCJ5IjoyNDN9LHsieCI6NDM4LjE4NzUsInkiOjI5NX0seyJ4Ijo0MDYuOTUyNTI0MDM4NDYxNTUsInkiOjMyMH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M373.219,374L373.219,378.167C373.219,382.333,373.219,390.667,373.219,398.333C373.219,406,373.219,413,373.219,416.5L373.219,420" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MzczLjIxODc1LCJ5IjozNzR9LHsieCI6MzczLjIxODc1LCJ5IjozOTl9LHsieCI6MzczLjIxODc1LCJ5Ijo0MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M104.813,62L104.813,66.167C104.813,70.333,104.813,78.667,104.813,86.333C104.813,94,104.813,101,104.813,104.5L104.813,108" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MTA0LjgxMjUsInkiOjYyfSx7IngiOjEwNC44MTI1LCJ5Ijo4N30seyJ4IjoxMDQuODEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M594.246,62L594.246,66.167C594.246,70.333,594.246,78.667,594.246,86.333C594.246,94,594.246,101,594.246,104.5L594.246,108" id="L_H_I_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_H_I_0" data-points="W3sieCI6NTk0LjI0NjA5Mzc1LCJ5Ijo2Mn0seyJ4Ijo1OTQuMjQ2MDkzNzUsInkiOjg3fSx7IngiOjU5NC4yNDYwOTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M842.871,62L842.871,66.167C842.871,70.333,842.871,78.667,842.871,86.333C842.871,94,842.871,101,842.871,104.5L842.871,108" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6ODQyLjg3MTA5Mzc1LCJ5Ijo2Mn0seyJ4Ijo4NDIuODcxMDkzNzUsInkiOjg3fSx7IngiOjg0Mi44NzEwOTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_H_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(373.21875, 35)"><rect class="basic label-container" style="" x="-51.1953125" y="-27" width="102.390625" height="54"/><g class="label" style="" transform="translate(-21.1953125, -12)"><rect/><foreignObject width="42.390625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(308.25, 139)"><rect class="basic label-container" style="" x="-62.1015625" y="-27" width="124.203125" height="54"/><g class="label" style="" transform="translate(-32.1015625, -12)"><rect/><foreignObject width="64.203125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieval</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(308.25, 243)"><rect class="basic label-container" style="" x="-94.9375" y="-27" width="189.875" height="54"/><g class="label" style="" transform="translate(-64.9375, -12)"><rect/><foreignObject width="129.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieved Context</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(373.21875, 347)"><rect class="basic label-container" style="" x="-69.9375" y="-27" width="139.875" height="54"/><g class="label" style="" transform="translate(-39.9375, -12)"><rect/><foreignObject width="79.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generation</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-9" transform="translate(373.21875, 451)"><rect class="basic label-container" style="" x="-82.484375" y="-27" width="164.96875" height="54"/><g class="label" style="" transform="translate(-52.484375, -12)"><rect/><foreignObject width="104.96875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Final Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-10" transform="translate(104.8125, 35)"><rect class="basic label-container" style="" x="-96.8125" y="-27" width="193.625" height="54"/><g class="label" style="" transform="translate(-66.8125, -12)"><rect/><foreignObject width="133.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Context Relevance</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(104.8125, 139)"><rect class="basic label-container" style="" x="-91.3359375" y="-27" width="182.671875" height="54"/><g class="label" style="" transform="translate(-61.3359375, -12)"><rect/><foreignObject width="122.671875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query vs Context</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-12" transform="translate(594.24609375, 35)"><rect class="basic label-container" style="" x="-73.125" y="-27" width="146.25" height="54"/><g class="label" style="" transform="translate(-43.125, -12)"><rect/><foreignObject width="86.25" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Faithfulness</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-13" transform="translate(594.24609375, 139)"><rect class="basic label-container" style="" x="-102.796875" y="-27" width="205.59375" height="54"/><g class="label" style="" transform="translate(-72.796875, -12)"><rect/><foreignObject width="145.59375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response vs Context</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-14" transform="translate(842.87109375, 35)"><rect class="basic label-container" style="" x="-100.171875" y="-27" width="200.34375" height="54"/><g class="label" style="" transform="translate(-70.171875, -12)"><rect/><foreignObject width="140.34375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answer Correctness</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-15" transform="translate(842.87109375, 139)"><rect class="basic label-container" style="" x="-95.828125" y="-27" width="191.65625" height="54"/><g class="label" style="" transform="translate(-65.828125, -12)"><rect/><foreignObject width="131.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response vs Query</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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
Evaluation: Entailed → Faithful ✓

Context: "Returns accepted within 30 days."
Response: "You can return items within 60 days."
Evaluation: Contradicted → Not Faithful ✗
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


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 633.562px; background-color: transparent;" viewBox="-35 0 633.5625 1606" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M422.48,62L422.48,66.167C422.48,70.333,422.48,78.667,422.48,86.333C422.48,94,422.48,101,422.48,104.5L422.48,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NDIyLjQ4MDQ2ODc1LCJ5Ijo2Mn0seyJ4Ijo0MjIuNDgwNDY4NzUsInkiOjg3fSx7IngiOjQyMi40ODA0Njg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M387.103,190L381.509,196.167C375.915,202.333,364.727,214.667,359.133,226.333C353.539,238,353.539,249,353.539,254.5L353.539,260" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6Mzg3LjEwMjY0MTg1ODU1MjYsInkiOjE5MH0seyJ4IjozNTMuNTM5MDYyNSwieSI6MjI3fSx7IngiOjM1My41MzkwNjI1LCJ5IjoyNjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M457.858,190L463.452,196.167C469.046,202.333,480.234,214.667,485.828,233.5C491.422,252.333,491.422,277.667,491.422,301C491.422,324.333,491.422,345.667,491.422,365C491.422,384.333,491.422,401.667,491.422,419C491.422,436.333,491.422,453.667,491.422,473C491.422,492.333,491.422,513.667,491.422,535C491.422,556.333,491.422,577.667,491.422,599C491.422,620.333,491.422,641.667,491.422,663C491.422,684.333,491.422,705.667,491.422,727C491.422,748.333,491.422,769.667,491.422,793C491.422,816.333,491.422,841.667,491.422,865C491.422,888.333,491.422,909.667,491.422,929C491.422,948.333,491.422,965.667,491.422,985C491.422,1004.333,491.422,1025.667,491.422,1047C491.422,1068.333,491.422,1089.667,491.422,1113C491.422,1136.333,491.422,1161.667,491.422,1187C491.422,1212.333,491.422,1237.667,491.422,1259C491.422,1280.333,491.422,1297.667,491.422,1315C491.422,1332.333,491.422,1349.667,491.422,1369C491.422,1388.333,491.422,1409.667,491.422,1433C491.422,1456.333,491.422,1481.667,491.422,1499.833C491.422,1518,491.422,1529,491.422,1534.5L491.422,1540" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6NDU3Ljg1ODI5NTY0MTQ0NzQsInkiOjE5MH0seyJ4Ijo0OTEuNDIxODc1LCJ5IjoyMjd9LHsieCI6NDkxLjQyMTg3NSwieSI6MzAzfSx7IngiOjQ5MS40MjE4NzUsInkiOjM2N30seyJ4Ijo0OTEuNDIxODc1LCJ5Ijo0MTl9LHsieCI6NDkxLjQyMTg3NSwieSI6NDcxfSx7IngiOjQ5MS40MjE4NzUsInkiOjUzNX0seyJ4Ijo0OTEuNDIxODc1LCJ5Ijo1OTl9LHsieCI6NDkxLjQyMTg3NSwieSI6NjYzfSx7IngiOjQ5MS40MjE4NzUsInkiOjcyN30seyJ4Ijo0OTEuNDIxODc1LCJ5Ijo3OTF9LHsieCI6NDkxLjQyMTg3NSwieSI6ODY3fSx7IngiOjQ5MS40MjE4NzUsInkiOjkzMX0seyJ4Ijo0OTEuNDIxODc1LCJ5Ijo5ODN9LHsieCI6NDkxLjQyMTg3NSwieSI6MTA0N30seyJ4Ijo0OTEuNDIxODc1LCJ5IjoxMTExfSx7IngiOjQ5MS40MjE4NzUsInkiOjExODd9LHsieCI6NDkxLjQyMTg3NSwieSI6MTI2M30seyJ4Ijo0OTEuNDIxODc1LCJ5IjoxMzE1fSx7IngiOjQ5MS40MjE4NzUsInkiOjEzNjd9LHsieCI6NDkxLjQyMTg3NSwieSI6MTQzMX0seyJ4Ijo0OTEuNDIxODc1LCJ5IjoxNTA3fSx7IngiOjQ5MS40MjE4NzUsInkiOjE1NDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M353.539,342L353.539,346.167C353.539,350.333,353.539,358.667,353.539,366.333C353.539,374,353.539,381,353.539,384.5L353.539,388" id="L_C_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_E_0" data-points="W3sieCI6MzUzLjUzOTA2MjUsInkiOjM0Mn0seyJ4IjozNTMuNTM5MDYyNSwieSI6MzY3fSx7IngiOjM1My41MzkwNjI1LCJ5IjozOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M353.539,446L353.539,450.167C353.539,454.333,353.539,462.667,353.539,472.333C353.539,482,353.539,493,353.539,498.5L353.539,504" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MzUzLjUzOTA2MjUsInkiOjQ0Nn0seyJ4IjozNTMuNTM5MDYyNSwieSI6NDcxfSx7IngiOjM1My41MzkwNjI1LCJ5Ijo1MDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M113.328,62L113.328,66.167C113.328,70.333,113.328,78.667,113.328,88.333C113.328,98,113.328,109,113.328,114.5L113.328,120" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6MTEzLjMyODEyNSwieSI6NjJ9LHsieCI6MTEzLjMyODEyNSwieSI6ODd9LHsieCI6MTEzLjMyODEyNSwieSI6MTI0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M113.328,178L113.328,186.167C113.328,194.333,113.328,210.667,113.328,226.333C113.328,242,113.328,257,113.328,264.5L113.328,272" id="L_H_I_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_H_I_0" data-points="W3sieCI6MTEzLjMyODEyNSwieSI6MTc4fSx7IngiOjExMy4zMjgxMjUsInkiOjIyN30seyJ4IjoxMTMuMzI4MTI1LCJ5IjoyNzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M113.328,330L113.328,336.167C113.328,342.333,113.328,354.667,113.328,364.333C113.328,374,113.328,381,113.328,384.5L113.328,388" id="L_I_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_I_J_0" data-points="W3sieCI6MTEzLjMyODEyNSwieSI6MzMwfSx7IngiOjExMy4zMjgxMjUsInkiOjM2N30seyJ4IjoxMTMuMzI4MTI1LCJ5IjozOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M113.328,446L113.328,450.167C113.328,454.333,113.328,462.667,113.328,470.333C113.328,478,113.328,485,113.328,488.5L113.328,492" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6MTEzLjMyODEyNSwieSI6NDQ2fSx7IngiOjExMy4zMjgxMjUsInkiOjQ3MX0seyJ4IjoxMTMuMzI4MTI1LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M353.539,562L353.539,568.167C353.539,574.333,353.539,586.667,353.539,596.333C353.539,606,353.539,613,353.539,616.5L353.539,620" id="L_F_L_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_L_0" data-points="W3sieCI6MzUzLjUzOTA2MjUsInkiOjU2Mn0seyJ4IjozNTMuNTM5MDYyNSwieSI6NTk5fSx7IngiOjM1My41MzkwNjI1LCJ5Ijo2MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M113.328,574L113.328,578.167C113.328,582.333,113.328,590.667,137.392,601.245C161.456,611.823,209.585,624.646,233.649,631.057L257.713,637.469" id="L_K_L_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_K_L_0" data-points="W3sieCI6MTEzLjMyODEyNSwieSI6NTc0fSx7IngiOjExMy4zMjgxMjUsInkiOjU5OX0seyJ4IjoyNjEuNTc4MTI1LCJ5Ijo2MzguNDk4NjE3NzUxMzI1NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M353.539,702L353.539,706.167C353.539,710.333,353.539,718.667,353.539,726.333C353.539,734,353.539,741,353.539,744.5L353.539,748" id="L_L_M_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_L_M_0" data-points="W3sieCI6MzUzLjUzOTA2MjUsInkiOjcwMn0seyJ4IjozNTMuNTM5MDYyNSwieSI6NzI3fSx7IngiOjM1My41MzkwNjI1LCJ5Ijo3NTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M353.539,830L353.539,836.167C353.539,842.333,353.539,854.667,347.271,866.551C341.003,878.435,328.467,889.87,322.199,895.587L315.931,901.304" id="L_M_N_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_M_N_0" data-points="W3sieCI6MzUzLjUzOTA2MjUsInkiOjgzMH0seyJ4IjozNTMuNTM5MDYyNSwieSI6ODY3fSx7IngiOjMxMi45NzU0NjM4NjcxODc1LCJ5Ijo5MDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M264.422,317.988L215.852,326.157C167.281,334.325,70.141,350.663,21.57,367.498C-27,384.333,-27,401.667,-27,419C-27,436.333,-27,453.667,-27,473C-27,492.333,-27,513.667,-27,535C-27,556.333,-27,577.667,-27,599C-27,620.333,-27,641.667,-27,663C-27,684.333,-27,705.667,-27,727C-27,748.333,-27,769.667,-27,793C-27,816.333,-27,841.667,7.41,861.429C41.819,881.191,110.638,895.381,145.048,902.477L179.457,909.572" id="L_C_N_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_N_0" data-points="W3sieCI6MjY0LjQyMTg3NSwieSI6MzE3Ljk4Nzk0ODgzOTAyMzZ9LHsieCI6LTI3LCJ5IjozNjd9LHsieCI6LTI3LCJ5Ijo0MTl9LHsieCI6LTI3LCJ5Ijo0NzF9LHsieCI6LTI3LCJ5Ijo1MzV9LHsieCI6LTI3LCJ5Ijo1OTl9LHsieCI6LTI3LCJ5Ijo2NjN9LHsieCI6LTI3LCJ5Ijo3Mjd9LHsieCI6LTI3LCJ5Ijo3OTF9LHsieCI6LTI3LCJ5Ijo4Njd9LHsieCI6MTgzLjM3NSwieSI6OTEwLjM3OTc4MjUyMTE0Mzd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M283.375,958L283.375,962.167C283.375,966.333,283.375,974.667,283.375,982.333C283.375,990,283.375,997,283.375,1000.5L283.375,1004" id="L_N_O_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_N_O_0" data-points="W3sieCI6MjgzLjM3NSwieSI6OTU4fSx7IngiOjI4My4zNzUsInkiOjk4M30seyJ4IjoyODMuMzc1LCJ5IjoxMDA4fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M283.375,1086L283.375,1090.167C283.375,1094.333,283.375,1102.667,283.375,1110.333C283.375,1118,283.375,1125,283.375,1128.5L283.375,1132" id="L_O_P_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_O_P_0" data-points="W3sieCI6MjgzLjM3NSwieSI6MTA4Nn0seyJ4IjoyODMuMzc1LCJ5IjoxMTExfSx7IngiOjI4My4zNzUsInkiOjExMzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M283.375,1238L283.375,1242.167C283.375,1246.333,283.375,1254.667,283.375,1262.333C283.375,1270,283.375,1277,283.375,1280.5L283.375,1284" id="L_P_Q_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_P_Q_0" data-points="W3sieCI6MjgzLjM3NSwieSI6MTIzOH0seyJ4IjoyODMuMzc1LCJ5IjoxMjYzfSx7IngiOjI4My4zNzUsInkiOjEyODh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M283.375,1342L283.375,1346.167C283.375,1350.333,283.375,1358.667,283.375,1366.333C283.375,1374,283.375,1381,283.375,1384.5L283.375,1388" id="L_Q_R_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Q_R_0" data-points="W3sieCI6MjgzLjM3NSwieSI6MTM0Mn0seyJ4IjoyODMuMzc1LCJ5IjoxMzY3fSx7IngiOjI4My4zNzUsInkiOjEzOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M259.042,1470L255.195,1476.167C251.347,1482.333,243.652,1494.667,239.805,1506.333C235.957,1518,235.957,1529,235.957,1534.5L235.957,1540" id="L_R_S_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R_S_0" data-points="W3sieCI6MjU5LjA0MjA5NDk4MzU1MjYsInkiOjE0NzB9LHsieCI6MjM1Ljk1NzAzMTI1LCJ5IjoxNTA3fSx7IngiOjIzNS45NTcwMzEyNSwieSI6MTU0NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M340.835,1470L349.92,1476.167C359.006,1482.333,377.177,1494.667,394.964,1506.63C412.752,1518.594,430.157,1530.188,438.859,1535.985L447.562,1541.782" id="L_R_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R_D_0" data-points="W3sieCI6MzQwLjgzNDY1MjU0OTM0MjEsInkiOjE0NzB9LHsieCI6Mzk1LjM0NzY1NjI1LCJ5IjoxNTA3fSx7IngiOjQ1MC44OTA1NjM5NjQ4NDM3NSwieSI6MTU0NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(353.5390625, 227)"><g class="label" data-id="L_B_C_0" transform="translate(-15.375, -12)"><foreignObject width="30.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Safe</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(491.421875, 867)"><g class="label" data-id="L_B_D_0" transform="translate(-24.3203125, -12)"><foreignObject width="48.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Unsafe</p></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_H_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_I_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_L_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_K_L_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_L_M_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_M_N_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_N_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_N_O_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_O_P_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_P_Q_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Q_R_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(235.95703125, 1507)"><g class="label" data-id="L_R_S_0" transform="translate(-15.375, -12)"><foreignObject width="30.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Safe</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(395.34765625, 1507)"><g class="label" data-id="L_R_D_0" transform="translate(-24.3203125, -12)"><foreignObject width="48.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Unsafe</p></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(422.48046875, 35)"><rect class="basic label-container" style="" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(422.48046875, 151)"><rect class="basic label-container" style="" x="-91.9453125" y="-39" width="183.890625" height="78"/><g class="label" style="" transform="translate(-61.9453125, -24)"><rect/><foreignObject width="123.890625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input Guardrails/<br />Safety Filtering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(353.5390625, 303)"><rect class="basic label-container" style="" x="-89.1171875" y="-39" width="178.234375" height="78"/><g class="label" style="" transform="translate(-59.1171875, -24)"><rect/><foreignObject width="118.234375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query Rewrite/<br />Query Expansion</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(491.421875, 1571)"><rect class="basic label-container" style="" x="-99.140625" y="-27" width="198.28125" height="54"/><g class="label" style="" transform="translate(-69.140625, -12)"><rect/><foreignObject width="138.28125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Rejection Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(353.5390625, 419)"><rect class="basic label-container" style="" x="-76.4453125" y="-27" width="152.890625" height="54"/><g class="label" style="" transform="translate(-46.4453125, -12)"><rect/><foreignObject width="92.890625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Encoder</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(353.5390625, 535)"><rect class="basic label-container" style="" x="-92.9375" y="-27" width="185.875" height="54"/><g class="label" style="" transform="translate(-62.9375, -12)"><rect/><foreignObject width="125.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Query Embedding</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-10" transform="translate(113.328125, 35)"><rect class="basic label-container" style="" x="-101.234375" y="-27" width="202.46875" height="54"/><g class="label" style="" transform="translate(-71.234375, -12)"><rect/><foreignObject width="142.46875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Document Database</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-11" transform="translate(113.328125, 151)"><rect class="basic label-container" style="" x="-93.796875" y="-27" width="187.59375" height="54"/><g class="label" style="" transform="translate(-63.796875, -12)"><rect/><foreignObject width="127.59375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Document Parsing</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-13" transform="translate(113.328125, 303)"><rect class="basic label-container" style="" x="-101.09375" y="-27" width="202.1875" height="54"/><g class="label" style="" transform="translate(-71.09375, -12)"><rect/><foreignObject width="142.1875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Document Chunking</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-15" transform="translate(113.328125, 419)"><rect class="basic label-container" style="" x="-105.328125" y="-27" width="210.65625" height="54"/><g class="label" style="" transform="translate(-75.328125, -12)"><rect/><foreignObject width="150.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text/Image Encoders</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-17" transform="translate(113.328125, 535)"><rect class="basic label-container" style="" x="-75.65625" y="-39" width="151.3125" height="78"/><g class="label" style="" transform="translate(-45.65625, -24)"><rect/><foreignObject width="91.3125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Indices<br />Text + Image</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-L-19" transform="translate(353.5390625, 663)"><rect class="basic label-container" style="" x="-91.9609375" y="-39" width="183.921875" height="78"/><g class="label" style="" transform="translate(-61.9609375, -24)"><rect/><foreignObject width="123.921875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Nearest Neighbor<br />Search ANN</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-M-23" transform="translate(353.5390625, 791)"><rect class="basic label-container" style="" x="-91.9453125" y="-39" width="183.890625" height="78"/><g class="label" style="" transform="translate(-61.9453125, -24)"><rect/><foreignObject width="123.890625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieved Chunks<br />Text + Images</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-N-25" transform="translate(283.375, 931)"><rect class="basic label-container" style="" x="-100" y="-27" width="200" height="54"/><g class="label" style="" transform="translate(-70, -12)"><rect/><foreignObject width="140" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt Engineering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-O-29" transform="translate(283.375, 1047)"><rect class="basic label-container" style="" x="-99.15625" y="-39" width="198.3125" height="78"/><g class="label" style="" transform="translate(-69.15625, -24)"><rect/><foreignObject width="138.3125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Enhanced Prompt +<br />Retrieved Context</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-P-31" transform="translate(283.375, 1187)"><rect class="basic label-container" style="" x="-98.6953125" y="-51" width="197.390625" height="102"/><g class="label" style="" transform="translate(-68.6953125, -36)"><rect/><foreignObject width="137.390625" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM<br />General Purpose or<br />RAFT Fine-Tuned</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Q-33" transform="translate(283.375, 1315)"><rect class="basic label-container" style="" x="-102.8828125" y="-27" width="205.765625" height="54"/><g class="label" style="" transform="translate(-72.8828125, -12)"><rect/><foreignObject width="145.765625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generated Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R-35" transform="translate(283.375, 1431)"><rect class="basic label-container" style="" x="-98.28125" y="-39" width="196.5625" height="78"/><g class="label" style="" transform="translate(-68.28125, -24)"><rect/><foreignObject width="136.5625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output Guardrails/<br />Response Safety</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-S-37" transform="translate(235.95703125, 1571)"><rect class="basic label-container" style="" x="-83.578125" y="-27" width="167.15625" height="54"/><g class="label" style="" transform="translate(-53.578125, -12)"><rect/><foreignObject width="107.15625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Display to User</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


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
