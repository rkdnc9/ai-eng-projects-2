# LLM Foundations

## Overview

This lecture notes provides a comprehensive foundation in Large Language Models (LLMs), covering the complete lifecycle from data preparation through training to deployment. The focus is on understanding how modern chatbot services like ChatGPT, Claude, Gemini, Grok, and Meta.ai are built, trained, and deployed in production systems. The lecture emphasizes both theoretical concepts and practical demonstrations, with particular attention to the two-stage training process: pre-training and post-training.

## Core Concepts

### What is an LLM?

* **Definition**: An AI model that can understand and generate text
* When trained correctly, LLMs can:
  * Answer questions in natural conversation
  * Handle follow-up questions
  * Engage in meaningful dialogue
  * Generate contextually relevant continuations

### Major LLM Providers and Chatbots

* **ChatGPT** by OpenAI - First widely available public chatbot
* **Claude** by Anthropic
* **Gemini** by Google
* **Grok** by xAI
* **Meta.ai** by Meta

**Key observation from live demonstration**: When asked "Where is Paris?", different LLMs produce different responses in tone, structure, and detail, though all are accurate. This demonstrates that:
* LLMs differ in output quality and style
* More complex questions reveal greater capability differences
* Model rankings change as companies release more powerful versions

### Two-Stage Training Architecture


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 550.277px; background-color: transparent;" viewBox="0 0 550.27734375 686" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M317.18,62L317.18,66.167C317.18,70.333,317.18,78.667,317.18,86.333C317.18,94,317.18,101,317.18,104.5L317.18,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MzE3LjE3OTY4NzUsInkiOjYyfSx7IngiOjMxNy4xNzk2ODc1LCJ5Ijo4N30seyJ4IjozMTcuMTc5Njg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M257.47,166L248.256,170.167C239.041,174.333,220.612,182.667,211.398,194.333C202.184,206,202.184,221,202.184,228.5L202.184,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjU3LjQ3MDE3NzI4MzY1Mzg3LCJ5IjoxNjZ9LHsieCI6MjAyLjE4MzU5Mzc1LCJ5IjoxOTF9LHsieCI6MjAyLjE4MzU5Mzc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M202.184,294L202.184,302.167C202.184,310.333,202.184,326.667,202.184,338.333C202.184,350,202.184,357,202.184,360.5L202.184,364" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MjAyLjE4MzU5Mzc1LCJ5IjoyOTR9LHsieCI6MjAyLjE4MzU5Mzc1LCJ5IjozNDN9LHsieCI6MjAyLjE4MzU5Mzc1LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M146.38,422L137.769,426.167C129.157,430.333,111.934,438.667,103.323,450.333C94.711,462,94.711,477,94.711,484.5L94.711,492" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTQ2LjM4MDQ4Mzc3NDAzODQ1LCJ5Ijo0MjJ9LHsieCI6OTQuNzEwOTM3NSwieSI6NDQ3fSx7IngiOjk0LjcxMDkzNzUsInkiOjQ5Nn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M94.711,550L94.711,558.167C94.711,566.333,94.711,582.667,94.711,594.333C94.711,606,94.711,613,94.711,616.5L94.711,620" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6OTQuNzEwOTM3NSwieSI6NTUwfSx7IngiOjk0LjcxMDkzNzUsInkiOjU5OX0seyJ4Ijo5NC43MTA5Mzc1LCJ5Ijo2MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M376.889,166L386.104,170.167C395.318,174.333,413.747,182.667,422.961,190.333C432.176,198,432.176,205,432.176,208.5L432.176,212" id="L_B_B1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_B1_0" data-points="W3sieCI6Mzc2Ljg4OTE5NzcxNjM0NjEzLCJ5IjoxNjZ9LHsieCI6NDMyLjE3NTc4MTI1LCJ5IjoxOTF9LHsieCI6NDMyLjE3NTc4MTI1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M257.987,422L266.598,426.167C275.21,430.333,292.433,438.667,301.045,446.333C309.656,454,309.656,461,309.656,464.5L309.656,468" id="L_D_D1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_D1_0" data-points="W3sieCI6MjU3Ljk4NjcwMzcyNTk2MTU1LCJ5Ijo0MjJ9LHsieCI6MzA5LjY1NjI1LCJ5Ijo0NDd9LHsieCI6MzA5LjY1NjI1LCJ5Ijo0NzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_B1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_D1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(317.1796875, 35)"><rect class="basic label-container" style="" x="-78.046875" y="-27" width="156.09375" height="54"/><g class="label" style="" transform="translate(-48.046875, -12)"><rect/><foreignObject width="96.09375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Internet Data</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(317.1796875, 139)"><rect class="basic label-container" style="" x="-94.90625" y="-27" width="189.8125" height="54"/><g class="label" style="" transform="translate(-64.90625, -12)"><rect/><foreignObject width="129.8125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Pre-Training Stage</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(202.18359375, 267)"><rect class="basic label-container" style="" x="-69.890625" y="-27" width="139.78125" height="54"/><g class="label" style="" transform="translate(-39.890625, -12)"><rect/><foreignObject width="79.78125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Base Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(202.18359375, 395)"><rect class="basic label-container" style="" x="-98.140625" y="-27" width="196.28125" height="54"/><g class="label" style="" transform="translate(-68.140625, -12)"><rect/><foreignObject width="136.28125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Post-Training Stage</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(94.7109375, 523)"><rect class="basic label-container" style="" x="-70.9765625" y="-27" width="141.953125" height="54"/><g class="label" style="" transform="translate(-40.9765625, -12)"><rect/><foreignObject width="81.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Final Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(94.7109375, 651)"><rect class="basic label-container" style="" x="-86.7109375" y="-27" width="173.421875" height="54"/><g class="label" style="" transform="translate(-56.7109375, -12)"><rect/><foreignObject width="113.421875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Chatbot Service</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B1-11" transform="translate(432.17578125, 267)"><rect class="basic label-container" style="" x="-110.1015625" y="-51" width="220.203125" height="102"/><g class="label" style="" transform="translate(-80.1015625, -36)"><rect/><foreignObject width="160.203125" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Months of training<br />Thousands of GPUs<br />Hundreds of millions $</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D1-13" transform="translate(309.65625, 523)"><rect class="basic label-container" style="" x="-93.96875" y="-51" width="187.9375" height="102"/><g class="label" style="" transform="translate(-63.96875, -36)"><rect/><foreignObject width="127.9375" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Days of training<br />Hundreds of GPUs<br />Lower cost</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Stage 1: Pre-Training**
* Train model on Internet data using next token prediction
* Extremely expensive: requires thousands of GPUs, months of training
* Cost: Hundreds of millions of dollars
* Output: **Base model** with implicit world knowledge
* Only well-funded startups and large companies can afford this stage

**Stage 2: Post-Training**
* Continue training base model on specialized post-training data
* Less expensive: requires hundreds of GPUs, days of training
* Cost: Significantly cheaper than pre-training
* Output: **Final model** ready for deployment

**Training Cost Reality** (from Stanford report):
* GPT-3 (175B parameters): ~$10 million
* GPT-4 and Gemini Ultra: Hundreds of millions of dollars
* Industry dominance: Most LLMs come from large tech companies with substantial funding

## Pre-Training Stage

### Data Preparation: Three Main Steps

#### Step 1: Crawling the Internet

**What is web crawling?**
* Software that starts from seed URLs
* Extracts content from each page
* Identifies outgoing links
* Recursively visits discovered links
* Continues until majority of Internet is covered

**Simple Web Crawler Logic** (Python demonstration):

```python
class SimpleWebCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited = set()  # Track visited URLs

    def crawl(self):
        to_visit = [self.base_url]  # Queue of URLs to explore

        while to_visit:
            url = to_visit.pop(0)
            if url in self.visited:
                continue

            # Get HTML content
            response = requests.get(url)
            html_content = response.text

            # Extract links using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a'):
                full_url = build_full_url(link['href'])
                to_visit.append(full_url)

            self.visited.add(url)
```

**Key components**:
* `visited` set: Tracks already-explored URLs
* `to_visit` list: Queue of discovered but unexplored URLs
* Loop continues until all reachable URLs are processed

**Two Approaches to Web Crawling**:

1. **Crawl Yourself** (preferred by large companies)
   * OpenAI and Anthropic both perform their own crawling
   * Provides maximum flexibility
   * Example: GPT-2 paper states "we created a new web script which emphasizes document quality"
   * Anthropic's website confirms: "Yes, we do crawl the web"

2. **Use Public Repositories** (preferred by smaller teams/startups)
   * **Common Crawl**: Nonprofit providing free web archives
   * Crawls web since 2008
   * Statistics:
     * ~2.7 billion web pages per crawl
     * 200-400 terabytes of HTML content per crawl
     * New crawl released every 1-2 months
   * Enables faster iteration for research and startups

#### Step 2: Data Cleaning

**Raw HTML Issues**:
* Contains many irrelevant tags, markdowns, attributes
* Most tags like `<html>`, `<head>`, `<div>` are not useful for LLM training
* Goal: Extract only meaningful text content (within `<h1>`, `<p>`, etc.)

**Additional Cleaning Requirements**:
* **Deduplication**: Remove duplicated content across the Internet
  * Same news appears on multiple websites
  * Prevents model from memorizing repeated content
  * Ensures model learns concepts, not specific text sequences
* **Content filtering**: Remove unsafe or unwanted content
* **Quality filtering**: Ensure only useful data remains

**Major Clean Datasets**:

1. **C4 (Colossal Clean Crawled Corpus)** by Google
   * Clean version of Common Crawl
   * 305 GB English data (cleaned version)
   * 2.3 TB uncleaned version
   * Format: Table with `text` and `url` columns
   * Widely used in early LLM development

2. **Dolma** (more recent)
   * Open corpus of 3 trillion tokens
   * Sources: Common Crawl, GitHub, Reddit, Wikipedia, others
   * Multiple filtering steps: quality filtering, content filtering
   * Paper provides detailed pipeline documentation

3. **RefinedWeb**
   * Another popular cleaned dataset
   * Focuses on high-quality web text
   * Similar multi-stage filtering approach

4. **FineWeb** by Hugging Face (most recent, recommended)
   * Openly available and well-documented
   * **Detailed cleaning pipeline**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 969.859px; background-color: transparent;" viewBox="0 0 969.859375 870" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M704.773,62L704.773,66.167C704.773,70.333,704.773,78.667,704.773,86.333C704.773,94,704.773,101,704.773,104.5L704.773,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NzA0Ljc3MzQzNzUsInkiOjYyfSx7IngiOjcwNC43NzM0Mzc1LCJ5Ijo4N30seyJ4Ijo3MDQuNzczNDM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M637.006,166L626.548,170.167C616.09,174.333,595.174,182.667,584.716,192.333C574.258,202,574.258,213,574.258,218.5L574.258,224" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NjM3LjAwNTcwOTEzNDYxNTQsInkiOjE2Nn0seyJ4Ijo1NzQuMjU3ODEyNSwieSI6MTkxfSx7IngiOjU3NC4yNTc4MTI1LCJ5IjoyMjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M516.006,282L502.702,288.167C489.397,294.333,462.788,306.667,449.484,318.333C436.18,330,436.18,341,436.18,346.5L436.18,352" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTE2LjAwNjEwMzUxNTYyNSwieSI6MjgyfSx7IngiOjQzNi4xNzk2ODc1LCJ5IjozMTl9LHsieCI6NDM2LjE3OTY4NzUsInkiOjM1Nn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M383.007,410L370.863,416.167C358.718,422.333,334.429,434.667,322.285,444.333C310.141,454,310.141,461,310.141,464.5L310.141,468" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MzgzLjAwNjk1ODAwNzgxMjUsInkiOjQxMH0seyJ4IjozMTAuMTQwNjI1LCJ5Ijo0NDd9LHsieCI6MzEwLjE0MDYyNSwieSI6NDcyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M310.141,526L310.141,530.167C310.141,534.333,310.141,542.667,310.141,550.333C310.141,558,310.141,565,310.141,568.5L310.141,572" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MzEwLjE0MDYyNSwieSI6NTI2fSx7IngiOjMxMC4xNDA2MjUsInkiOjU1MX0seyJ4IjozMTAuMTQwNjI1LCJ5Ijo1NzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M248.875,630L239.421,634.167C229.966,638.333,211.057,646.667,201.603,654.333C192.148,662,192.148,669,192.148,672.5L192.148,676" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MjQ4Ljg3NTQ1MDcyMTE1Mzg0LCJ5Ijo2MzB9LHsieCI6MTkyLjE0ODQzNzUsInkiOjY1NX0seyJ4IjoxOTIuMTQ4NDM3NSwieSI6NjgwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M132.348,734L123.119,738.167C113.891,742.333,95.434,750.667,86.205,760.333C76.977,770,76.977,781,76.977,786.5L76.977,792" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6MTMyLjM0NzY1NjI1LCJ5Ijo3MzR9LHsieCI6NzYuOTc2NTYyNSwieSI6NzU5fSx7IngiOjc2Ljk3NjU2MjUsInkiOjc5Nn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M772.541,166L782.999,170.167C793.457,174.333,814.373,182.667,824.831,190.333C835.289,198,835.289,205,835.289,208.5L835.289,212" id="L_B_B1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_B1_0" data-points="W3sieCI6NzcyLjU0MTE2NTg2NTM4NDYsInkiOjE2Nn0seyJ4Ijo4MzUuMjg5MDYyNSwieSI6MTkxfSx7IngiOjgzNS4yODkwNjI1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M632.51,282L645.814,288.167C659.118,294.333,685.727,306.667,699.032,316.333C712.336,326,712.336,333,712.336,336.5L712.336,340" id="L_C_C1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_C1_0" data-points="W3sieCI6NjMyLjUwOTUyMTQ4NDM3NSwieSI6MjgyfSx7IngiOjcxMi4zMzU5Mzc1LCJ5IjozMTl9LHsieCI6NzEyLjMzNTkzNzUsInkiOjM0NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M489.352,410L501.497,416.167C513.641,422.333,537.93,434.667,550.074,444.333C562.219,454,562.219,461,562.219,464.5L562.219,468" id="L_D_D1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_D1_0" data-points="W3sieCI6NDg5LjM1MjQxNjk5MjE4NzUsInkiOjQxMH0seyJ4Ijo1NjIuMjE4NzUsInkiOjQ0N30seyJ4Ijo1NjIuMjE4NzUsInkiOjQ3Mn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M371.406,630L380.86,634.167C390.315,638.333,409.224,646.667,418.678,654.333C428.133,662,428.133,669,428.133,672.5L428.133,676" id="L_F_F1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_F1_0" data-points="W3sieCI6MzcxLjQwNTc5OTI3ODg0NjEzLCJ5Ijo2MzB9LHsieCI6NDI4LjEzMjgxMjUsInkiOjY1NX0seyJ4Ijo0MjguMTMyODEyNSwieSI6NjgwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M251.949,734L261.178,738.167C270.406,742.333,288.863,750.667,298.092,758.333C307.32,766,307.32,773,307.32,776.5L307.32,780" id="L_G_G1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_G1_0" data-points="W3sieCI6MjUxLjk0OTIxODc1LCJ5Ijo3MzR9LHsieCI6MzA3LjMyMDMxMjUsInkiOjc1OX0seyJ4IjozMDcuMzIwMzEyNSwieSI6Nzg0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_B1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_C1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_D1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_F1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_G1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(704.7734375, 35)"><rect class="basic label-container" style="" x="-101.0703125" y="-27" width="202.140625" height="54"/><g class="label" style="" transform="translate(-71.0703125, -12)"><rect/><foreignObject width="142.140625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Raw Common Crawl</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(704.7734375, 139)"><rect class="basic label-container" style="" x="-76.1640625" y="-27" width="152.328125" height="54"/><g class="label" style="" transform="translate(-46.1640625, -12)"><rect/><foreignObject width="92.328125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>URL Filtering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(574.2578125, 255)"><rect class="basic label-container" style="" x="-84.4609375" y="-27" width="168.921875" height="54"/><g class="label" style="" transform="translate(-54.4609375, -12)"><rect/><foreignObject width="108.921875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Extraction</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(436.1796875, 383)"><rect class="basic label-container" style="" x="-96.15625" y="-27" width="192.3125" height="54"/><g class="label" style="" transform="translate(-66.15625, -12)"><rect/><foreignObject width="132.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Language Filtering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(310.140625, 499)"><rect class="basic label-container" style="" x="-88.3046875" y="-27" width="176.609375" height="54"/><g class="label" style="" transform="translate(-58.3046875, -12)"><rect/><foreignObject width="116.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Quality Filtering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(310.140625, 603)"><rect class="basic label-container" style="" x="-79.4765625" y="-27" width="158.953125" height="54"/><g class="label" style="" transform="translate(-49.4765625, -12)"><rect/><foreignObject width="98.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Deduplication</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(192.1484375, 707)"><rect class="basic label-container" style="" x="-71.4375" y="-27" width="142.875" height="54"/><g class="label" style="" transform="translate(-41.4375, -12)"><rect/><foreignObject width="82.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>PII Removal</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(76.9765625, 823)"><rect class="basic label-container" style="" x="-68.9765625" y="-27" width="137.953125" height="54"/><g class="label" style="" transform="translate(-38.9765625, -12)"><rect/><foreignObject width="77.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Clean Data</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B1-15" transform="translate(835.2890625, 255)"><rect class="basic label-container" style="" x="-126.5703125" y="-39" width="253.140625" height="78"/><g class="label" style="" transform="translate(-96.5703125, -24)"><rect/><foreignObject width="193.140625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Block adult content<br />Filter unwanted categories</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C1-17" transform="translate(712.3359375, 383)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Extract text from HTML tags</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D1-19" transform="translate(562.21875, 499)"><rect class="basic label-container" style="" x="-113.7734375" y="-27" width="227.546875" height="54"/><g class="label" style="" transform="translate(-83.7734375, -12)"><rect/><foreignObject width="167.546875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Keep desired languages</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F1-21" transform="translate(428.1328125, 707)"><rect class="basic label-container" style="" x="-114.546875" y="-27" width="229.09375" height="54"/><g class="label" style="" transform="translate(-84.546875, -12)"><rect/><foreignObject width="169.09375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Remove similar content</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G1-23" transform="translate(307.3203125, 823)"><rect class="basic label-container" style="" x="-111.3671875" y="-39" width="222.734375" height="78"/><g class="label" style="" transform="translate(-81.3671875, -24)"><rect/><foreignObject width="162.734375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Remove bank accounts<br />phone numbers, etc.</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**FineWeb Statistics**:
* 44 terabytes disk space
* ~15 trillion tokens (words/subwords)
* Data format: Table with `text` and `url` columns
* Example content: Random text from diverse domains ("Did you know...", "Five reasons I love Boston...")

**Key Principle**: Cleaning pipelines remove redundancy but preserve unique content and domain diversity

#### Step 3: Tokenization

**Purpose**: Convert raw clean text to sequence of discrete numbers
* Machine learning models require numerical inputs, not text
* Tokenization bridges text and numerical representation

**Two Phases of Tokenization**:

**Phase 1: Training Phase**


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1204.17px; background-color: transparent;" viewBox="0 0 1204.171875 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M209.391,47L213.557,47C217.724,47,226.057,47,233.724,47C241.391,47,248.391,47,251.891,47L255.391,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjA5LjM5MDYyNSwieSI6NDd9LHsieCI6MjM0LjM5MDYyNSwieSI6NDd9LHsieCI6MjU5LjM5MDYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M456.531,47L460.698,47C464.865,47,473.198,47,480.865,47C488.531,47,495.531,47,499.031,47L502.531,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDU2LjUzMTI1LCJ5Ijo0N30seyJ4Ijo0ODEuNTMxMjUsInkiOjQ3fSx7IngiOjUwNi41MzEyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M711.25,47L715.417,47C719.583,47,727.917,47,735.583,47C743.25,47,750.25,47,753.75,47L757.25,47" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NzExLjI1LCJ5Ijo0N30seyJ4Ijo3MzYuMjUsInkiOjQ3fSx7IngiOjc2MS4yNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M940.234,47L944.401,47C948.568,47,956.901,47,964.568,47C972.234,47,979.234,47,982.734,47L986.234,47" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6OTQwLjIzNDM3NSwieSI6NDd9LHsieCI6OTY1LjIzNDM3NSwieSI6NDd9LHsieCI6OTkwLjIzNDM3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(108.6953125, 47)"><rect class="basic label-container" style="" x="-100.6953125" y="-27" width="201.390625" height="54"/><g class="label" style="" transform="translate(-70.6953125, -12)"><rect/><foreignObject width="141.390625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Long Text Sequence</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(357.9609375, 47)"><rect class="basic label-container" style="" x="-98.5703125" y="-27" width="197.140625" height="54"/><g class="label" style="" transform="translate(-68.5703125, -12)"><rect/><foreignObject width="137.140625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Text Splitting Logic</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(608.890625, 47)"><rect class="basic label-container" style="" x="-102.359375" y="-27" width="204.71875" height="54"/><g class="label" style="" transform="translate(-72.359375, -12)"><rect/><foreignObject width="144.71875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>List of Smaller Units</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(850.7421875, 47)"><rect class="basic label-container" style="" x="-89.4921875" y="-27" width="178.984375" height="54"/><g class="label" style="" transform="translate(-59.4921875, -12)"><rect/><foreignObject width="118.984375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Build Vocabulary</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(1093.203125, 47)"><rect class="basic label-container" style="" x="-102.96875" y="-39" width="205.9375" height="78"/><g class="label" style="" transform="translate(-72.96875, -24)"><rect/><foreignObject width="145.9375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Vocabulary Table<br />Token to ID mapping</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* **Text Splitting**: Apply algorithm to break text into smaller units (tokens)
  * Example: "Machine learning. ML is a subfield of artificial intelligence." becomes ["machine", "learning", ".", "ML", "is", "a", "subfield", "of", "artificial", "intelligence", "."]
* **Build Vocabulary**: Find all unique tokens and assign IDs
  * Example vocabulary: {a: 0, about: 1, after: 2, all: 3, also: 4, ...}
  * Vocabulary size varies based on splitting algorithm (can be 50K-270K+ tokens)

**Phase 2: Inference Phase**

* **Encoding** (text to numbers):
  * Apply same text splitting logic
  * Replace each token with its vocabulary ID
  * Example: "Tell me a joke" becomes [10512, 502, 257, 9707]

* **Decoding** (numbers to text):
  * Use inverse vocabulary table (ID to token)
  * Replace each ID with corresponding token
  * Apply inverse of text splitting to reconstruct text

### Tokenization Algorithms: Three Categories

#### 1. Word-Level Tokenization

**Approach**: Split text by whitespaces to get individual words

**Example**: "It's perfectly fine" becomes ["It's", "perfectly", "fine"]

**Characteristics**:
* Each word gets own token ID
* Vocabulary can be 100K-270K+ tokens
* Large vocabulary because Internet has many unique words

**Limitation**:
* **Huge vocabulary size** makes training expensive
* Must learn associations for hundreds of thousands of tokens
* Not used in modern LLMs

#### 2. Character-Level Tokenization

**Approach**: Split text into individual characters

**Example**: "perfectly fine" becomes ["p", "e", "r", "f", "e", "c", "t", "l", "y", " ", "f", "i", "n", "e"]

**Characteristics**:
* Very small vocabulary (~105 tokens: lowercase, uppercase, punctuation)
* Low token IDs (typically single or double digits)

**Limitation**:
* **Long sequence of numbers** after tokenization
* Model must learn dependencies across many tokens
* Computationally expensive during training
* Not used in modern LLMs

#### 3. Subword-Level Tokenization (Modern Standard)

**Key Insight**: Balance between word-level and character-level
* Tokens are **larger than characters** but **smaller than words**
* Example: "perfectly fine" becomes ["perfect", "ly", "fine"]

**Advantages**:
* Manageable vocabulary size
* Reasonable sequence length
* Best trade-off for efficiency and effectiveness

**Most Popular Algorithm: Byte Pair Encoding (BPE)**

**BPE Algorithm**:
1. Start with character-level tokens
2. Iteratively merge most frequent pairs
3. Create new tokens from frequent pairs
4. Continue until vocabulary reaches target size (e.g., 50K tokens)

**Example of BPE Process**:
* Initial: Individual characters as tokens
* Find most frequent pair (e.g., "U" + "G" appears often)
* Create new token "UG"
* Continue merging until vocabulary size limit reached

**BPE Usage in Modern LLMs**:
* GPT-2 paper: Uses BPE as tokenizer
* Llama 3 paper: "The tokenizer is a BPE model"
* Industry standard for advanced LLMs

**Vocabulary Size Comparison**:
* Character-level: ~105 tokens
* Subword-level: 50K-200K tokens (typical range)
* Word-level: 270K+ tokens (potentially millions with multiple languages)

**Real Example - Token Distribution**:
* Common English words: Single tokens ("the", "of", "home")
* Common suffixes: Single tokens ("ing", "ed", "able")
* Uncommon words: Multiple tokens
* Example: "walking" = "walk" + "ing"

### Practical Tokenization Tools

#### TikToken Library (by OpenAI)

**Features**:
* Fast BPE tokenizer (3-6x faster than alternatives)
* Open source and easy to use
* Pre-trained tokenizers available for GPT models

**Code demonstration**:

```python
import tiktoken

# Load pre-trained tokenizer for GPT-3.5
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Encode text to token IDs
text = "I love machine learning!"
token_ids = tokenizer.encode(text)
# Output: [40, 3021, 5780, 6975, 0]

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)
# Output: "I love machine learning!"

# Check vocabulary size
vocab_size = tokenizer.n_vocab
# Output: 100277 tokens

# Try different model tokenizer
tokenizer_gpt4 = tiktoken.encoding_for_model("gpt-4o")
# Uses: o200k_base encoding
# Vocabulary: 200,019 tokens
```

**Key Observations**:
* Different models use different tokenizers
* Larger vocabularies in newer models (GPT-4: 200K vs GPT-3.5: 100K)
* Larger vocabulary = more efficient tokenization

#### TikTokenizer Website (Visualization Tool)

**Purpose**: Visualize how text gets tokenized

**Example Session**:
* Input: "I love machine learning!"
* Output: 5 tokens = [" I", " love", " machine", " learning", "!"]
* Each token shows its ID from vocabulary

**Interesting Cases**:
* "perfectly" (correct spelling) = 1 token
* "perfectely" (misspelling) = 3 tokens ["perfect", "el", "y"]
* Common sequences get single tokens
* Rare/random sequences split into smaller units

**Available Tokenizers**:
* Multiple pre-trained tokenizers to test
* Can compare different encoding schemes
* Useful for understanding token efficiency

### Model Architecture

#### Neural Networks: Foundation Concepts

**Goal of Machine Learning Models**: Learn mapping from input X to output Y

**Three Example Domains**:

1. **House Price Prediction**
   * Input X: Location, bedrooms, house size (converted to numbers)
   * Output Y: Price (single number)
   * Goal: Learn X to Y mapping

2. **Email Spam Classification**
   * Input X: Sender, content, subject (converted to numbers)
   * Output Y: Spam (1) or Not Spam (0)
   * Goal: Classify emails based on features

3. **Tumor Detection**
   * Input X: Image pixels (numbers)
   * Output Y: Location coordinates (4 numbers: top-left x,y and bottom-right x,y)
   * Goal: Detect and locate tumors

**Neural Network Definition**: Sequence of parameterized transformations that maps input to output

**Conceptual Flow**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1108.59px; background-color: transparent;" viewBox="0 0 1108.59375 296" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M118.922,203L123.089,203C127.255,203,135.589,203,143.255,203C150.922,203,157.922,203,161.422,203L164.922,203" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTE4LjkyMTg3NSwieSI6MjAzfSx7IngiOjE0My45MjE4NzUsInkiOjIwM30seyJ4IjoxNjguOTIxODc1LCJ5IjoyMDN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M295.589,164L300.54,160.833C305.492,157.667,315.394,151.333,330.961,148.167C346.529,145,367.76,145,378.376,145L388.992,145" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6Mjk1LjU4ODkwMDg2MjA2ODk1LCJ5IjoxNjR9LHsieCI6MzI1LjI5Njg3NSwieSI6MTQ1fSx7IngiOjM5Mi45OTIxODc1LCJ5IjoxNDV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M524.367,116.437L535.65,111.53C546.932,106.624,569.497,96.812,591.396,91.906C613.294,87,634.526,87,645.142,87L655.758,87" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTI0LjM2NzE4NzUsInkiOjExNi40MzY1MzcyMjI1MTUwOH0seyJ4Ijo1OTIuMDYyNSwieSI6ODd9LHsieCI6NjU5Ljc1NzgxMjUsInkiOjg3fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M791.133,61.391L802.415,56.993C813.698,52.594,836.263,43.797,858.815,39.399C881.367,35,903.906,35,915.176,35L926.445,35" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NzkxLjEzMjgxMjUsInkiOjYxLjM5MTM3ODE5OTQ5NjI4fSx7IngiOjg1OC44MjgxMjUsInkiOjM1fSx7IngiOjkzMC40NDUzMTI1LCJ5IjozNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M295.589,242L300.54,245.167C305.492,248.333,315.394,254.667,323.846,257.833C332.297,261,339.297,261,342.797,261L346.297,261" id="L_B_B1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_B1_0" data-points="W3sieCI6Mjk1LjU4ODkwMDg2MjA2ODk1LCJ5IjoyNDJ9LHsieCI6MzI1LjI5Njg3NSwieSI6MjYxfSx7IngiOjM1MC4yOTY4NzUsInkiOjI2MX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M524.367,173.563L535.65,178.47C546.932,183.376,569.497,193.188,584.28,198.094C599.063,203,606.063,203,609.563,203L613.063,203" id="L_C_C1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_C1_0" data-points="W3sieCI6NTI0LjM2NzE4NzUsInkiOjE3My41NjM0NjI3Nzc0ODQ5Mn0seyJ4Ijo1OTIuMDYyNSwieSI6MjAzfSx7IngiOjYxNy4wNjI1LCJ5IjoyMDN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M791.133,112.609L802.415,117.007C813.698,121.406,836.263,130.203,851.046,134.601C865.828,139,872.828,139,876.328,139L879.828,139" id="L_D_D1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_D1_0" data-points="W3sieCI6NzkxLjEzMjgxMjUsInkiOjExMi42MDg2MjE4MDA1MDM3M30seyJ4Ijo4NTguODI4MTI1LCJ5IjoxMzl9LHsieCI6ODgzLjgyODEyNSwieSI6MTM5fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_B1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_C1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_D1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(63.4609375, 203)"><rect class="basic label-container" style="" x="-55.4609375" y="-27" width="110.921875" height="54"/><g class="label" style="" transform="translate(-25.4609375, -12)"><rect/><foreignObject width="50.921875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input X</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(234.609375, 203)"><rect class="basic label-container" style="" x="-65.6875" y="-39" width="131.375" height="78"/><g class="label" style="" transform="translate(-35.6875, -24)"><rect/><foreignObject width="71.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Layer 1<br />Transform</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(458.6796875, 145)"><rect class="basic label-container" style="" x="-65.6875" y="-39" width="131.375" height="78"/><g class="label" style="" transform="translate(-35.6875, -24)"><rect/><foreignObject width="71.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Layer 2<br />Transform</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(725.4453125, 87)"><rect class="basic label-container" style="" x="-65.6875" y="-39" width="131.375" height="78"/><g class="label" style="" transform="translate(-35.6875, -24)"><rect/><foreignObject width="71.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Layer n<br />Transform</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(992.2109375, 35)"><rect class="basic label-container" style="" x="-61.765625" y="-27" width="123.53125" height="54"/><g class="label" style="" transform="translate(-31.765625, -12)"><rect/><foreignObject width="63.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output Y</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B1-9" transform="translate(458.6796875, 261)"><rect class="basic label-container" style="" x="-108.3828125" y="-27" width="216.765625" height="54"/><g class="label" style="" transform="translate(-78.3828125, -12)"><rect/><foreignObject width="156.765625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Learnable Parameters</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C1-11" transform="translate(725.4453125, 203)"><rect class="basic label-container" style="" x="-108.3828125" y="-27" width="216.765625" height="54"/><g class="label" style="" transform="translate(-78.3828125, -12)"><rect/><foreignObject width="156.765625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Learnable Parameters</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D1-13" transform="translate(992.2109375, 139)"><rect class="basic label-container" style="" x="-108.3828125" y="-27" width="216.765625" height="54"/><g class="label" style="" transform="translate(-78.3828125, -12)"><rect/><foreignObject width="156.765625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Learnable Parameters</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* Each layer transforms its input using learnable parameters (weights)
* Parameters are learned during training
* Multiple layers stack to create deep transformations

#### Common Neural Network Layers

**1. Linear Layer (Fundamental Building Block)**

**Mathematical Definition**: Y = W × X + b
* W: Weight matrix (learnable parameters)
* X: Input vector
* b: Bias vector (learnable parameter)
* Y: Output vector

**Example**:
* Input X = [0.1, 0.5, 0.3, 0.8] (4 numbers)
* Output Y = single number (0.6)
* Weight matrix W shape: 1 × 4 (one output, four inputs)
* Bias b shape: 1 (one output)

**Visualization as Neural Network**:

```
Input Layer (4 neurons)    Output Layer (1 neuron)
    O (0.1) ---w11--->
    O (0.5) ---w12--->     O (0.6)
    O (0.3) ---w13--->
    O (0.8) ---w14--->
```

* Circles = neurons
* Lines = connections with weights
* Term "neural network" comes from neuron-connection metaphor

**Generalizing to Multiple Outputs**:
* More output neurons = more rows in weight matrix
* 2 outputs, 4 inputs = weight matrix shape 2 × 4
* More connections = more parameters to learn

**Key Insight**: Linear layer is just a mathematical expression (matrix multiplication + addition)

**2. Other Layer Types** (mentioned for context):
* **Convolutional layers**: Better for spatial inputs (images)
* **Activation layers**: Introduce non-linearity
* **Attention layers**: Focus on relevant parts of input
* Each layer type has different transformation formula
* All layers share core purpose: transform input to output

**Neural Network as Composition**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1115.2px; background-color: transparent;" viewBox="0 0 1115.203125 94" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M76.906,47L81.073,47C85.24,47,93.573,47,101.24,47C108.906,47,115.906,47,119.406,47L122.906,47" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NzYuOTA2MjUsInkiOjQ3fSx7IngiOjEwMS45MDYyNSwieSI6NDd9LHsieCI6MTI2LjkwNjI1LCJ5Ijo0N31d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M244.875,47L249.042,47C253.208,47,261.542,47,269.208,47C276.875,47,283.875,47,287.375,47L290.875,47" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjQ0Ljg3NSwieSI6NDd9LHsieCI6MjY5Ljg3NSwieSI6NDd9LHsieCI6Mjk0Ljg3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448.328,47L452.495,47C456.661,47,464.995,47,472.661,47C480.328,47,487.328,47,490.828,47L494.328,47" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NDQ4LjMyODEyNSwieSI6NDd9LHsieCI6NDczLjMyODEyNSwieSI6NDd9LHsieCI6NDk4LjMyODEyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M616.297,47L620.464,47C624.63,47,632.964,47,640.63,47C648.297,47,655.297,47,658.797,47L662.297,47" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NjE2LjI5Njg3NSwieSI6NDd9LHsieCI6NjQxLjI5Njg3NSwieSI6NDd9LHsieCI6NjY2LjI5Njg3NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M819.75,47L823.917,47C828.083,47,836.417,47,844.083,47C851.75,47,858.75,47,862.25,47L865.75,47" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6ODE5Ljc1LCJ5Ijo0N30seyJ4Ijo4NDQuNzUsInkiOjQ3fSx7IngiOjg2OS43NSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M988.078,47L992.245,47C996.411,47,1004.745,47,1012.411,47C1020.078,47,1027.078,47,1030.578,47L1034.078,47" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6OTg4LjA3ODEyNSwieSI6NDd9LHsieCI6MTAxMy4wNzgxMjUsInkiOjQ3fSx7IngiOjEwMzguMDc4MTI1LCJ5Ijo0N31d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(42.453125, 47)"><rect class="basic label-container" style="" x="-34.453125" y="-27" width="68.90625" height="54"/><g class="label" style="" transform="translate(-4.453125, -12)"><rect/><foreignObject width="8.90625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>X</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(185.890625, 47)"><rect class="basic label-container" style="" x="-58.984375" y="-39" width="117.96875" height="78"/><g class="label" style="" transform="translate(-28.984375, -24)"><rect/><foreignObject width="57.96875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Linear 1<br />W1, b1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(371.6015625, 47)"><rect class="basic label-container" style="" x="-76.7265625" y="-27" width="153.453125" height="54"/><g class="label" style="" transform="translate(-46.7265625, -12)"><rect/><foreignObject width="93.453125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Intermediate</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(557.3125, 47)"><rect class="basic label-container" style="" x="-58.984375" y="-39" width="117.96875" height="78"/><g class="label" style="" transform="translate(-28.984375, -24)"><rect/><foreignObject width="57.96875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Linear 2<br />W2, b2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(743.0234375, 47)"><rect class="basic label-container" style="" x="-76.7265625" y="-27" width="153.453125" height="54"/><g class="label" style="" transform="translate(-46.7265625, -12)"><rect/><foreignObject width="93.453125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Intermediate</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(928.9140625, 47)"><rect class="basic label-container" style="" x="-59.1640625" y="-39" width="118.328125" height="78"/><g class="label" style="" transform="translate(-29.1640625, -24)"><rect/><foreignObject width="58.328125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Linear n<br />Wn, bn</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(1072.640625, 47)"><rect class="basic label-container" style="" x="-34.5625" y="-27" width="69.125" height="54"/><g class="label" style="" transform="translate(-4.5625, -12)"><rect/><foreignObject width="9.125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Y</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* Entire network = single mathematical expression
* Output Y computed from input X using all layer weights
* Training adjusts weights to learn desired X to Y mapping

#### Transformer Architecture

**Historical Context**: In 2017, Google published "Attention is All You Need"
* Introduced **Transformer architecture**
* Unique way of combining existing layers
* Initially designed for machine translation

**Transformer Structure**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 682.16px; background-color: transparent;" viewBox="0 0 682.16015625 486" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M414.109,62L414.109,66.167C414.109,70.333,414.109,78.667,414.109,86.333C414.109,94,414.109,101,414.109,104.5L414.109,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NDE0LjEwOTM3NSwieSI6NjJ9LHsieCI6NDE0LjEwOTM3NSwieSI6ODd9LHsieCI6NDE0LjEwOTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M340.251,166L328.853,170.167C317.455,174.333,294.659,182.667,283.261,190.333C271.863,198,271.863,205,271.863,208.5L271.863,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzQwLjI1MDgyNjMyMjExNTM2LCJ5IjoxNjZ9LHsieCI6MjcxLjg2MzI4MTI1LCJ5IjoxOTF9LHsieCI6MjcxLjg2MzI4MTI1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M271.863,270L271.863,274.167C271.863,278.333,271.863,286.667,271.863,294.333C271.863,302,271.863,309,271.863,312.5L271.863,316" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MjcxLjg2MzI4MTI1LCJ5IjoyNzB9LHsieCI6MjcxLjg2MzI4MTI1LCJ5IjoyOTV9LHsieCI6MjcxLjg2MzI4MTI1LCJ5IjozMjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M196.467,374L184.832,378.167C173.197,382.333,149.927,390.667,138.291,398.333C126.656,406,126.656,413,126.656,416.5L126.656,420" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTk2LjQ2NzMyMjcxNjM0NjE2LCJ5IjozNzR9LHsieCI6MTI2LjY1NjI1LCJ5IjozOTl9LHsieCI6MTI2LjY1NjI1LCJ5Ijo0MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M487.968,166L499.366,170.167C510.764,174.333,533.56,182.667,544.958,190.333C556.355,198,556.355,205,556.355,208.5L556.355,212" id="L_B_B1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_B1_0" data-points="W3sieCI6NDg3Ljk2NzkyMzY3Nzg4NDY0LCJ5IjoxNjZ9LHsieCI6NTU2LjM1NTQ2ODc1LCJ5IjoxOTF9LHsieCI6NTU2LjM1NTQ2ODc1LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M347.259,374L358.894,378.167C370.53,382.333,393.8,390.667,405.435,398.333C417.07,406,417.07,413,417.07,416.5L417.07,420" id="L_D_D1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_D1_0" data-points="W3sieCI6MzQ3LjI1OTIzOTc4MzY1MzgsInkiOjM3NH0seyJ4Ijo0MTcuMDcwMzEyNSwieSI6Mzk5fSx7IngiOjQxNy4wNzAzMTI1LCJ5Ijo0MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_B1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_D1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(414.109375, 35)"><rect class="basic label-container" style="" x="-113.8984375" y="-27" width="227.796875" height="54"/><g class="label" style="" transform="translate(-83.8984375, -12)"><rect/><foreignObject width="167.796875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input: Source Language</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(414.109375, 139)"><rect class="basic label-container" style="" x="-80.46875" y="-27" width="160.9375" height="54"/><g class="label" style="" transform="translate(-50.46875, -12)"><rect/><foreignObject width="100.9375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Encoder Stack</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(271.86328125, 243)"><rect class="basic label-container" style="" x="-116.6875" y="-27" width="233.375" height="54"/><g class="label" style="" transform="translate(-86.6875, -12)"><rect/><foreignObject width="173.375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Encoded Representation</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(271.86328125, 347)"><rect class="basic label-container" style="" x="-81.0859375" y="-27" width="162.171875" height="54"/><g class="label" style="" transform="translate(-51.0859375, -12)"><rect/><foreignObject width="102.171875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Decoder Stack</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(126.65625, 451)"><rect class="basic label-container" style="" x="-118.65625" y="-27" width="237.3125" height="54"/><g class="label" style="" transform="translate(-88.65625, -12)"><rect/><foreignObject width="177.3125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output: Target Language</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B1-9" transform="translate(556.35546875, 243)"><rect class="basic label-container" style="" x="-117.8046875" y="-27" width="235.609375" height="54"/><g class="label" style="" transform="translate(-87.8046875, -12)"><rect/><foreignObject width="175.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Left side of architecture</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D1-11" transform="translate(417.0703125, 451)"><rect class="basic label-container" style="" x="-121.7578125" y="-27" width="243.515625" height="54"/><g class="label" style="" transform="translate(-91.7578125, -12)"><rect/><foreignObject width="183.515625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Right side of architecture</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* **Encoder**: Processes source language text
* **Decoder**: Generates target language text
* Both composed of stacked layers with attention mechanisms

**Key Innovation**: Specific way of stacking layers works extremely well for text tasks

**Transformer as Mathematical Expression**:
* Like any neural network, entire transformer = single mathematical expression
* Complex but computable formula from inputs to outputs
* All intermediate computations use learnable parameters

**Recommended Resource**: Jay Alamar's "The Illustrated Transformer"
* Visual explanations of all components
* Detailed breakdown of encoder and decoder
* Shows data flow from source to target language

#### Decoder-Only Transformer (Modern LLM Architecture)

**Key Discovery**: Keeping only decoder part (right side) is powerful for text generation

**Why Decoder-Only?**
* **Text generation** is core LLM capability
* Decoder specializes in generating sequential output
* Encoder not needed when task is pure generation (not translation)

**Decoder-Only Structure**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 457.109px; background-color: transparent;" viewBox="0 0 457.109375 662" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M109.531,98L109.531,108.167C109.531,118.333,109.531,138.667,109.531,152.333C109.531,166,109.531,173,109.531,176.5L109.531,180" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTA5LjUzMTI1LCJ5Ijo5OH0seyJ4IjoxMDkuNTMxMjUsInkiOjE1OX0seyJ4IjoxMDkuNTMxMjUsInkiOjE4NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.531,238L109.531,242.167C109.531,246.333,109.531,254.667,109.531,262.333C109.531,270,109.531,277,109.531,280.5L109.531,284" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTA5LjUzMTI1LCJ5IjoyMzh9LHsieCI6MTA5LjUzMTI1LCJ5IjoyNjN9LHsieCI6MTA5LjUzMTI1LCJ5IjoyODh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.531,342L109.531,346.167C109.531,350.333,109.531,358.667,109.531,366.333C109.531,374,109.531,381,109.531,384.5L109.531,388" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTA5LjUzMTI1LCJ5IjozNDJ9LHsieCI6MTA5LjUzMTI1LCJ5IjozNjd9LHsieCI6MTA5LjUzMTI1LCJ5IjozOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.531,446L109.531,450.167C109.531,454.333,109.531,462.667,109.531,470.333C109.531,478,109.531,485,109.531,488.5L109.531,492" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTA5LjUzMTI1LCJ5Ijo0NDZ9LHsieCI6MTA5LjUzMTI1LCJ5Ijo0NzF9LHsieCI6MTA5LjUzMTI1LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.531,550L109.531,554.167C109.531,558.333,109.531,566.667,109.531,574.333C109.531,582,109.531,589,109.531,592.5L109.531,596" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MTA5LjUzMTI1LCJ5Ijo1NTB9LHsieCI6MTA5LjUzMTI1LCJ5Ijo1NzV9LHsieCI6MTA5LjUzMTI1LCJ5Ijo2MDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(109.53125, 71)"><rect class="basic label-container" style="" x="-74.8125" y="-27" width="149.625" height="54"/><g class="label" style="" transform="translate(-44.8125, -12)"><rect/><foreignObject width="89.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input Tokens</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(109.53125, 211)"><rect class="basic label-container" style="" x="-101.3515625" y="-27" width="202.703125" height="54"/><g class="label" style="" transform="translate(-71.3515625, -12)"><rect/><foreignObject width="142.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Transformer Block 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(109.53125, 315)"><rect class="basic label-container" style="" x="-101.3515625" y="-27" width="202.703125" height="54"/><g class="label" style="" transform="translate(-71.3515625, -12)"><rect/><foreignObject width="142.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Transformer Block 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(109.53125, 419)"><rect class="basic label-container" style="" x="-38.8125" y="-27" width="77.625" height="54"/><g class="label" style="" transform="translate(-8.8125, -12)"><rect/><foreignObject width="17.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>...</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(109.53125, 523)"><rect class="basic label-container" style="" x="-101.53125" y="-27" width="203.0625" height="54"/><g class="label" style="" transform="translate(-71.53125, -12)"><rect/><foreignObject width="143.0625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Transformer Block n</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(109.53125, 627)"><rect class="basic label-container" style="" x="-54.9375" y="-27" width="109.875" height="54"/><g class="label" style="" transform="translate(-24.9375, -12)"><rect/><foreignObject width="49.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-10" transform="translate(341.7265625, 71)"><rect class="basic label-container" style="" x="-107.3828125" y="-63" width="214.765625" height="126"/><g class="label" style="" transform="translate(-77.3828125, -48)"><rect/><foreignObject width="154.765625" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Each block contains:<br />- Attention layers<br />- Feed-forward layers<br />- Normalization</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* Multiple transformer blocks stacked (n times)
* Each block = attention + feed-forward layers
* Blocks are identical in structure, different in learned parameters

**Critical Fact**: **All modern LLMs use decoder-only transformer architecture**
* LLM = decoder-only transformer trained on Internet data
* Industry standard across all major models

**What Differs Between LLMs**: Hyperparameters

**GPT-2 Model Variations** (OpenAI):

| Model | Layers (n) | Model Dimension | Parameters |
|-------|-----------|----------------|------------|
| Small | 12 | 768 | 117M |
| Medium | 24 | 1024 | 345M |
| Large | 36 | 1280 | 762M |
| XL | 48 | 1600 | 1.5B |

* **Layers (n)**: Number of stacked transformer blocks
* **Model dimension**: Size of internal vectors
* More layers + larger dimension = more parameters

**GPT-3 Model Variations** (OpenAI):

| Model | Layers | Model Dimension | Parameters |
|-------|--------|----------------|------------|
| Small | 12 | 768 | 125M |
| ... | ... | ... | ... |
| Largest | 96 | 12,288 | 175B |

* **Scaling trend**: Each generation increases model size
* GPT-3 largest: 175 billion parameters
* Requires significant engineering to train and serve

**Llama 3 Model Variations** (Meta):

| Model | Layers | Model Dimension | Parameters |
|-------|--------|----------------|------------|
| 8B | 32 | 4,096 | 8B |
| 70B | 80 | 8,192 | 70B |
| 405B | 126 | 16,384 | 405B |

* **Llama 3 405B**: Even larger than GPT-3
* More parameters = more capacity to learn complex patterns
* Also more expensive to train and deploy

**Scaling Principle**:
* More parameters = more model capacity
* Greater capacity = better performance (generally)
* Trade-off: Higher computational cost and memory requirements

### Model Training

#### The Training Challenge

**Problem Before Training**: Random parameters produce random outputs
* All layer weights initialized randomly
* Input "I hope you are" produces meaningless probability distribution
* Cannot rely on model predictions

**Purpose of Training**: Tune internal parameters using training data
* Expose model to entire Internet data
* Adjust weights so predictions become accurate
* Transform random model into useful next-token predictor

#### Training Process Overview

**Training Data**: Clean tokenized Internet text
* Example paragraph: "Albert Einstein was a German-born physicist and mathematician..."
* Represented as sequence of token IDs (shown as text here for clarity)

**Core Training Loop**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1243.14px; background-color: transparent;" viewBox="0 0 1243.140625 161" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M268,102L272.167,102C276.333,102,284.667,102,292.333,102C300,102,307,102,310.5,102L314,102" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjY4LCJ5IjoxMDJ9LHsieCI6MjkzLCJ5IjoxMDJ9LHsieCI6MzE4LCJ5IjoxMDJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448.999,75L454.942,72.333C460.885,69.667,472.771,64.333,482.213,61.667C491.656,59,498.656,59,502.156,59L505.656,59" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDQ4Ljk5OTI3MzI1NTgxMzkzLCJ5Ijo3NX0seyJ4Ijo0ODQuNjU2MjUsInkiOjU5fSx7IngiOjUwOS42NTYyNSwieSI6NTl9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M677.656,59L681.823,59C685.99,59,694.323,59,701.99,59C709.656,59,716.656,59,720.156,59L723.656,59" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6Njc3LjY1NjI1LCJ5Ijo1OX0seyJ4Ijo3MDIuNjU2MjUsInkiOjU5fSx7IngiOjcyNy42NTYyNSwieSI6NTl9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M987.656,59L991.823,59C995.99,59,1004.323,59,1012.027,60.229C1019.73,61.458,1026.804,63.916,1030.341,65.145L1033.878,66.374" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6OTg3LjY1NjI1LCJ5Ijo1OX0seyJ4IjoxMDEyLjY1NjI1LCJ5Ijo1OX0seyJ4IjoxMDM3LjY1NjI1LCJ5Ijo2Ny42ODc0MTcxMzQ5MjAxNH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M1037.656,136.313L1033.49,137.76C1029.323,139.208,1020.99,142.104,990.99,143.552C960.99,145,909.323,145,857.656,145C805.99,145,754.323,145,710.323,145C666.323,145,629.99,145,593.656,145C557.323,145,520.99,145,497.488,142.606C473.987,140.213,463.318,135.425,457.983,133.031L452.649,130.638" id="L_E_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_B_0" data-points="W3sieCI6MTAzNy42NTYyNSwieSI6MTM2LjMxMjU4Mjg2NTA3OTg2fSx7IngiOjEwMTIuNjU2MjUsInkiOjE0NX0seyJ4Ijo4NTcuNjU2MjUsInkiOjE0NX0seyJ4Ijo3MDIuNjU2MjUsInkiOjE0NX0seyJ4Ijo1OTMuNjU2MjUsInkiOjE0NX0seyJ4Ijo0ODQuNjU2MjUsInkiOjE0NX0seyJ4Ijo0NDguOTk5MjczMjU1ODEzOTMsInkiOjEyOX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(138, 102)"><rect class="basic label-container" style="" x="-130" y="-51" width="260" height="102"/><g class="label" style="" transform="translate(-100, -36)"><rect/><foreignObject width="200" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Sample Text<br />Albert Einstein was a German-born</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(388.828125, 102)"><rect class="basic label-container" style="" x="-70.828125" y="-27" width="141.65625" height="54"/><g class="label" style="" transform="translate(-40.828125, -12)"><rect/><foreignObject width="81.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Pass to LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(593.65625, 59)"><rect class="basic label-container" style="" x="-84" y="-39" width="168" height="78"/><g class="label" style="" transform="translate(-54, -24)"><rect/><foreignObject width="108" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Get Probability<br />Distribution</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(857.65625, 59)"><rect class="basic label-container" style="" x="-130" y="-51" width="260" height="102"/><g class="label" style="" transform="translate(-100, -36)"><rect/><foreignObject width="200" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Calculate Loss<br />Compare with correct token</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(1136.3984375, 102)"><rect class="basic label-container" style="" x="-98.7421875" y="-39" width="197.484375" height="78"/><g class="label" style="" transform="translate(-68.7421875, -24)"><rect/><foreignObject width="137.484375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Update Parameters<br />Using Optimizer</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Step-by-Step Process**:

1. **Sample partial text**: "Albert Einstein was a German-born"
2. **Pass to LLM**: Model outputs probability distribution over all tokens
3. **Know correct next token**: From training data, we know "physicist" is correct
4. **Create target vector**: One-hot vector with 1 at "physicist" ID, 0 elsewhere
5. **Calculate loss**: Measure difference between predicted probabilities and target
6. **Update parameters**: Use optimizer to adjust weights
7. **Repeat**: Continue with different text samples

#### Loss Function: Cross-Entropy

**Purpose**: Measure quality of predictions
* Compares predicted probability distribution with correct token
* **High loss**: Predictions far from correct answer
* **Low loss**: Predictions close to correct answer

**Mathematical Formula**:
* Cross-entropy has specific mathematical formulation
* Available on Wikipedia and ML textbooks
* Provides single number representing prediction quality

**Intuition**:
* If model assigns high probability to correct token: Low loss (good)
* If model assigns low probability to correct token: High loss (bad)
* Training goal: Minimize loss over all training examples

#### Optimization Algorithm

**Purpose**: Update model parameters to reduce loss

**Process**:
1. Take calculated loss value
2. Compute gradients (how to adjust each parameter)
3. Update all LLM parameters based on gradients
4. Goal: Next time same input is seen, predictions are more accurate

**Common Optimizers**:
* Adam
* SGD (Stochastic Gradient Descent)
* AdamW
* Others

**Key Point**: Optimization algorithms are well-established
* Same algorithms work across different models
* Core training code is straightforward
* Engineering complexity comes from scale, not algorithm

#### Training Outcome

**Result of Repeated Training**:
* Model learns statistical patterns in Internet data
* Weights adjusted to predict next token accurately
* Example: "Albert Einstein was a German-born" gets high probability for "physicist"

**Statistical Learning**:
* Model learns co-occurrence patterns
* Understands which words typically follow others
* Captures domain-specific terminology relationships

**Implicit Knowledge**:
* Model gains understanding of various domains
* Example: "I like machine learning because" triggers ML-related vocabulary
* Model has learned which terms are contextually relevant
* This is implicit world knowledge from Internet exposure

#### Engineering Challenges at Scale

**Why Training Large LLMs is Difficult**: Not algorithm complexity, but scale

**Memory Requirements** (example: Llama 3 405B parameters):

* **Model parameters**: 405B × 4 bytes (FP32) = 1.6 TB memory
* **Optimizer state**: Additional memory for gradients and momentum
* **Activations**: Intermediate values during forward/backward pass
* **Total estimate**: 2-5+ TB memory needed

**Hardware Constraints**:
* Best GPUs: ~80-100 GB memory
* Single GPU cannot fit large models
* **Minimum**: 16-20 GPUs just to load model
* **Realistic**: 2,000+ GPUs for efficient training

**Storage Requirements**:
* Must save model checkpoints during training
* Each checkpoint: 2-5 TB
* Multiple checkpoints for safety: 10-20+ TB total

**Llama 3 Training Setup** (from technical report):
* **16,000 H100 GPUs** used
* H100 = high-end GPU (most powerful available)
* Distributed across multiple data centers
* Complex infrastructure for networking and storage

**Engineering Solutions Required**:
* **Model parallelism**: Split model across GPUs
* **Data parallelism**: Process different batches on different GPUs
* **Pipeline parallelism**: Split computation stages
* **Memory optimization**: Gradient checkpointing, mixed precision
* **Network optimization**: Fast interconnects between GPUs

**Key Insight from Llama 3 Report**:
* Long section on infrastructure and scaling
* Many tricks and techniques needed
* Core algorithm is simple: Calculate loss, run optimizer
* Complexity is in making it work at scale

**In Practice**:
* Code for loss calculation: Simple (few lines)
* Code for optimization: Standard libraries
* Code for distributed training: Complex (thousands of lines)

### Text Generation

#### From Probabilities to Text

**LLM Output**: Probability distribution over vocabulary
* Example input: "Albert Einstein"
* LLM outputs: Vector of probabilities for next token
* Each value = probability of that token being next

**Challenge**: Need strategy to convert probabilities to actual token
* Cannot just use probabilities directly
* Need algorithm to select which token to generate

**Text Generation as Iterative Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 276px; background-color: transparent;" viewBox="0 0 276 870" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M138,62L138,66.167C138,70.333,138,78.667,138,86.333C138,94,138,101,138,104.5L138,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTM4LCJ5Ijo2Mn0seyJ4IjoxMzgsInkiOjg3fSx7IngiOjEzOCwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,166L138,170.167C138,174.333,138,182.667,138,190.333C138,198,138,205,138,208.5L138,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTM4LCJ5IjoxNjZ9LHsieCI6MTM4LCJ5IjoxOTF9LHsieCI6MTM4LCJ5IjoyMTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,270L138,274.167C138,278.333,138,286.667,138,294.333C138,302,138,309,138,312.5L138,316" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTM4LCJ5IjoyNzB9LHsieCI6MTM4LCJ5IjoyOTV9LHsieCI6MTM4LCJ5IjozMjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,374L138,378.167C138,382.333,138,390.667,138,398.333C138,406,138,413,138,416.5L138,420" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTM4LCJ5IjozNzR9LHsieCI6MTM4LCJ5IjozOTl9LHsieCI6MTM4LCJ5Ijo0MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,478L138,482.167C138,486.333,138,494.667,138,502.333C138,510,138,517,138,520.5L138,524" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MTM4LCJ5Ijo0Nzh9LHsieCI6MTM4LCJ5Ijo1MDN9LHsieCI6MTM4LCJ5Ijo1Mjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,582L138,586.167C138,590.333,138,598.667,138,606.333C138,614,138,621,138,624.5L138,628" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MTM4LCJ5Ijo1ODJ9LHsieCI6MTM4LCJ5Ijo2MDd9LHsieCI6MTM4LCJ5Ijo2MzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,710L138,714.167C138,718.333,138,726.667,138,734.333C138,742,138,749,138,752.5L138,756" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6MTM4LCJ5Ijo3MTB9LHsieCI6MTM4LCJ5Ijo3MzV9LHsieCI6MTM4LCJ5Ijo3NjB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(138, 35)"><rect class="basic label-container" style="" x="-75.6875" y="-27" width="151.375" height="54"/><g class="label" style="" transform="translate(-45.6875, -12)"><rect/><foreignObject width="91.375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input: Albert</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(138, 139)"><rect class="basic label-container" style="" x="-125.8046875" y="-27" width="251.609375" height="54"/><g class="label" style="" transform="translate(-95.8046875, -12)"><rect/><foreignObject width="191.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM produces probabilities</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(138, 243)"><rect class="basic label-container" style="" x="-125.78125" y="-27" width="251.5625" height="54"/><g class="label" style="" transform="translate(-95.78125, -12)"><rect/><foreignObject width="191.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Algorithm selects: Einstein</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(138, 347)"><rect class="basic label-container" style="" x="-124.34375" y="-27" width="248.6875" height="54"/><g class="label" style="" transform="translate(-94.34375, -12)"><rect/><foreignObject width="188.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>New input: Albert Einstein</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(138, 451)"><rect class="basic label-container" style="" x="-125.8046875" y="-27" width="251.609375" height="54"/><g class="label" style="" transform="translate(-95.8046875, -12)"><rect/><foreignObject width="191.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>LLM produces probabilities</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(138, 555)"><rect class="basic label-container" style="" x="-110.8125" y="-27" width="221.625" height="54"/><g class="label" style="" transform="translate(-80.8125, -12)"><rect/><foreignObject width="161.625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Algorithm selects: was</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(138, 671)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>New input: Albert Einstein was</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(138, 811)"><rect class="basic label-container" style="" x="-111.1640625" y="-51" width="222.328125" height="102"/><g class="label" style="" transform="translate(-81.1640625, -36)"><rect/><foreignObject width="162.328125" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Continue until<br />end-of-sentence token<br />or max length</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Stopping Conditions**:
1. **End-of-sentence token**: Special token indicating completion
2. **Maximum length**: Pre-set limit on number of tokens

#### Decoding Algorithms: Two Categories

**Deterministic vs. Stochastic**:

* **Deterministic**: No randomness
  * Same input + same model = same output (always)
  * Repeatable generations
  * Examples: Greedy search, Beam search

* **Stochastic**: Includes randomness
  * Same input + same model = different outputs (each run)
  * Diverse generations
  * Examples: Multinomial sampling, Top-K, Top-P

#### Deterministic Algorithms

**1. Greedy Search**

**Algorithm**: Always pick highest probability token

**Example**:
* Input: "how"
* Probabilities: {are: 37%, is: 21%, do: 18%, ...}
* Greedy choice: "are" (highest at 37%)

**Visualization as Tree**:

```
Input: "how"
├─ are (56%) ✓ SELECTED
│  └─ you (91%) ✓
│     └─ doing (78%) ✓
├─ is (23%)
└─ do (21%)
```

* At each step, follow highest probability branch
* Final path: "how are you doing"

**Advantages**:
* Simple to implement
* Computationally efficient
* Fast generation

**Limitations**:
1. **No look-ahead**: Only considers current step probability
   * May miss better overall sequences in other branches
   * Locally optimal ≠ globally optimal

2. **Repetitive outputs**: Major problem in practice
   * Some token sequences have high probability
   * Greedy search keeps selecting same sequences
   * Results in text with repeated phrases

**Practical Demonstration** (GPT-2 model):

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "I enjoy walking with my cute dog"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_new_tokens=40)
# Result: "I enjoy walking with my cute dog, but I'm not sure
# if I'll ever be able to walk with my dog. I'm not sure
# if I'll ever be able to walk with my dog."
```

**Observation**: Clear repetition of "I'm not sure if I'll ever be able to walk with my dog"
* This sequence has high probability
* Greedy search deterministically selects it repeatedly
* **Not used in modern LLM production systems**

**2. Beam Search**

**Algorithm**: Keep track of top K paths (beams)

**How it Works** (K=3 example):

```
Step 1: Input "how"
├─ come (24%)  ✓ Keep
├─ are (31%)   ✓ Keep
└─ do (26%)    ✓ Keep
   └─ [discard all other options]

Step 2: Expand 3 paths
how come → animals (36%), ...
how are → you (91%), ...
how do → you (63%), ...

Calculate cumulative probabilities:
- how come animals: 24% × 36% = 8.64%
- how are you: 31% × 91% = 28.21% ✓
- how do you: 26% × 63% = 16.38% ✓

Keep top 3 cumulative probability paths
Continue until end...
```

**Advantages**:
* Explores multiple possibilities
* May find better overall sequences than greedy
* Sometimes outperforms greedy search

**Limitations**:
* Still suffers from repetition issues
* Fixed beam width may miss good alternatives
* More computation than greedy (K times more)
* **Not used in modern production LLMs**

**When to Use**:
* Historical interest mainly
* Some specialized applications
* Generally replaced by stochastic methods

#### Stochastic Algorithms

**1. Multinomial Sampling (Basic Randomness)**

**Algorithm**: Sample token according to probability distribution

**Example**:
* Probabilities: {are: 37%, is: 21%, do: 18%, hard: 5%, ...}
* Sample from distribution:
  * 37% chance of selecting "are"
  * 21% chance of selecting "is"
  * 5% chance of selecting "hard"

**Advantages**:
* Introduces diversity in generations
* Can discover different continuations
* Avoids deterministic repetition

**Critical Limitation**: **May sample very unlikely tokens**
* With enough runs, might select grammatically incorrect tokens
* Low-probability tokens can derail generation
* Example: "how hard" might be grammatically odd
* **Not used in production due to quality issues**

**2. Top-K Sampling (Improved Multinomial)**

**Algorithm**:
1. Keep only top K highest probability tokens
2. Discard all other tokens
3. Sample from remaining K tokens according to their probabilities

**Example** (K=3):
* Original: {are: 37%, is: 21%, do: 18%, hard: 5%, ...}
* After filtering: {are: 37%, is: 21%, do: 18%}
* Sample only from these 3 tokens

**Advantages**:
* Prevents sampling very unlikely tokens
* Maintains diversity among likely candidates
* Better quality than pure multinomial

**Limitation**: **Fixed K is not adaptive to probability distribution shape**

**Problem Illustration**:

Case 1: Model is confident
* Probabilities: {lot: 89%, much: 5%, high: 3%, ...}
* Model strongly prefers "lot"
* Top-K=3 still considers "much" and "high" (unnecessary)

Case 2: Model is uncertain
* Probabilities: {are: 31%, is: 29%, do: 22%, come: 11%, ...}
* Multiple good options
* Top-K=3 only considers 3, missing other reasonable choices

**Issue**: Fixed K doesn't adapt to model's confidence level

**3. Top-P (Nucleus) Sampling (Modern Standard)**

**Algorithm**:
1. Sort tokens by probability (highest to lowest)
2. Keep tokens until cumulative probability exceeds P
3. Discard remaining tokens
4. Sample from kept tokens

**Example** (P=0.88):

Case 1: Confident model
* Input: "Thanks a"
* Probabilities: {lot: 89%, much: 5%, high: 3%, ...}
* Cumulative: lot (89%) > 88% threshold
* **Keep only "lot"** (model is confident)

Case 2: Uncertain model
* Input: "how"
* Probabilities: {are: 31%, is: 29%, do: 22%, come: 11%, ...}
* Cumulative: are (31%) < 88%, are+is (60%) < 88%, are+is+do (82%) < 88%, are+is+do+come (93%) > 88%
* **Keep: are, is, do, come** (model uncertain, keep more options)

**Key Advantage**: **Adaptive K based on probability distribution**
* When model confident: Small K (few tokens)
* When model uncertain: Large K (many tokens)
* Adapts to each generation step

**Why Top-P is Production Standard**:
* Balances quality and diversity
* Adapts to model confidence
* Prevents both repetition and low-quality tokens
* **Used in all modern chatbot services**

**Practical Demonstration** (GPT-2 with Top-P):

```python
input_text = "I enjoy walking with my cute dog"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    do_sample=True,      # Enable stochastic sampling
    top_p=0.92,          # Use Top-P with P=0.92
    max_new_tokens=40
)

# Run 1: "I enjoy walking with my cute dog, RM65 who is on a
# leash at home."

# Run 2 (same input): "I enjoy walking with my cute dog. I don't
# know what that is. I love walking with my baby dog and having
# an empty apartment without dogs and other convenience."
```

**Observations**:
* No repetition issues
* Different output each run (stochastic)
* Quality depends on base model (GPT-2 not strongest)
* With better models (GPT-3, GPT-4), much better continuations

#### Hyperparameters and Task-Specific Tuning

**Top-P Value (P)**:
* Controls diversity vs. quality trade-off
* **Low P (e.g., 0.1)**: Very conservative, high quality
* **High P (e.g., 0.95)**: More diverse, more creative

**Temperature**:
* Another hyperparameter that smooths probability distribution
* **Low temperature (0.1-0.5)**: Sharpens distribution, more deterministic
* **High temperature (1.5-2.0)**: Flattens distribution, more random

**Task-Specific Recommendations** (empirical guidelines):

| Task | Top-P | Temperature | Reasoning |
|------|-------|-------------|-----------|
| Code generation | 0.1 | Low | Need syntax correctness, not creativity |
| Factual QA | 0.3-0.5 | Low | Want accurate, reliable answers |
| General conversation | 0.9 | Medium | Balance quality and engagement |
| Creative writing | 0.95 | High | Want novelty and diverse ideas |

**Code Generation Example**:
* Want syntactically correct code
* Don't want creative/novel syntax
* Low P ensures only highly probable tokens
* Reduces risk of syntax errors

**Creative Writing Example**:
* Want interesting, varied language
* Okay with less common word choices
* High P allows more diverse vocabulary
* Increases novelty and creativity

**Important**: These are empirical guidelines, not strict rules
* Different models may have different optimal values
* Task specifics matter
* Experimentation recommended

## Post-Training Stage

### The Problem with Base Models

**Base Model Characteristics** (after pre-training only):
* Excellent at next token prediction
* Strong implicit knowledge of world
* Understands domain-specific terminology
* **But**: Only continues text, does not answer questions

**Practical Demonstration** (GPT-2 base model):

```python
input_text = "I enjoy walking with my cute dog"
# Output: "I enjoy walking with my cute dog, my wife, and my friend.
# They always have me covered with a blanket..."
# → Continuation, not useful for Q&A

input_text = "I like machine learning because"
# Output: "I like machine learning because it's easier to understand,
# and you can use it to predict what people are going to say..."
# → Uses ML terminology correctly, but just continues

input_text = "How is the weather"
# Output: "How is the weather? Weather is one of the most common
# problems for people who are tired and stressed."
# → Does NOT answer the question, just continues
```

**Observations**:
* Model has domain knowledge (uses ML terms correctly)
* Contextually relevant continuations
* But fundamentally wrong behavior for chatbot use case

**Larger Base Models Perform Better** (Llama 3.1 405B base):

```
Input: "I like machine learning because"
Output: "I like machine learning because it's a set of tools that
can be applied to a variety of problems. You can use machine learning
to predict the price of a house or the probability that someone will
click on an ad. You can also use it to identify faces..."
```

* Much more coherent and detailed
* Demonstrates deeper understanding
* Still just continuation, not question-answering

```
Input: "In quantum computing"
Output: "In quantum computing, we often want to construct a circuit
to perform a particular operation. One way to do this is to design
a circuit directly..."
```

* Sophisticated domain knowledge
* Proper terminology usage
* But still not answering questions

**Key Insight**: Base models have knowledge but wrong output format
* Need to adapt from "completion engine" to "question-answering system"
* This is the purpose of post-training

### Post-Training Overview

**Two Steps**:
1. **Supervised Fine-Tuning (SFT)**: Adapt format from completion to Q&A
2. **Reinforcement Learning (RL)**: Improve quality, safety, helpfulness


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1279px; background-color: transparent;" viewBox="0 0 1279 244" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M127.161,130L134.764,125.833C142.368,121.667,157.574,113.333,171.647,109.167C185.719,105,198.656,105,205.125,105L211.594,105" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTI3LjE2MDc1NzIxMTUzODQ1LCJ5IjoxMzB9LHsieCI6MTcyLjc4MTI1LCJ5IjoxMDV9LHsieCI6MjE1LjU5Mzc1LCJ5IjoxMDV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M344.719,105L351.854,105C358.99,105,373.26,105,383.896,105C394.531,105,401.531,105,405.031,105L408.531,105" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzQ0LjcxODc1LCJ5IjoxMDV9LHsieCI6Mzg3LjUzMTI1LCJ5IjoxMDV9LHsieCI6NDEyLjUzMTI1LCJ5IjoxMDV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M521.054,78L529.17,72.833C537.286,67.667,553.518,57.333,570.928,52.167C588.339,47,606.927,47,616.221,47L625.516,47" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTIxLjA1MzYwOTkxMzc5MzEsInkiOjc4fSx7IngiOjU2OS43NSwieSI6NDd9LHsieCI6NjI5LjUxNTYyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M750.359,47L760.32,47C770.281,47,790.203,47,803.664,47C817.125,47,824.125,47,827.625,47L831.125,47" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NzUwLjM1OTM3NSwieSI6NDd9LHsieCI6ODEwLjEyNSwieSI6NDd9LHsieCI6ODM1LjEyNSwieSI6NDd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M127.161,184L134.764,188.167C142.368,192.333,157.574,200.667,168.678,204.833C179.781,209,186.781,209,190.281,209L193.781,209" id="L_A_A1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_A1_0" data-points="W3sieCI6MTI3LjE2MDc1NzIxMTUzODQ1LCJ5IjoxODR9LHsieCI6MTcyLjc4MTI1LCJ5IjoyMDl9LHsieCI6MTk3Ljc4MTI1LCJ5IjoyMDl9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M521.054,132L529.17,137.167C537.286,142.333,553.518,152.667,565.134,157.833C576.75,163,583.75,163,587.25,163L590.75,163" id="L_C_C1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_C1_0" data-points="W3sieCI6NTIxLjA1MzYwOTkxMzc5MzEsInkiOjEzMn0seyJ4Ijo1NjkuNzUsInkiOjE2M30seyJ4Ijo1OTQuNzUsInkiOjE2M31d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M977.078,47L981.245,47C985.411,47,993.745,47,1001.411,47C1009.078,47,1016.078,47,1019.578,47L1023.078,47" id="L_E_E1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_E1_0" data-points="W3sieCI6OTc3LjA3ODEyNSwieSI6NDd9LHsieCI6MTAwMi4wNzgxMjUsInkiOjQ3fSx7IngiOjEwMjcuMDc4MTI1LCJ5Ijo0N31d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A_A1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_C1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_E1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(77.890625, 157)"><rect class="basic label-container" style="" x="-69.890625" y="-27" width="139.78125" height="54"/><g class="label" style="" transform="translate(-39.890625, -12)"><rect/><foreignObject width="79.78125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Base Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(280.15625, 105)"><rect class="basic label-container" style="" x="-64.5625" y="-27" width="129.125" height="54"/><g class="label" style="" transform="translate(-34.5625, -12)"><rect/><foreignObject width="69.125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Stage</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(478.640625, 105)"><rect class="basic label-container" style="" x="-66.109375" y="-27" width="132.21875" height="54"/><g class="label" style="" transform="translate(-36.109375, -12)"><rect/><foreignObject width="72.21875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(689.9375, 47)"><rect class="basic label-container" style="" x="-60.421875" y="-27" width="120.84375" height="54"/><g class="label" style="" transform="translate(-30.421875, -12)"><rect/><foreignObject width="60.84375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>RL Stage</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(906.1015625, 47)"><rect class="basic label-container" style="" x="-70.9765625" y="-27" width="141.953125" height="54"/><g class="label" style="" transform="translate(-40.9765625, -12)"><rect/><foreignObject width="81.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Final Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-A1-9" transform="translate(280.15625, 209)"><rect class="basic label-container" style="" x="-82.375" y="-27" width="164.75" height="54"/><g class="label" style="" transform="translate(-52.375, -12)"><rect/><foreignObject width="104.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Continues text</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C1-11" transform="translate(689.9375, 163)"><rect class="basic label-container" style="" x="-95.1875" y="-39" width="190.375" height="78"/><g class="label" style="" transform="translate(-65.1875, -24)"><rect/><foreignObject width="130.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answers questions<br />but basic quality</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E1-13" transform="translate(1149.0390625, 47)"><rect class="basic label-container" style="" x="-121.9609375" y="-39" width="243.921875" height="78"/><g class="label" style="" transform="translate(-91.9609375, -24)"><rect/><foreignObject width="183.921875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answers questions<br />high quality, safe, helpful</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


### Step 1: Supervised Fine-Tuning (SFT)

**Also Called**: Instruction fine-tuning

**Goal**: Teach model to follow instructions rather than just complete text

**Example Behavior Change**:

Before SFT (base model):
```
Input: "I want to learn ML. What should I do?"
Output: "Is it even easy? I'm not so confident..."
→ Continues with more questions
```

After SFT:
```
Input: "I want to learn ML. What should I do?"
Output: "Take Andrew Ng's course on Coursera."
→ Actually answers the question
```

#### SFT Data Preparation

**Data Format Requirements**: Prompt-response pairs

**Example Format**:
```
[PROMPT] Give three tips for staying healthy
[RESPONSE] 1. Exercise regularly for at least 30 minutes daily
2. Maintain a balanced diet rich in fruits and vegetables
3. Get 7-9 hours of quality sleep each night
```

**Special tokens** mark prompt and response sections

**Data Characteristics**:
* **Manually curated**: Requires human experts/annotators
* **High quality**: Carefully crafted responses
* **Size**: Tens of thousands to hundreds of thousands of examples
* **Much smaller than pre-training data** but higher quality

**Example Datasets**:

1. **Alpaca Dataset** (open source):
   * Format: Instruction + Output columns
   * Examples:
     * "Give three tips for staying healthy" → [three tips]
     * "How can we reduce air pollution?" → [detailed answer]
     * "Describe a time when you had to make a difficult decision" → [response]
   * Available on Hugging Face

2. **InstructGPT Dataset** (OpenAI, not open source):
   * Created by hiring expert annotators
   * ~14,500 prompt-response pairs
   * Used to train GPT-3.5 from GPT-3 base
   * Published in paper: "Training Language Models to Follow Instructions with Human Feedback" (March 2022)
   * Likely foundation for initial ChatGPT release

3. **Other Open Source Datasets**:
   * **Dolly** by Databricks
   * **FLAN** by Google
   * Many domain-specific instruction datasets on Hugging Face
   * Examples: Math-focused, code-focused, etc.

**Data Size Comparison**:
* Pre-training data: Trillions of tokens (entire Internet)
* SFT data: 10K-100K examples
* **Quality over quantity**: SFT data is curated, pre-training data is noisy

#### SFT Training Process

**Key Insight**: **Identical to pre-training algorithm**
* Same loss function (cross-entropy)
* Same optimization algorithms
* Same next-token prediction objective
* **Only difference**: Replace pre-training data with SFT data

**Training Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1182.59px; background-color: transparent;" viewBox="0 0 1182.59375 105" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M226.266,40.778L230.432,39.815C234.599,38.852,242.932,36.926,250.599,35.963C258.266,35,265.266,35,268.766,35L272.266,35" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjI2LjI2NTYyNSwieSI6NDAuNzc3ODU1NDM3MTI1MDU1fSx7IngiOjI1MS4yNjU2MjUsInkiOjM1fSx7IngiOjI3Ni4yNjU2MjUsInkiOjM1fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M470.156,35L474.323,35C478.49,35,486.823,35,494.49,35C502.156,35,509.156,35,512.656,35L516.156,35" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDcwLjE1NjI1LCJ5IjozNX0seyJ4Ijo0OTUuMTU2MjUsInkiOjM1fSx7IngiOjUyMC4xNTYyNSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M715.094,35L719.26,35C723.427,35,731.76,35,739.427,35C747.094,35,754.094,35,757.594,35L761.094,35" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NzE1LjA5Mzc1LCJ5IjozNX0seyJ4Ijo3NDAuMDkzNzUsInkiOjM1fSx7IngiOjc2NS4wOTM3NSwieSI6MzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M927.109,35L931.276,35C935.443,35,943.776,35,951.463,35.882C959.149,36.764,966.189,38.527,969.709,39.409L973.229,40.291" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6OTI3LjEwOTM3NSwieSI6MzV9LHsieCI6OTUyLjEwOTM3NSwieSI6MzV9LHsieCI6OTc3LjEwOTM3NSwieSI6NDEuMjYzMDIxNjU1NDA3NTR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M977.109,90.737L972.943,91.781C968.776,92.825,960.443,94.912,938.608,95.956C916.773,97,881.438,97,846.102,97C810.766,97,775.43,97,737.35,97C699.271,97,658.448,97,617.625,97C576.802,97,535.979,97,495.243,97C454.508,97,413.859,97,373.211,97C332.563,97,291.914,97,268.073,96.187C244.231,95.374,237.197,93.749,233.68,92.936L230.163,92.123" id="L_E_A_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_A_0" data-points="W3sieCI6OTc3LjEwOTM3NSwieSI6OTAuNzM2OTc4MzQ0NTkyNDZ9LHsieCI6OTUyLjEwOTM3NSwieSI6OTd9LHsieCI6ODQ2LjEwMTU2MjUsInkiOjk3fSx7IngiOjc0MC4wOTM3NSwieSI6OTd9LHsieCI6NjE3LjYyNSwieSI6OTd9LHsieCI6NDk1LjE1NjI1LCJ5Ijo5N30seyJ4IjozNzMuMjEwOTM3NSwieSI6OTd9LHsieCI6MjUxLjI2NTYyNSwieSI6OTd9LHsieCI6MjI2LjI2NTYyNSwieSI6OTEuMjIyMTQ0NTYyODc0OTR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_A_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(117.1328125, 66)"><rect class="basic label-container" style="" x="-109.1328125" y="-27" width="218.265625" height="54"/><g class="label" style="" transform="translate(-79.1328125, -12)"><rect/><foreignObject width="158.265625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Sample from SFT Data</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(373.2109375, 35)"><rect class="basic label-container" style="" x="-96.9453125" y="-27" width="193.890625" height="54"/><g class="label" style="" transform="translate(-66.9453125, -12)"><rect/><foreignObject width="133.890625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Pass to Base Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(617.625, 35)"><rect class="basic label-container" style="" x="-97.46875" y="-27" width="194.9375" height="54"/><g class="label" style="" transform="translate(-67.46875, -12)"><rect/><foreignObject width="134.9375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Predict Next Token</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(846.1015625, 35)"><rect class="basic label-container" style="" x="-81.0078125" y="-27" width="162.015625" height="54"/><g class="label" style="" transform="translate(-51.0078125, -12)"><rect/><foreignObject width="102.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Calculate Loss</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(1075.8515625, 66)"><rect class="basic label-container" style="" x="-98.7421875" y="-27" width="197.484375" height="54"/><g class="label" style="" transform="translate(-68.7421875, -12)"><rect/><foreignObject width="137.484375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Update Parameters</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Example**:
* Sample: "What is the capital of France? [RESPONSE] Paris is the capital..."
* Model learns to predict tokens in response section
* Over many examples, learns to answer rather than continue

**Training Details**:
* Start from base model parameters (not random initialization)
* Continue training on SFT data
* Run for enough iterations until model learns instruction-following
* Result: **SFT model** that answers questions

**Important**: No code changes needed from pre-training
* Same training script
* Just different data source
* Much faster than pre-training (days vs. months)

#### SFT Limitations

**Problem**: SFT model answers questions, but quality varies widely

**Example Comparison** (same question, different quality):

Response A:
```
"Take Andrew Ng's course on Coursera."
```
* Correct, contextually relevant
* But minimal, not very helpful

Response B:
```
"Start with a solid foundation in Python, linear algebra, probability,
and statistics. Then take a beginner-friendly ML course like Andrew Ng's
on Coursera. Practice by building small projects and exploring real datasets."
```
* Much more detailed
* More helpful and actionable
* Better overall response

**Both are valid SFT outputs**, but B is clearly superior

**Another Example**: "What are effective ways to reduce stress?"

Possible SFT outputs:
1. "Go skydiving for an adrenaline rush." (Not accurate)
2. "Exercise regularly and maintain a healthy diet." (Good, helpful)
3. "Shame on you, try meditation." (Rude, unsafe)
4. "Ignore your problems and hope they go away." (Not helpful)

**Issue**: SFT model can produce any of these
* All are grammatically correct
* All are contextually relevant (answer the question)
* But quality, safety, helpfulness vary greatly

**Solution**: Need reinforcement learning to prefer better responses

### Step 2: Reinforcement Learning (RL)

**Goal**: Generate responses that are:
* More correct and accurate
* More helpful and detailed
* More safe and polite
* Aligned with human preferences

**High-Level Approach**: **Practicing algorithm**


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1250.31px; background-color: transparent;" viewBox="0 0 1250.31298828125 638" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M900.73,62L900.73,66.167C900.73,70.333,900.73,78.667,900.73,86.333C900.73,94,900.73,101,900.73,104.5L900.73,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6OTAwLjczMDQ2ODc1LCJ5Ijo2Mn0seyJ4Ijo5MDAuNzMwNDY4NzUsInkiOjg3fSx7IngiOjkwMC43MzA0Njg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M834.621,149.064L788.709,156.053C742.797,163.043,650.973,177.021,605.061,187.511C559.148,198,559.148,205,559.148,208.5L559.148,212" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6ODM0LjYyMTA5Mzc1LCJ5IjoxNDkuMDY0MDE3MzgyMzU0NjJ9LHsieCI6NTU5LjE0ODQzNzUsInkiOjE5MX0seyJ4Ijo1NTkuMTQ4NDM3NSwieSI6MjE2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M429.148,273.065L374.055,280.721C318.961,288.377,208.773,303.688,153.68,314.844C98.586,326,98.586,333,98.586,336.5L98.586,340" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NDI5LjE0ODQzNzUsInkiOjI3My4wNjQ4NjYzMzE5MzEwNn0seyJ4Ijo5OC41ODU5Mzc1LCJ5IjozMTl9LHsieCI6OTguNTg1OTM3NSwieSI6MzQ0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M429.148,292.078L413.417,296.565C397.685,301.052,366.221,310.026,350.49,318.013C334.758,326,334.758,333,334.758,336.5L334.758,340" id="L_C_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_E_0" data-points="W3sieCI6NDI5LjE0ODQzNzUsInkiOjI5Mi4wNzgxOTc4OTcwODIzNn0seyJ4IjozMzQuNzU3ODEyNSwieSI6MzE5fSx7IngiOjMzNC43NTc4MTI1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M689.148,288.964L708.31,293.97C727.471,298.976,765.794,308.988,784.956,317.494C804.117,326,804.117,333,804.117,336.5L804.117,340" id="L_C_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_F_0" data-points="W3sieCI6Njg5LjE0ODQzNzUsInkiOjI4OC45NjM1MTU3NTQ1NjA1NX0seyJ4Ijo4MDQuMTE3MTg3NSwieSI6MzE5fSx7IngiOjgwNC4xMTcxODc1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M689.148,270.709L755.757,278.757C822.365,286.806,955.581,302.903,1022.189,314.451C1088.797,326,1088.797,333,1088.797,336.5L1088.797,340" id="L_C_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_G_0" data-points="W3sieCI6Njg5LjE0ODQzNzUsInkiOjI3MC43MDg1MzMwNzc2NjA2fSx7IngiOjEwODguNzk2ODc1LCJ5IjozMTl9LHsieCI6MTA4OC43OTY4NzUsInkiOjM0NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M559.148,294L559.148,298.167C559.148,302.333,559.148,310.667,559.148,318.333C559.148,326,559.148,333,559.148,336.5L559.148,340" id="L_C_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_H_0" data-points="W3sieCI6NTU5LjE0ODQzNzUsInkiOjI5NH0seyJ4Ijo1NTkuMTQ4NDM3NSwieSI6MzE5fSx7IngiOjU1OS4xNDg0Mzc1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M98.586,398L98.586,402.167C98.586,406.333,98.586,414.667,126.864,425.215C155.142,435.763,211.698,448.526,239.976,454.907L268.254,461.288" id="L_D_I_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_I_0" data-points="W3sieCI6OTguNTg1OTM3NSwieSI6Mzk4fSx7IngiOjk4LjU4NTkzNzUsInkiOjQyM30seyJ4IjoyNzIuMTU2MjUsInkiOjQ2Mi4xNjg4MDgyNzI1ODg2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M334.758,398L334.758,402.167C334.758,406.333,334.758,414.667,334.371,422.337C333.984,430.008,333.21,437.016,332.823,440.52L332.436,444.024" id="L_E_I_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_I_0" data-points="W3sieCI6MzM0Ljc1NzgxMjUsInkiOjM5OH0seyJ4IjozMzQuNzU3ODEyNSwieSI6NDIzfSx7IngiOjMzMS45OTcxNDU0MzI2OTIzLCJ5Ijo0NDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M559.148,398L559.148,402.167C559.148,406.333,559.148,414.667,530.92,425.212C502.691,435.757,446.234,448.514,418.005,454.892L389.777,461.271" id="L_H_I_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_H_I_0" data-points="W3sieCI6NTU5LjE0ODQzNzUsInkiOjM5OH0seyJ4Ijo1NTkuMTQ4NDM3NSwieSI6NDIzfSx7IngiOjM4NS44NzUsInkiOjQ2Mi4xNTIyNTU4MzA1MzI2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M804.117,398L804.117,402.167C804.117,406.333,804.117,414.667,817.572,423.955C831.026,433.244,857.935,443.487,871.389,448.609L884.844,453.731" id="L_F_J_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_J_0" data-points="W3sieCI6ODA0LjExNzE4NzUsInkiOjM5OH0seyJ4Ijo4MDQuMTE3MTg3NSwieSI6NDIzfSx7IngiOjg4OC41ODIwMzEyNSwieSI6NDU1LjE1NDA3OTMyNzQwNDI1fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M1088.797,398L1088.797,402.167C1088.797,406.333,1088.797,414.667,1073.434,424.228C1058.072,433.789,1027.347,444.579,1011.984,449.973L996.622,455.368" id="L_G_J_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_J_0" data-points="W3sieCI6MTA4OC43OTY4NzUsInkiOjM5OH0seyJ4IjoxMDg4Ljc5Njg3NSwieSI6NDIzfSx7IngiOjk5Mi44NDc2NTYyNSwieSI6NDU2LjY5MzIxMjY5MzU1NTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M329.016,502L329.016,506.167C329.016,510.333,329.016,518.667,408.636,531.164C488.256,543.661,647.496,560.322,727.116,568.652L806.737,576.982" id="L_I_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_I_K_0" data-points="W3sieCI6MzI5LjAxNTYyNSwieSI6NTAyfSx7IngiOjMyOS4wMTU2MjUsInkiOjUyN30seyJ4Ijo4MTAuNzE0ODQzNzUsInkiOjU3Ny4zOTg1NDQwMTQ4MTUzfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M940.715,502L940.715,506.167C940.715,510.333,940.715,518.667,940.715,526.333C940.715,534,940.715,541,940.715,544.5L940.715,548" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6OTQwLjcxNDg0Mzc1LCJ5Ijo1MDJ9LHsieCI6OTQwLjcxNDg0Mzc1LCJ5Ijo1Mjd9LHsieCI6OTQwLjcxNDg0Mzc1LCJ5Ijo1NTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M1070.715,563.414L1099.314,557.345C1127.914,551.276,1185.113,539.138,1213.713,524.402C1242.313,509.667,1242.313,492.333,1242.313,475C1242.313,457.667,1242.313,440.333,1242.313,423C1242.313,405.667,1242.313,388.333,1242.313,371C1242.313,353.667,1242.313,336.333,1242.313,317C1242.313,297.667,1242.313,276.333,1242.313,255C1242.313,233.667,1242.313,212.333,1197.059,194.778C1151.806,177.222,1061.3,163.444,1016.047,156.555L970.794,149.666" id="L_K_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_K_B_0" data-points="W3sieCI6MTA3MC43MTQ4NDM3NSwieSI6NTYzLjQxMzU3ODcyNzg2ODV9LHsieCI6MTI0Mi4zMTI1LCJ5Ijo1Mjd9LHsieCI6MTI0Mi4zMTI1LCJ5Ijo0NzV9LHsieCI6MTI0Mi4zMTI1LCJ5Ijo0MjN9LHsieCI6MTI0Mi4zMTI1LCJ5IjozNzF9LHsieCI6MTI0Mi4zMTI1LCJ5IjozMTl9LHsieCI6MTI0Mi4zMTI1LCJ5IjoyNTV9LHsieCI6MTI0Mi4zMTI1LCJ5IjoxOTF9LHsieCI6OTY2LjgzOTg0Mzc1LCJ5IjoxNDkuMDY0MDE3MzgyMzU0NjJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_H_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_I_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_K_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(900.73046875, 35)"><rect class="basic label-container" style="" x="-98.375" y="-27" width="196.75" height="54"/><g class="label" style="" transform="translate(-68.375, -12)"><rect/><foreignObject width="136.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input: What is 2+2?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(900.73046875, 139)"><rect class="basic label-container" style="" x="-66.109375" y="-27" width="132.21875" height="54"/><g class="label" style="" transform="translate(-36.109375, -12)"><rect/><foreignObject width="72.21875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(559.1484375, 255)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Generate Multiple Responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(98.5859375, 371)"><rect class="basic label-container" style="" x="-90.5859375" y="-27" width="181.171875" height="54"/><g class="label" style="" transform="translate(-60.5859375, -12)"><rect/><foreignObject width="121.171875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response 1: Four</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(334.7578125, 371)"><rect class="basic label-container" style="" x="-95.5859375" y="-27" width="191.171875" height="54"/><g class="label" style="" transform="translate(-65.5859375, -12)"><rect/><foreignObject width="131.171875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response 2: 2+2=4</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(804.1171875, 371)"><rect class="basic label-container" style="" x="-116.1640625" y="-27" width="232.328125" height="54"/><g class="label" style="" transform="translate(-86.1640625, -12)"><rect/><foreignObject width="172.328125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response 3: Twenty-two</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(1088.796875, 371)"><rect class="basic label-container" style="" x="-118.515625" y="-27" width="237.03125" height="54"/><g class="label" style="" transform="translate(-88.515625, -12)"><rect/><foreignObject width="177.03125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response 4: Math is hard</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(559.1484375, 371)"><rect class="basic label-container" style="" x="-78.8046875" y="-27" width="157.609375" height="54"/><g class="label" style="" transform="translate(-48.8046875, -12)"><rect/><foreignObject width="97.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response 5: 4</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-15" transform="translate(329.015625, 475)"><rect class="basic label-container" style="" x="-56.859375" y="-27" width="113.71875" height="54"/><g class="label" style="" transform="translate(-26.859375, -12)"><rect/><foreignObject width="53.71875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Good ✓</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-21" transform="translate(940.71484375, 475)"><rect class="basic label-container" style="" x="-52.1328125" y="-27" width="104.265625" height="54"/><g class="label" style="" transform="translate(-22.1328125, -12)"><rect/><foreignObject width="44.265625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Bad ✗</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-25" transform="translate(940.71484375, 591)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Update model to prefer these</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Process**:
1. Give SFT model same prompt multiple times
2. Model generates various responses
3. Evaluate which responses are better
4. Update model parameters to prefer better responses
5. Repeat with many prompts

**Key Question**: How do we determine which responses are better?

**Answer depends on task type**: Verifiable vs. Unverifiable

#### Task Categorization: Verifiable vs. Unverifiable

**Verifiable Tasks**: Can automatically verify correctness

**Examples**:
* **Math problems**: "What is 2+2?" → Correct answer is "4"
  * Easy to check if response contains correct final answer
  * Incorrect if final answer is not "4"

* **Coding**: "Write a function to sort a list" → Can run and test code
  * Execute code
  * Check if output matches expected behavior
  * Verify no errors

**Characteristics**:
* Clear correct answer exists
* Can write automated checker
* No human judgment needed

**Unverifiable Tasks**: Cannot easily verify correctness automatically

**Examples**:
* **Creative writing**: "Help me choose a name for my startup"
  * Multiple valid answers
  * Subjective preferences
  * No single "correct" answer

* **Brainstorming**: "Suggest marketing strategies"
  * Many possible good responses
  * Difficult to rank objectively
  * Context-dependent quality

**Characteristics**:
* Subjective evaluation
* Multiple valid answers
* Requires human judgment

**Why This Matters**: Different approaches for RL training
* Verifiable: Can automate response scoring
* Unverifiable: Need human feedback

#### RL for Verifiable Tasks

**Approach**: Automatic verification + RL algorithm

**Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 259.422px; background-color: transparent;" viewBox="0 0 259.421875 814" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M129.711,62L129.711,66.167C129.711,70.333,129.711,78.667,129.711,86.333C129.711,94,129.711,101,129.711,104.5L129.711,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjYyfSx7IngiOjEyOS43MTA5Mzc1LCJ5Ijo4N30seyJ4IjoxMjkuNzEwOTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M129.711,190L129.711,194.167C129.711,198.333,129.711,206.667,129.711,214.333C129.711,222,129.711,229,129.711,232.5L129.711,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjE5MH0seyJ4IjoxMjkuNzEwOTM3NSwieSI6MjE1fSx7IngiOjEyOS43MTA5Mzc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M129.711,318L129.711,322.167C129.711,326.333,129.711,334.667,129.711,342.333C129.711,350,129.711,357,129.711,360.5L129.711,364" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjMxOH0seyJ4IjoxMjkuNzEwOTM3NSwieSI6MzQzfSx7IngiOjEyOS43MTA5Mzc1LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M129.711,446L129.711,450.167C129.711,454.333,129.711,462.667,129.711,470.333C129.711,478,129.711,485,129.711,488.5L129.711,492" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjQ0Nn0seyJ4IjoxMjkuNzEwOTM3NSwieSI6NDcxfSx7IngiOjEyOS43MTA5Mzc1LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M129.711,574L129.711,578.167C129.711,582.333,129.711,590.667,129.711,598.333C129.711,606,129.711,613,129.711,616.5L129.711,620" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjU3NH0seyJ4IjoxMjkuNzEwOTM3NSwieSI6NTk5fSx7IngiOjEyOS43MTA5Mzc1LCJ5Ijo2MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M129.711,702L129.711,706.167C129.711,710.333,129.711,718.667,129.711,726.333C129.711,734,129.711,741,129.711,744.5L129.711,748" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MTI5LjcxMDkzNzUsInkiOjcwMn0seyJ4IjoxMjkuNzEwOTM3NSwieSI6NzI3fSx7IngiOjEyOS43MTA5Mzc1LCJ5Ijo3NTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(129.7109375, 35)"><rect class="basic label-container" style="" x="-121.7109375" y="-27" width="243.421875" height="54"/><g class="label" style="" transform="translate(-91.7109375, -12)"><rect/><foreignObject width="183.421875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Dataset of Math Problems</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(129.7109375, 151)"><rect class="basic label-container" style="" x="-105.109375" y="-39" width="210.21875" height="78"/><g class="label" style="" transform="translate(-75.109375, -24)"><rect/><foreignObject width="150.21875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Model Generates<br />Multiple Responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(129.7109375, 279)"><rect class="basic label-container" style="" x="-101.1953125" y="-39" width="202.390625" height="78"/><g class="label" style="" transform="translate(-71.1953125, -24)"><rect/><foreignObject width="142.390625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Automated Checker<br />Verifies Correctness</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(129.7109375, 407)"><rect class="basic label-container" style="" x="-89.9765625" y="-39" width="179.953125" height="78"/><g class="label" style="" transform="translate(-59.9765625, -24)"><rect/><foreignObject width="119.953125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Label: Correct ✓<br />or Incorrect ✗</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(129.7109375, 535)"><rect class="basic label-container" style="" x="-89.1796875" y="-39" width="178.359375" height="78"/><g class="label" style="" transform="translate(-59.1796875, -24)"><rect/><foreignObject width="118.359375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>RL Algorithm<br />e.g., PPO, GRPO</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(129.7109375, 663)"><rect class="basic label-container" style="" x="-94.359375" y="-39" width="188.71875" height="78"/><g class="label" style="" transform="translate(-64.359375, -24)"><rect/><foreignObject width="128.71875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Update SFT Model<br />Parameters</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(129.7109375, 779)"><rect class="basic label-container" style="" x="-70.9765625" y="-27" width="141.953125" height="54"/><g class="label" style="" transform="translate(-40.9765625, -12)"><rect/><foreignObject width="81.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Final Model</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Step 1: Generate and Label**
* Have dataset of problems with known answers
* SFT model generates multiple responses per problem
* **Automated checker** (Python code) verifies final answer
* Label each response: Correct or Incorrect

**Example**:
```
Problem: "What is 2+2?"
Response 1: "The answer is four." → Correct ✓
Response 2: "2+2 equals 4." → Correct ✓
Response 3: "It's 22." → Incorrect ✗
Response 4: "Math is hard." → Incorrect ✗
Response 5: "Four." → Correct ✓
```

**Step 2: Apply RL Algorithm**
* Use algorithm like **PPO** (Proximal Policy Optimization) or **GRPO**
* Algorithm takes:
  * Prompt
  * Multiple responses
  * Labels (correct/incorrect)
* Updates model parameters to:
  * **Increase probability** of generating correct responses
  * **Decrease probability** of generating incorrect responses

**Outcome**: Model reinforced to produce correct answers
* Given "What is 2+2?", more likely to generate "4"
* Learns patterns that lead to correct answers

**Advantages**:
* Fully automated (no human annotation needed)
* Scalable to large datasets
* Clear objective (correctness)

#### RL for Unverifiable Tasks: RLHF

**RLHF**: Reinforcement Learning from Human Feedback

**Challenge**: Cannot automatically determine which responses are better

**Solution**: Train a separate model to score responses

**Two Stages**:
1. **Train Reward Model**: Learn to score responses like humans would
2. **Optimize with RL**: Use reward model scores to improve SFT model

##### Stage 1: Training Reward Model

**Step 1: Collect Prompts**
* Gather diverse prompts: "What is the capital of France?", "Name a famous physicist", etc.

**Step 2: Generate Multiple Responses**
* Use SFT model to create multiple responses per prompt

**Example**:
```
Prompt: "What is the capital of France?"
Response 1: "Paris"
Response 2: "It's in Europe"
Response 3: "It's Eiffel Tower"
```

**Step 3: Human Ranking**
* Hire annotators (actual humans)
* Annotators rank responses by quality
* Example ranking: Paris > It's in Europe > It's Eiffel Tower

**Step 4: Create Training Data**
* Convert rankings to pairwise comparisons
* Format: Prompt + Winning Response + Losing Response

**Training Examples**:
```
Prompt: "What is the capital of France?"
Winning: "Paris"
Losing: "It's in Europe"

Prompt: "What is the capital of France?"
Winning: "It's in Europe"
Losing: "It's Eiffel Tower"

Prompt: "What is 2+2?"
Winning: "Four"
Losing: "Math is hard"
```

**Step 5: Train Reward Model**

**Reward Model**:
* Input: Prompt + Response
* Output: Score (single number)

**Training Process**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1130.95px; background-color: transparent;" viewBox="0 0 1130.953125 209" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M264.484,66L268.651,66C272.818,66,281.151,66,295.795,73.286C310.44,80.572,331.395,95.144,341.872,102.43L352.35,109.716" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjY0LjQ4NDM3NSwieSI6NjZ9LHsieCI6Mjg5LjQ4NDM3NSwieSI6NjZ9LHsieCI6MzU1LjYzMzk4OTcyNjAyNzQsInkiOjExMn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M421.714,112L434.668,99.167C447.622,86.333,473.53,60.667,489.984,47.833C506.438,35,513.438,35,516.938,35L520.438,35" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6NDIxLjcxNDQ2ODE0OTAzODQ1LCJ5IjoxMTJ9LHsieCI6NDk5LjQzNzUsInkiOjM1fSx7IngiOjUyNC40Mzc1LCJ5IjozNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M258.336,170L263.527,170C268.719,170,279.102,170,287.82,168.958C296.539,167.917,303.594,165.834,307.121,164.792L310.648,163.75" id="L_D_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_B_0" data-points="W3sieCI6MjU4LjMzNTkzNzUsInkiOjE3MH0seyJ4IjoyODkuNDg0Mzc1LCJ5IjoxNzB9LHsieCI6MzE0LjQ4NDM3NSwieSI6MTYyLjYxNzM5OTcxNzE5ODh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M474.438,139L478.604,139C482.771,139,491.104,139,498.771,139C506.438,139,513.438,139,516.938,139L520.438,139" id="L_B_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_E_0" data-points="W3sieCI6NDc0LjQzNzUsInkiOjEzOX0seyJ4Ijo0OTkuNDM3NSwieSI6MTM5fSx7IngiOjUyNC40Mzc1LCJ5IjoxMzl9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M636.797,35L640.964,35C645.13,35,653.464,35,667.129,38.913C680.794,42.825,699.791,50.651,709.289,54.564L718.788,58.476" id="L_C_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_F_0" data-points="W3sieCI6NjM2Ljc5Njg3NSwieSI6MzV9LHsieCI6NjYxLjc5Njg3NSwieSI6MzV9LHsieCI6NzIyLjQ4NjQ3ODM2NTM4NDYsInkiOjYwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M636.797,139L640.964,139C645.13,139,653.464,139,667.129,135.087C680.794,131.175,699.791,123.349,709.289,119.436L718.788,115.524" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6NjM2Ljc5Njg3NSwieSI6MTM5fSx7IngiOjY2MS43OTY4NzUsInkiOjEzOX0seyJ4Ijo3MjIuNDg2NDc4MzY1Mzg0NiwieSI6MTE0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M889.266,87L893.432,87C897.599,87,905.932,87,913.621,87.935C921.31,88.869,928.355,90.738,931.877,91.673L935.399,92.607" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6ODg5LjI2NTYyNSwieSI6ODd9LHsieCI6OTE0LjI2NTYyNSwieSI6ODd9LHsieCI6OTM5LjI2NTYyNSwieSI6OTMuNjMyNzg5NTE1OTEzMzR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M976.207,157L965.883,164.333C955.56,171.667,934.913,186.333,903.55,193.667C872.188,201,830.109,201,788.031,201C745.953,201,703.875,201,669.306,201C634.737,201,607.677,201,580.617,201C553.557,201,526.497,201,503.665,195.506C480.832,190.011,462.226,179.023,452.923,173.528L443.621,168.034" id="L_G_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_B_0" data-points="W3sieCI6OTc2LjIwNjg5MDA2MDI0MSwieSI6MTU3fSx7IngiOjkxNC4yNjU2MjUsInkiOjIwMX0seyJ4Ijo3ODguMDMxMjUsInkiOjIwMX0seyJ4Ijo2NjEuNzk2ODc1LCJ5IjoyMDF9LHsieCI6NTgwLjYxNzE4NzUsInkiOjIwMX0seyJ4Ijo0OTkuNDM3NSwieSI6MjAxfSx7IngiOjQ0MC4xNzY1MzcyOTgzODcxLCJ5IjoxNjZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(136.2421875, 66)"><rect class="basic label-container" style="" x="-128.2421875" y="-27" width="256.484375" height="54"/><g class="label" style="" transform="translate(-98.2421875, -12)"><rect/><foreignObject width="196.484375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt + Winning Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(394.4609375, 139)"><rect class="basic label-container" style="" x="-79.9765625" y="-27" width="159.953125" height="54"/><g class="label" style="" transform="translate(-49.9765625, -12)"><rect/><foreignObject width="99.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Reward Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(580.6171875, 35)"><rect class="basic label-container" style="" x="-56.1796875" y="-27" width="112.359375" height="54"/><g class="label" style="" transform="translate(-26.1796875, -12)"><rect/><foreignObject width="52.359375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Score 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-4" transform="translate(136.2421875, 170)"><rect class="basic label-container" style="" x="-122.09375" y="-27" width="244.1875" height="54"/><g class="label" style="" transform="translate(-92.09375, -12)"><rect/><foreignObject width="184.1875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt + Losing Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(580.6171875, 139)"><rect class="basic label-container" style="" x="-56.1796875" y="-27" width="112.359375" height="54"/><g class="label" style="" transform="translate(-26.1796875, -12)"><rect/><foreignObject width="52.359375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Score 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(788.03125, 87)"><rect class="basic label-container" style="" x="-101.234375" y="-27" width="202.46875" height="54"/><g class="label" style="" transform="translate(-71.234375, -12)"><rect/><foreignObject width="142.46875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Margin Ranking Loss</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-13" transform="translate(1031.109375, 118)"><rect class="basic label-container" style="" x="-91.84375" y="-39" width="183.6875" height="78"/><g class="label" style="" transform="translate(-61.84375, -24)"><rect/><foreignObject width="123.6875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Maximize Score 1<br />Minimize Score 2</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


* Pass winning response: Get score S1
* Pass losing response: Get score S2
* **Loss function** (Margin Ranking Loss): Tries to maximize (S1 - S2)
* Goal: Winning responses get higher scores than losing responses

**Outcome**: Reward model that scores responses aligned with human preferences

**Reward Model as Proxy**: Replaces human annotators
* Instead of humans ranking each response
* Reward model automatically scores responses
* Should align with human judgments (learned from training data)

##### Stage 2: Optimize SFT Model with RL

**Process**: Very similar to verifiable tasks, but uses reward model


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 896px; background-color: transparent;" viewBox="0 0 896 838" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M448,86L448,90.167C448,94.333,448,102.667,448,110.333C448,118,448,125,448,128.5L448,132" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6NDQ4LCJ5Ijo4Nn0seyJ4Ijo0NDgsInkiOjExMX0seyJ4Ijo0NDgsInkiOjEzNn1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M342.891,196.7L308.742,203.75C274.594,210.8,206.297,224.9,172.148,235.45C138,246,138,253,138,256.5L138,260" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzQyLjg5MDYyNSwieSI6MTk2Ljd9LHsieCI6MTM4LCJ5IjoyMzl9LHsieCI6MTM4LCJ5IjoyNjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448,214L448,218.167C448,222.333,448,230.667,448,238.333C448,246,448,253,448,256.5L448,260" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6NDQ4LCJ5IjoyMTR9LHsieCI6NDQ4LCJ5IjoyMzl9LHsieCI6NDQ4LCJ5IjoyNjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M553.109,196.7L587.258,203.75C621.406,210.8,689.703,224.9,723.852,235.45C758,246,758,253,758,256.5L758,260" id="L_B_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_E_0" data-points="W3sieCI6NTUzLjEwOTM3NSwieSI6MTk2Ljd9LHsieCI6NzU4LCJ5IjoyMzl9LHsieCI6NzU4LCJ5IjoyNjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,342L138,346.167C138,350.333,138,358.667,175.68,369.154C213.36,379.641,288.719,392.282,326.399,398.602L364.079,404.923" id="L_C_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_F_0" data-points="W3sieCI6MTM4LCJ5IjozNDJ9LHsieCI6MTM4LCJ5IjozNjd9LHsieCI6MzY4LjAyMzQzNzUsInkiOjQwNS41ODQ1NzY2MTI5MDMyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448,342L448,346.167C448,350.333,448,358.667,448,366.333C448,374,448,381,448,384.5L448,388" id="L_D_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_F_0" data-points="W3sieCI6NDQ4LCJ5IjozNDJ9LHsieCI6NDQ4LCJ5IjozNjd9LHsieCI6NDQ4LCJ5IjozOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M758,342L758,346.167C758,350.333,758,358.667,720.32,369.154C682.64,379.641,607.281,392.282,569.601,398.602L531.921,404.923" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6NzU4LCJ5IjozNDJ9LHsieCI6NzU4LCJ5IjozNjd9LHsieCI6NTI3Ljk3NjU2MjUsInkiOjQwNS41ODQ1NzY2MTI5MDMyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M368.023,441.788L350.936,446.657C333.849,451.525,299.674,461.263,282.587,469.631C265.5,478,265.5,485,265.5,488.5L265.5,492" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MzY4LjAyMzQzNzUsInkiOjQ0MS43ODc4NDI0NjU3NTM0NH0seyJ4IjoyNjUuNSwieSI6NDcxfSx7IngiOjI2NS41LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448,446L448,450.167C448,454.333,448,462.667,448,470.333C448,478,448,485,448,488.5L448,492" id="L_F_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_H_0" data-points="W3sieCI6NDQ4LCJ5Ijo0NDZ9LHsieCI6NDQ4LCJ5Ijo0NzF9LHsieCI6NDQ4LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M527.977,441.788L545.064,446.657C562.151,451.525,596.326,461.263,613.413,469.631C630.5,478,630.5,485,630.5,488.5L630.5,492" id="L_F_I_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_I_0" data-points="W3sieCI6NTI3Ljk3NjU2MjUsInkiOjQ0MS43ODc4NDI0NjU3NTM0NH0seyJ4Ijo2MzAuNSwieSI6NDcxfSx7IngiOjYzMC41LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M265.5,550L265.5,554.167C265.5,558.333,265.5,566.667,282.73,576.876C299.961,587.085,334.421,599.17,351.651,605.212L368.882,611.254" id="L_G_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_J_0" data-points="W3sieCI6MjY1LjUsInkiOjU1MH0seyJ4IjoyNjUuNSwieSI6NTc1fSx7IngiOjM3Mi42NTYyNSwieSI6NjEyLjU3ODA4MjE5MTc4MDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448,550L448,554.167C448,558.333,448,566.667,448,574.333C448,582,448,589,448,592.5L448,596" id="L_H_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_H_J_0" data-points="W3sieCI6NDQ4LCJ5Ijo1NTB9LHsieCI6NDQ4LCJ5Ijo1NzV9LHsieCI6NDQ4LCJ5Ijo2MDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M630.5,550L630.5,554.167C630.5,558.333,630.5,566.667,613.27,576.876C596.039,587.085,561.579,599.17,544.349,605.212L527.118,611.254" id="L_I_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_I_J_0" data-points="W3sieCI6NjMwLjUsInkiOjU1MH0seyJ4Ijo2MzAuNSwieSI6NTc1fSx7IngiOjUyMy4zNDM3NSwieSI6NjEyLjU3ODA4MjE5MTc4MDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M448,678L448,682.167C448,686.333,448,694.667,448,702.333C448,710,448,717,448,720.5L448,724" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6NDQ4LCJ5Ijo2Nzh9LHsieCI6NDQ4LCJ5Ijo3MDN9LHsieCI6NDQ4LCJ5Ijo3Mjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_I_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_H_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_I_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(448, 47)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Prompt: How should I learn ML?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(448, 175)"><rect class="basic label-container" style="" x="-105.109375" y="-39" width="210.21875" height="78"/><g class="label" style="" transform="translate(-75.109375, -24)"><rect/><foreignObject width="150.21875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Model Generates<br />Multiple Responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(138, 303)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Response 1: Start with Python...</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(448, 303)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Response 2: Take a course...</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(758, 303)"><rect class="basic label-container" style="" x="-130" y="-39" width="260" height="78"/><g class="label" style="" transform="translate(-100, -24)"><rect/><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Response 3: Just Google it...</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(448, 419)"><rect class="basic label-container" style="" x="-79.9765625" y="-27" width="159.953125" height="54"/><g class="label" style="" transform="translate(-49.9765625, -12)"><rect/><foreignObject width="99.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Reward Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-15" transform="translate(265.5, 523)"><rect class="basic label-container" style="" x="-66.25" y="-27" width="132.5" height="54"/><g class="label" style="" transform="translate(-36.25, -12)"><rect/><foreignObject width="72.5" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Score: 8.5</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-17" transform="translate(448, 523)"><rect class="basic label-container" style="" x="-66.25" y="-27" width="132.5" height="54"/><g class="label" style="" transform="translate(-36.25, -12)"><rect/><foreignObject width="72.5" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Score: 7.2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-19" transform="translate(630.5, 523)"><rect class="basic label-container" style="" x="-66.25" y="-27" width="132.5" height="54"/><g class="label" style="" transform="translate(-36.25, -12)"><rect/><foreignObject width="72.5" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Score: 3.1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-21" transform="translate(448, 639)"><rect class="basic label-container" style="" x="-75.34375" y="-39" width="150.6875" height="78"/><g class="label" style="" transform="translate(-45.34375, -24)"><rect/><foreignObject width="90.6875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>RL Algorithm<br />e.g., PPO</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-27" transform="translate(448, 779)"><rect class="basic label-container" style="" x="-130" y="-51" width="260" height="102"/><g class="label" style="" transform="translate(-100, -36)"><rect/><foreignObject width="200" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Update SFT Model<br />to prefer high-scoring responses</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Steps**:
1. Give prompt to SFT model
2. Generate multiple responses
3. **Reward model scores each response**
4. RL algorithm (PPO) updates SFT model parameters
5. Goal: Generate responses with higher reward scores

**Key Difference from Verifiable**:
* Verifiable: Automated checker determines correctness
* Unverifiable: Reward model determines quality

**Outcome**: **Final model** that produces responses aligned with human preferences
* More detailed and helpful
* More accurate and correct
* Safer and more polite

#### Complete Post-Training Summary


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 478.859px; background-color: transparent;" viewBox="0 0 478.859375 1151.640625" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M237.66,62L237.66,66.167C237.66,70.333,237.66,78.667,237.66,86.333C237.66,94,237.66,101,237.66,104.5L237.66,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MjM3LjY2MDE1NjI1LCJ5Ijo2Mn0seyJ4IjoyMzcuNjYwMTU2MjUsInkiOjg3fSx7IngiOjIzNy42NjAxNTYyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M237.66,190L237.66,194.167C237.66,198.333,237.66,206.667,237.66,214.333C237.66,222,237.66,229,237.66,232.5L237.66,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjM3LjY2MDE1NjI1LCJ5IjoxOTB9LHsieCI6MjM3LjY2MDE1NjI1LCJ5IjoyMTV9LHsieCI6MjM3LjY2MDE1NjI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M237.66,294L237.66,298.167C237.66,302.333,237.66,310.667,237.66,318.333C237.66,326,237.66,333,237.66,336.5L237.66,340" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MjM3LjY2MDE1NjI1LCJ5IjoyOTR9LHsieCI6MjM3LjY2MDE1NjI1LCJ5IjozMTl9LHsieCI6MjM3LjY2MDE1NjI1LCJ5IjozNDR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M201.96,435.941L186.577,448.057C171.195,460.174,140.429,484.407,125.047,507.191C109.664,529.974,109.664,551.307,109.664,572.641C109.664,593.974,109.664,615.307,109.664,636.641C109.664,657.974,109.664,679.307,109.664,698.641C109.664,717.974,109.664,735.307,109.664,747.474C109.664,759.641,109.664,766.641,109.664,770.141L109.664,773.641" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MjAxLjk2MDEzNDQwMzc1NDksInkiOjQzNS45NDA2MDMxNTM3NTQ4Nn0seyJ4IjoxMDkuNjY0MDYyNSwieSI6NTA4LjY0MDYyNX0seyJ4IjoxMDkuNjY0MDYyNSwieSI6NTcyLjY0MDYyNX0seyJ4IjoxMDkuNjY0MDYyNSwieSI6NjM2LjY0MDYyNX0seyJ4IjoxMDkuNjY0MDYyNSwieSI6NzAwLjY0MDYyNX0seyJ4IjoxMDkuNjY0MDYyNSwieSI6NzUyLjY0MDYyNX0seyJ4IjoxMDkuNjY0MDYyNSwieSI6Nzc3LjY0MDYyNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.664,831.641L109.664,835.807C109.664,839.974,109.664,848.307,109.664,855.974C109.664,863.641,109.664,870.641,109.664,874.141L109.664,877.641" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MTA5LjY2NDA2MjUsInkiOjgzMS42NDA2MjV9LHsieCI6MTA5LjY2NDA2MjUsInkiOjg1Ni42NDA2MjV9LHsieCI6MTA5LjY2NDA2MjUsInkiOjg4MS42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.664,935.641L109.664,939.807C109.664,943.974,109.664,952.307,109.664,959.974C109.664,967.641,109.664,974.641,109.664,978.141L109.664,981.641" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MTA5LjY2NDA2MjUsInkiOjkzNS42NDA2MjV9LHsieCI6MTA5LjY2NDA2MjUsInkiOjk2MC42NDA2MjV9LHsieCI6MTA5LjY2NDA2MjUsInkiOjk4NS42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M109.664,1039.641L109.664,1043.807C109.664,1047.974,109.664,1056.307,119.303,1064.39C128.941,1072.472,148.218,1080.304,157.856,1084.219L167.495,1088.135" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6MTA5LjY2NDA2MjUsInkiOjEwMzkuNjQwNjI1fSx7IngiOjEwOS42NjQwNjI1LCJ5IjoxMDY0LjY0MDYyNX0seyJ4IjoxNzEuMjAwNjQ2MDMzNjUzODQsInkiOjEwODkuNjQwNjI1fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M273.36,435.941L288.743,448.057C304.126,460.174,334.891,484.407,350.274,502.024C365.656,519.641,365.656,530.641,365.656,536.141L365.656,541.641" id="L_D_I_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_I_0" data-points="W3sieCI6MjczLjM2MDE3ODA5NjI0NTE0LCJ5Ijo0MzUuOTQwNjAzMTUzNzU0ODZ9LHsieCI6MzY1LjY1NjI1LCJ5Ijo1MDguNjQwNjI1fSx7IngiOjM2NS42NTYyNSwieSI6NTQ1LjY0MDYyNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M365.656,599.641L365.656,605.807C365.656,611.974,365.656,624.307,365.656,635.974C365.656,647.641,365.656,658.641,365.656,664.141L365.656,669.641" id="L_I_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_I_J_0" data-points="W3sieCI6MzY1LjY1NjI1LCJ5Ijo1OTkuNjQwNjI1fSx7IngiOjM2NS42NTYyNSwieSI6NjM2LjY0MDYyNX0seyJ4IjozNjUuNjU2MjUsInkiOjY3My42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M365.656,727.641L365.656,731.807C365.656,735.974,365.656,744.307,365.656,751.974C365.656,759.641,365.656,766.641,365.656,770.141L365.656,773.641" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6MzY1LjY1NjI1LCJ5Ijo3MjcuNjQwNjI1fSx7IngiOjM2NS42NTYyNSwieSI6NzUyLjY0MDYyNX0seyJ4IjozNjUuNjU2MjUsInkiOjc3Ny42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M365.656,831.641L365.656,835.807C365.656,839.974,365.656,848.307,365.656,855.974C365.656,863.641,365.656,870.641,365.656,874.141L365.656,877.641" id="L_K_L_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_K_L_0" data-points="W3sieCI6MzY1LjY1NjI1LCJ5Ijo4MzEuNjQwNjI1fSx7IngiOjM2NS42NTYyNSwieSI6ODU2LjY0MDYyNX0seyJ4IjozNjUuNjU2MjUsInkiOjg4MS42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M365.656,935.641L365.656,939.807C365.656,943.974,365.656,952.307,365.656,959.974C365.656,967.641,365.656,974.641,365.656,978.141L365.656,981.641" id="L_L_M_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_L_M_0" data-points="W3sieCI6MzY1LjY1NjI1LCJ5Ijo5MzUuNjQwNjI1fSx7IngiOjM2NS42NTYyNSwieSI6OTYwLjY0MDYyNX0seyJ4IjozNjUuNjU2MjUsInkiOjk4NS42NDA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M365.656,1039.641L365.656,1043.807C365.656,1047.974,365.656,1056.307,356.018,1064.39C346.379,1072.472,327.102,1080.304,317.464,1084.219L307.826,1088.135" id="L_M_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_M_H_0" data-points="W3sieCI6MzY1LjY1NjI1LCJ5IjoxMDM5LjY0MDYyNX0seyJ4IjozNjUuNjU2MjUsInkiOjEwNjQuNjQwNjI1fSx7IngiOjMwNC4xMTk2NjY0NjYzNDYyLCJ5IjoxMDg5LjY0MDYyNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(109.6640625, 636.640625)"><g class="label" data-id="L_D_E_0" transform="translate(-34.5625, -12)"><foreignObject width="69.125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Verifiable</p></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(365.65625, 508.640625)"><g class="label" data-id="L_D_I_0" transform="translate(-43.8515625, -12)"><foreignObject width="87.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Unverifiable</p></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_I_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_K_L_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_L_M_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_M_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(237.66015625, 35)"><rect class="basic label-container" style="" x="-69.890625" y="-27" width="139.78125" height="54"/><g class="label" style="" transform="translate(-39.890625, -12)"><rect/><foreignObject width="79.78125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Base Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(237.66015625, 151)"><rect class="basic label-container" style="" x="-111.9921875" y="-39" width="223.984375" height="78"/><g class="label" style="" transform="translate(-81.9921875, -24)"><rect/><foreignObject width="163.984375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Supervised Fine-Tuning<br />SFT</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(237.66015625, 267)"><rect class="basic label-container" style="" x="-66.109375" y="-27" width="132.21875" height="54"/><g class="label" style="" transform="translate(-36.109375, -12)"><rect/><foreignObject width="72.21875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>SFT Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(237.66015625, 407.8203125)"><polygon points="63.8203125,0 127.640625,-63.8203125 63.8203125,-127.640625 0,-63.8203125" class="label-container" transform="translate(-63.3203125, 63.8203125)"/><g class="label" style="" transform="translate(-36.8203125, -12)"><rect/><foreignObject width="73.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Task Type?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(109.6640625, 804.640625)"><rect class="basic label-container" style="" x="-101.6640625" y="-27" width="203.328125" height="54"/><g class="label" style="" transform="translate(-71.6640625, -12)"><rect/><foreignObject width="143.328125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generate Responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(109.6640625, 908.640625)"><rect class="basic label-container" style="" x="-100.7890625" y="-27" width="201.578125" height="54"/><g class="label" style="" transform="translate(-70.7890625, -12)"><rect/><foreignObject width="141.578125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Automated Checker</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(109.6640625, 1012.640625)"><rect class="basic label-container" style="" x="-75.34375" y="-27" width="150.6875" height="54"/><g class="label" style="" transform="translate(-45.34375, -12)"><rect/><foreignObject width="90.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>RL Algorithm</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(237.66015625, 1116.640625)"><rect class="basic label-container" style="" x="-70.9765625" y="-27" width="141.953125" height="54"/><g class="label" style="" transform="translate(-40.9765625, -12)"><rect/><foreignObject width="81.953125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Final Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-15" transform="translate(365.65625, 572.640625)"><rect class="basic label-container" style="" x="-101.6640625" y="-27" width="203.328125" height="54"/><g class="label" style="" transform="translate(-71.6640625, -12)"><rect/><foreignObject width="143.328125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generate Responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-17" transform="translate(365.65625, 700.640625)"><rect class="basic label-container" style="" x="-85.1640625" y="-27" width="170.328125" height="54"/><g class="label" style="" transform="translate(-55.1640625, -12)"><rect/><foreignObject width="110.328125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Human Ranking</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-19" transform="translate(365.65625, 804.640625)"><rect class="basic label-container" style="" x="-100.1171875" y="-27" width="200.234375" height="54"/><g class="label" style="" transform="translate(-70.1171875, -12)"><rect/><foreignObject width="140.234375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Train Reward Model</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-L-21" transform="translate(365.65625, 908.640625)"><rect class="basic label-container" style="" x="-105.203125" y="-27" width="210.40625" height="54"/><g class="label" style="" transform="translate(-75.203125, -12)"><rect/><foreignObject width="150.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Reward Model Scores</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-M-23" transform="translate(365.65625, 1012.640625)"><rect class="basic label-container" style="" x="-75.34375" y="-27" width="150.6875" height="54"/><g class="label" style="" transform="translate(-45.34375, -12)"><rect/><foreignObject width="90.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>RL Algorithm</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Stage Summary**:

1. **SFT**: Base model → SFT model
   * Learns to answer questions (format adaptation)
   * Uses demonstration data (prompt-response pairs)
   * Identical algorithm to pre-training (just different data)

2. **RL** (if verifiable tasks):
   * SFT model → Final model
   * Uses automated verification
   * Reinforces correct responses

3. **RLHF** (if unverifiable tasks):
   * Train reward model from human feedback
   * Use reward model to score responses
   * Reinforce high-scoring responses

**Final Result**: Model suitable for production chatbot deployment

### Practical Demonstration: Base vs. Post-Trained

**Base Model** (Llama 3.1 405B base):
```
Input: "I want to learn ML. What should I do?"
Output: "What are the resources I should follow? How long will it take?..."
→ Continues with more questions (completion behavior)
```

**Post-Trained Model** (Llama 3.1 405B post-trained):
```
Input: "I want to learn ML. What should I do?"
Output: "Learning machine learning is an exciting journey. Here is a
step-by-step guide to help you get started:

Step 1: Build foundational knowledge
- Mathematics: Linear algebra, calculus, probability, statistics
- Programming: Python (essential for ML)

Step 2: Take structured courses
- Andrew Ng's Machine Learning course on Coursera
- Fast.ai's Practical Deep Learning for Coders
...

[Formatted with bold text, bullet points, clear structure]"
```

**Differences**:
* **Format**: Post-trained provides structured, formatted answer
* **Helpfulness**: Detailed, actionable steps vs. more questions
* **Quality**: Professional, complete response vs. incomplete continuation
* **This is the impact of post-training**

## Model Evaluation

### Training Stage Summary

**Overview of Four Stages**:

| Stage | Data | Size | Quality | Compute | Training Time | ML Objective | Output |
|-------|------|------|---------|---------|---------------|--------------|--------|
| **Pre-Training** | Internet data | Trillions of tokens | Low quality, high quantity | Thousands of GPUs | Months | Next token prediction | Base model |
| **SFT** | Prompt-response pairs | 10K-100K examples | High quality, low quantity | Hundreds of GPUs | Days | Next token prediction | SFT model |
| **Reward Modeling** | Comparison data (ranked responses) | 10K-100K comparisons | High quality | Hundreds of GPUs | Days | Predict scores | Reward model |
| **Reinforcement Learning** | Prompts only | 10K-100K prompts | N/A | Hundreds of GPUs | Days | Maximize reward score | Final model |

**Key Points**:
* Pre-training is by far most expensive (cost, time, compute)
* Post-training stages relatively cheap in comparison
* Same next-token prediction objective for pre-training and SFT
* RL uses different objective (maximize reward)

### Why Evaluation Matters

**Reality**: Many LLMs exist from different companies
* DeepSeek, Qwen, Llama, GPT, Claude, Gemini, Grok, and many more
* Models of different sizes: Small, medium, large
* Continuously updated with new versions

**Questions**:
* Which model is better?
* How to compare models objectively?
* How to track improvement over time?

**Solution**: Systematic evaluation methods

### Evaluation Categories

**Two Main Approaches**:

1. **Offline Evaluation**: Test models in controlled environment
   * Use evaluation datasets
   * Measure performance metrics
   * Compare before deployment

2. **Online Evaluation**: Test models in production
   * Real user interactions
   * Monitor live performance
   * Collect feedback from actual usage

### Offline Evaluation Methods

#### 1. Traditional Metrics: Perplexity

**Definition**: Measures how accurately model predicts exact token sequences

**How it Works**:
* Take evaluation data: "how are you doing"
* Calculate probability of model generating this exact sequence
* Pass "how" → get P(are | how)
* Pass "how are" → get P(you | how are)
* Pass "how are you" → get P(doing | how are you)
* Combine probabilities (multiply or sum of logs)

**Formula**: Mathematical formula available on Wikipedia

**Interpretation**:
* **Low perplexity**: Model can reproduce evaluation data well
* **High perplexity**: Model struggles to reproduce evaluation data

**Limitations**: **No longer meaningful for modern LLMs**
* Reproducing text ≠ useful for humans
* Doesn't measure helpfulness, correctness, or safety
* Just measures memorization/statistical fit
* **Not used as primary evaluation metric today**

#### 2. Task-Specific Benchmarks (Modern Standard)

**Purpose**: Assess performance on diverse real-world tasks

**Major Domains**:
* Mathematics
* Code generation
* Common sense reasoning
* World knowledge
* Language understanding
* Problem solving

**How it Works**:
* Each domain has benchmark datasets
* Datasets contain problems with known correct answers
* Pass problems to LLM
* Compare LLM output with correct answer
* Calculate accuracy/score

**Example Benchmarks**:

**Common Sense Reasoning**:
```
Prompt: "The trophy doesn't fit in the brown suitcase because
it's too large. What is too large?
(A) The trophy
(B) The suitcase"

Correct Answer: (A) The trophy
```

* Pass to LLM, see if it selects (A)
* Repeat across many examples
* Calculate percentage correct

**Benchmark examples**: HellaSwag, PIQA, ARC

**World Knowledge**:
```
Prompt: "Who wrote Romeo and Juliet?"
Correct Answer: "William Shakespeare"
```

* Test factual knowledge
* Measure accuracy on fact-based questions

**Benchmark examples**: TriviaQA, NaturalQuestions

**Mathematical Reasoning**:
```
Prompt: "If a train travels 60 miles per hour for 3 hours,
how far does it travel?"
Correct Answer: "180 miles"
```

* Test math capabilities
* Can check numerical answer

**Benchmark examples**: GSM8K, MATH

**Code Generation**:
```
Prompt: "Write a Python function to check if a number is prime."
Correct Answer: [valid Python function]
```

* Can execute code
* Test if output is correct
* Check for errors

**Benchmark examples**: HumanEval, MBPP

**Why This Matters**:
* These benchmarks test capabilities humans care about
* Provide quantitative comparison between models
* Track progress over time
* Industry standard for model evaluation

**Common Practice**: Model papers report scores on multiple benchmarks
* Allows direct comparison with other models
* Shows strengths and weaknesses across domains

#### 3. Human Evaluation

**Approach**: Expert humans assess model outputs

**Process**:
* Hire domain experts
* Give them challenging questions
* Ask models to answer
* Experts evaluate:
  * Correctness
  * Helpfulness
  * Clarity
  * Safety

**Advantages**:
* Can assess nuanced quality
* Captures human preferences directly
* Evaluates aspects hard to automate

**Limitations**:
* **Subjective**: Different evaluators may disagree
* **Biased**: Personal preferences influence judgments
* **Expensive**: Requires paid expert time
* **Slow**: Cannot scale to large test sets
* **Domain-dependent**: Quality depends on evaluator expertise

**When Used**:
* Assessing creativity, style, tone
* Evaluating complex reasoning
* Safety and ethical considerations
* Complementing automated benchmarks

### Online Evaluation Methods

#### 1. Human Feedback in Production

**Implementation**: Built into user interface

**Example (ChatGPT)**:
* User asks question
* Model generates response
* UI shows thumbs up / thumbs down buttons
* User rates response

**Data Collected**:
* User satisfaction signals
* Which responses are helpful
* Which responses are problematic

**Uses**:
1. **Evaluation**: Track model performance over time
   * Compare thumbs up/down ratios between model versions
   * Identify problems or regressions

2. **Training Data**: Use for further improvement
   * Negative feedback highlights areas to improve
   * Positive feedback shows successful patterns
   * Can feed into future RLHF training

**Advantages**:
* Real user feedback
* Large scale data collection
* Continuous monitoring

**Limitations**:
* Users may not always rate accurately
* Selection bias (who chooses to rate)
* May not represent all use cases

#### 2. Crowdsourcing Platforms

**Most Notable: LMSYS Chatbot Arena**

**What it is**:
* Public web-based platform
* Developed by Berkeley students/researchers
* Ranks LLMs through crowdsourced comparisons

**How it Works**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 243.375px; background-color: transparent;" viewBox="0 0 243.375 710" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M121.688,62L121.688,66.167C121.688,70.333,121.688,78.667,121.688,86.333C121.688,94,121.688,101,121.688,104.5L121.688,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTIxLjY4NzUsInkiOjYyfSx7IngiOjEyMS42ODc1LCJ5Ijo4N30seyJ4IjoxMjEuNjg3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M121.688,190L121.688,194.167C121.688,198.333,121.688,206.667,121.688,214.333C121.688,222,121.688,229,121.688,232.5L121.688,236" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTIxLjY4NzUsInkiOjE5MH0seyJ4IjoxMjEuNjg3NSwieSI6MjE1fSx7IngiOjEyMS42ODc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M121.688,318L121.688,322.167C121.688,326.333,121.688,334.667,121.688,342.333C121.688,350,121.688,357,121.688,360.5L121.688,364" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6MTIxLjY4NzUsInkiOjMxOH0seyJ4IjoxMjEuNjg3NSwieSI6MzQzfSx7IngiOjEyMS42ODc1LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M121.688,446L121.688,450.167C121.688,454.333,121.688,462.667,121.688,470.333C121.688,478,121.688,485,121.688,488.5L121.688,492" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTIxLjY4NzUsInkiOjQ0Nn0seyJ4IjoxMjEuNjg3NSwieSI6NDcxfSx7IngiOjEyMS42ODc1LCJ5Ijo0OTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M121.688,574L121.688,578.167C121.688,582.333,121.688,590.667,121.688,598.333C121.688,606,121.688,613,121.688,616.5L121.688,620" id="L_E_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_E_F_0" data-points="W3sieCI6MTIxLjY4NzUsInkiOjU3NH0seyJ4IjoxMjEuNjg3NSwieSI6NTk5fSx7IngiOjEyMS42ODc1LCJ5Ijo2MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_E_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(121.6875, 35)"><rect class="basic label-container" style="" x="-99.46875" y="-27" width="198.9375" height="54"/><g class="label" style="" transform="translate(-69.46875, -12)"><rect/><foreignObject width="138.9375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User enters prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(121.6875, 151)"><rect class="basic label-container" style="" x="-113.6875" y="-39" width="227.375" height="78"/><g class="label" style="" transform="translate(-83.6875, -24)"><rect/><foreignObject width="167.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Two anonymous models<br />generate responses</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(121.6875, 279)"><rect class="basic label-container" style="" x="-105.265625" y="-39" width="210.53125" height="78"/><g class="label" style="" transform="translate(-75.265625, -24)"><rect/><foreignObject width="150.53125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User sees Response A<br />and Response B</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(121.6875, 407)"><rect class="basic label-container" style="" x="-87.71875" y="-39" width="175.4375" height="78"/><g class="label" style="" transform="translate(-57.71875, -24)"><rect/><foreignObject width="115.4375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User votes:<br />Which is better?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(121.6875, 535)"><rect class="basic label-container" style="" x="-87.9375" y="-39" width="175.875" height="78"/><g class="label" style="" transform="translate(-57.9375, -24)"><rect/><foreignObject width="115.875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Models revealed<br />after voting</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(121.6875, 663)"><rect class="basic label-container" style="" x="-96.9375" y="-39" width="193.875" height="78"/><g class="label" style="" transform="translate(-66.9375, -24)"><rect/><foreignObject width="133.875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Votes aggregated<br />to update rankings</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Key Features**:
* **Anonymous comparison**: User doesn't know which model is which
* **Prevents bias**: Can't favor based on model name/company
* **Pairwise voting**: Choose between two responses
* **Large scale**: Thousands of user votes
* **Public leaderboard**: Rankings visible to everyone

**LMSYS Leaderboard** (example snapshot):

| Rank | Model | Developer | Score | License | Notes |
|------|-------|-----------|-------|---------|-------|
| 1 | Gemini 2.5 Pro | Google | 1343 | Proprietary | Closed |
| 2 | o3 | OpenAI | 1337 | Proprietary | Closed |
| 3 | GPT-4.5 | OpenAI | 1328 | Proprietary | Closed |
| 4 | Claude Opus 4 | Anthropic | 1315 | Proprietary | Closed |
| ... | ... | ... | ... | ... | ... |
| 30 | DeepSeek | DeepSeek | 1245 | MIT | Open source, open weights |
| ... | ... | ... | ... | ... | ... |
| 209 | Llama 13B | Meta | 892 | Open | Older model |

**Observations**:
* **211 models** currently ranked
* Rankings continuously updated
* Newer models typically rank higher
* Mix of closed and open source models
* Clear metric for comparing models

**Advantages**:
* Large-scale real user preferences
* Unbiased (anonymous voting)
* Covers diverse use cases
* Public and transparent

**Limitations**:
* May not represent enterprise use cases
* Users have varying expertise levels
* Some tasks better suited for voting than others

**Industry Impact**:
* Companies track their ranking
* Influences model development priorities
* Used in marketing and product positioning
* Community reference point

### Evaluation Summary

**Multi-Faceted Approach**: No single perfect evaluation method

**Best Practice**: Combine multiple evaluation types
* **Automated benchmarks**: Quantitative, scalable, consistent
* **Human evaluation**: Qualitative, nuanced, contextual
* **Production feedback**: Real-world usage, actual user needs
* **Crowdsourcing**: Large-scale preferences, unbiased comparison

**Evolution of Models**: Rankings change frequently
* Companies release new versions regularly
* Competition drives rapid improvement
* Today's leader may not be tomorrow's

## System Design and Architecture

### Chatbot System Components

**Reality**: Trained LLM is small part of complete chatbot service

**Full System Architecture**:


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 543.723px; background-color: transparent;" viewBox="0 0 543.72265625 1222" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M167.559,62L167.559,66.167C167.559,70.333,167.559,78.667,167.559,86.333C167.559,94,167.559,101,167.559,104.5L167.559,108" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTY3LjU1ODU5Mzc1LCJ5Ijo2Mn0seyJ4IjoxNjcuNTU4NTkzNzUsInkiOjg3fSx7IngiOjE2Ny41NTg1OTM3NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M203.083,190L208.7,196.167C214.317,202.333,225.551,214.667,231.168,226.333C236.785,238,236.785,249,236.785,254.5L236.785,260" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MjAzLjA4Mjc1MDgyMjM2ODQsInkiOjE5MH0seyJ4IjoyMzYuNzg1MTU2MjUsInkiOjIyN30seyJ4IjoyMzYuNzg1MTU2MjUsInkiOjI2NH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M132.034,190L126.417,196.167C120.8,202.333,109.566,214.667,103.949,231.5C98.332,248.333,98.332,269.667,98.332,289C98.332,308.333,98.332,325.667,98.332,343C98.332,360.333,98.332,377.667,98.332,395C98.332,412.333,98.332,429.667,98.332,449C98.332,468.333,98.332,489.667,98.332,513C98.332,536.333,98.332,561.667,98.332,587C98.332,612.333,98.332,637.667,98.332,661C98.332,684.333,98.332,705.667,98.332,725C98.332,744.333,98.332,761.667,98.332,779C98.332,796.333,98.332,813.667,98.332,833C98.332,852.333,98.332,873.667,98.332,897C98.332,920.333,98.332,945.667,99.943,963.86C101.555,982.053,104.778,993.107,106.389,998.633L108.001,1004.16" id="L_B_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_D_0" data-points="W3sieCI6MTMyLjAzNDQzNjY3NzYzMTYsInkiOjE5MH0seyJ4Ijo5OC4zMzIwMzEyNSwieSI6MjI3fSx7IngiOjk4LjMzMjAzMTI1LCJ5IjoyOTF9LHsieCI6OTguMzMyMDMxMjUsInkiOjM0M30seyJ4Ijo5OC4zMzIwMzEyNSwieSI6Mzk1fSx7IngiOjk4LjMzMjAzMTI1LCJ5Ijo0NDd9LHsieCI6OTguMzMyMDMxMjUsInkiOjUxMX0seyJ4Ijo5OC4zMzIwMzEyNSwieSI6NTg3fSx7IngiOjk4LjMzMjAzMTI1LCJ5Ijo2NjN9LHsieCI6OTguMzMyMDMxMjUsInkiOjcyN30seyJ4Ijo5OC4zMzIwMzEyNSwieSI6Nzc5fSx7IngiOjk4LjMzMjAzMTI1LCJ5Ijo4MzF9LHsieCI6OTguMzMyMDMxMjUsInkiOjg5NX0seyJ4Ijo5OC4zMzIwMzEyNSwieSI6OTcxfSx7IngiOjEwOS4xMjA1MjgzNzE3MTA1MiwieSI6MTAwOH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M120.492,1086L120.492,1090.167C120.492,1094.333,120.492,1102.667,120.492,1110.333C120.492,1118,120.492,1125,120.492,1128.5L120.492,1132" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6MTIwLjQ5MjE4NzUsInkiOjEwODZ9LHsieCI6MTIwLjQ5MjE4NzUsInkiOjExMTF9LHsieCI6MTIwLjQ5MjE4NzUsInkiOjExMzZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M236.785,318L236.785,322.167C236.785,326.333,236.785,334.667,236.785,342.333C236.785,350,236.785,357,236.785,360.5L236.785,364" id="L_C_F_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_F_0" data-points="W3sieCI6MjM2Ljc4NTE1NjI1LCJ5IjozMTh9LHsieCI6MjM2Ljc4NTE1NjI1LCJ5IjozNDN9LHsieCI6MjM2Ljc4NTE1NjI1LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M236.785,422L236.785,426.167C236.785,430.333,236.785,438.667,236.785,446.333C236.785,454,236.785,461,236.785,464.5L236.785,468" id="L_F_G_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_F_G_0" data-points="W3sieCI6MjM2Ljc4NTE1NjI1LCJ5Ijo0MjJ9LHsieCI6MjM2Ljc4NTE1NjI1LCJ5Ijo0NDd9LHsieCI6MjM2Ljc4NTE1NjI1LCJ5Ijo0NzJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M236.785,550L236.785,556.167C236.785,562.333,236.785,574.667,245.409,586.628C254.032,598.59,271.279,610.179,279.902,615.974L288.526,621.769" id="L_G_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_G_H_0" data-points="W3sieCI6MjM2Ljc4NTE1NjI1LCJ5Ijo1NTB9LHsieCI6MjM2Ljc4NTE1NjI1LCJ5Ijo1ODd9LHsieCI6MjkxLjg0NTg1NzMxOTA3ODk2LCJ5Ijo2MjR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M462.98,542L462.98,549.5C462.98,557,462.98,572,454.357,585.295C445.734,598.59,428.487,610.179,419.863,615.974L411.24,621.769" id="L_I_H_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_I_H_0" data-points="W3sieCI6NDYyLjk4MDQ2ODc1LCJ5Ijo1Mzh9LHsieCI6NDYyLjk4MDQ2ODc1LCJ5Ijo1ODd9LHsieCI6NDA3LjkxOTc2NzY4MDkyMTA0LCJ5Ijo2MjR9XQ==" marker-start="url(#my-svg_flowchart-v2-pointStart)" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M349.883,702L349.883,706.167C349.883,710.333,349.883,718.667,349.883,726.333C349.883,734,349.883,741,349.883,744.5L349.883,748" id="L_H_J_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_H_J_0" data-points="W3sieCI6MzQ5Ljg4MjgxMjUsInkiOjcwMn0seyJ4IjozNDkuODgyODEyNSwieSI6NzI3fSx7IngiOjM0OS44ODI4MTI1LCJ5Ijo3NTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M349.883,806L349.883,810.167C349.883,814.333,349.883,822.667,349.883,830.333C349.883,838,349.883,845,349.883,848.5L349.883,852" id="L_J_K_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_J_K_0" data-points="W3sieCI6MzQ5Ljg4MjgxMjUsInkiOjgwNn0seyJ4IjozNDkuODgyODEyNSwieSI6ODMxfSx7IngiOjM0OS44ODI4MTI1LCJ5Ijo4NTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M381.29,934L386.256,940.167C391.222,946.333,401.154,958.667,406.12,972.333C411.086,986,411.086,1001,411.086,1008.5L411.086,1016" id="L_K_L_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_K_L_0" data-points="W3sieCI6MzgxLjI4OTY3OTI3NjMxNTgsInkiOjkzNH0seyJ4Ijo0MTEuMDg1OTM3NSwieSI6OTcxfSx7IngiOjQxMS4wODU5Mzc1LCJ5IjoxMDIwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M267.693,934L254.698,940.167C241.702,946.333,215.71,958.667,197.546,970.507C179.382,982.348,169.046,993.695,163.878,999.369L158.71,1005.043" id="L_K_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_K_D_0" data-points="W3sieCI6MjY3LjY5MzM1OTM3NSwieSI6OTM0fSx7IngiOjE4OS43MTg3NSwieSI6OTcxfSx7IngiOjE1Ni4wMTYzNDQ1NzIzNjg0LCJ5IjoxMDA4fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(236.78515625, 227)"><g class="label" data-id="L_B_C_0" transform="translate(-15.375, -12)"><foreignObject width="30.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Safe</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(98.33203125, 587)"><g class="label" data-id="L_B_D_0" transform="translate(-24.3203125, -12)"><foreignObject width="48.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Unsafe</p></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_F_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_F_G_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_G_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_I_H_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_H_J_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_J_K_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(411.0859375, 971)"><g class="label" data-id="L_K_L_0" transform="translate(-15.375, -12)"><foreignObject width="30.75" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Safe</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(189.71875, 971)"><g class="label" data-id="L_K_D_0" transform="translate(-24.3203125, -12)"><foreignObject width="48.640625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Unsafe</p></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(167.55859375, 35)"><rect class="basic label-container" style="" x="-91.5234375" y="-27" width="183.046875" height="54"/><g class="label" style="" transform="translate(-61.5234375, -12)"><rect/><foreignObject width="123.046875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Text Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(167.55859375, 151)"><rect class="basic label-container" style="" x="-87.75" y="-39" width="175.5" height="78"/><g class="label" style="" transform="translate(-57.75, -24)"><rect/><foreignObject width="115.5" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Input Guardrails<br />Safety Filtering</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(236.78515625, 291)"><rect class="basic label-container" style="" x="-91.203125" y="-27" width="182.40625" height="54"/><g class="label" style="" transform="translate(-61.203125, -12)"><rect/><foreignObject width="122.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt Enhancer</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(120.4921875, 1047)"><rect class="basic label-container" style="" x="-99.140625" y="-39" width="198.28125" height="78"/><g class="label" style="" transform="translate(-69.140625, -24)"><rect/><foreignObject width="138.28125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Rejection Response<br />Generator</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(120.4921875, 1175)"><rect class="basic label-container" style="" x="-112.4921875" y="-39" width="224.984375" height="78"/><g class="label" style="" transform="translate(-82.4921875, -24)"><rect/><foreignObject width="164.984375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Show to User:<br />Cannot assist with that</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-F-9" transform="translate(236.78515625, 395)"><rect class="basic label-container" style="" x="-92.5546875" y="-27" width="185.109375" height="54"/><g class="label" style="" transform="translate(-62.5546875, -12)"><rect/><foreignObject width="125.109375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Enhanced Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-G-11" transform="translate(236.78515625, 511)"><rect class="basic label-container" style="" x="-103.453125" y="-39" width="206.90625" height="78"/><g class="label" style="" transform="translate(-73.453125, -24)"><rect/><foreignObject width="146.90625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Session Management<br />Append Chat History</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-H-13" transform="translate(349.8828125, 663)"><rect class="basic label-container" style="" x="-101.4609375" y="-39" width="202.921875" height="78"/><g class="label" style="" transform="translate(-71.4609375, -24)"><rect/><foreignObject width="142.921875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response Generator<br />Top-P Sampling</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-I-14" transform="translate(462.98046875, 511)"><rect class="basic label-container" style="" x="-72.7421875" y="-27" width="145.484375" height="54"/><g class="label" style="" transform="translate(-42.7421875, -12)"><rect/><foreignObject width="85.484375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Trained LLM</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-J-17" transform="translate(349.8828125, 779)"><rect class="basic label-container" style="" x="-102.8828125" y="-27" width="205.765625" height="54"/><g class="label" style="" transform="translate(-72.8828125, -12)"><rect/><foreignObject width="145.765625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Generated Response</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-K-19" transform="translate(349.8828125, 895)"><rect class="basic label-container" style="" x="-94.0859375" y="-39" width="188.171875" height="78"/><g class="label" style="" transform="translate(-64.0859375, -24)"><rect/><foreignObject width="128.171875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Output Guardrails<br />Response Safety</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-L-21" transform="translate(411.0859375, 1047)"><rect class="basic label-container" style="" x="-111.71875" y="-27" width="223.4375" height="54"/><g class="label" style="" transform="translate(-81.71875, -12)"><rect/><foreignObject width="163.4375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Show Response to User</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Important**: This is simplified - production systems may have more components

### Component Details

#### 1. Input Guardrails (Safety Filtering)

**Purpose**: Ensure user prompt is safe to process

**What it Checks**:
* Violence or harmful content
* Illegal requests
* Inappropriate content
* Attempts to jailbreak or misuse model

**How it Works**:
* Usually machine learning classifier
* Trained to detect unsafe prompts
* Fast screening before expensive LLM processing

**Decision**:
* **If unsafe**: Route to rejection response
  * Example: "I cannot assist with that request"
  * No LLM processing (saves compute)

* **If safe**: Continue to next stage

**Why Important**:
* Prevents misuse
* Protects brand reputation
* Ensures compliance with policies
* Saves compute on problematic requests

#### 2. Prompt Enhancer

**Purpose**: Improve input quality before LLM processing

**What it Fixes**:
* **Typos and misspellings**: "machien lerning" → "machine learning"
* **Grammar errors**: Corrects grammatical mistakes
* **Punctuation**: Adds missing punctuation
* **Ambiguity**: Clarifies vague prompts
* **Formatting**: Standardizes input format

**Implementation**:
* Combination of heuristics (rule-based)
* Machine learning models (trained for text correction)
* May use smaller specialized models (faster than main LLM)

**Why Important**:
* LLM performs better with clean input
* Reduces ambiguity in responses
* Improves overall response quality
* Helps model understand user intent

**Example**:
```
Original: "hw do i lern python its very hard"
Enhanced: "How do I learn Python? It's very hard."
→ Clear, well-formed question for LLM
```

#### 3. Session Management (Chat History)

**Purpose**: Enable multi-turn conversations with context

**Challenge**: LLM only sees current input
* Without memory, cannot handle follow-ups
* Example: "Make it shorter" - what is "it"?

**Solution**: Append entire conversation history to each new prompt

**How it Works**:

```
Turn 1:
User: "Help me choose a startup name"
Assistant: "Here are some suggestions: TechFlow, InnovateLab, ..."

Turn 2:
Input to LLM:
"[Previous conversation]
User: Help me choose a startup name
Assistant: Here are some suggestions: TechFlow, InnovateLab, ...

[New message]
User: Make them more formal"
```

**Key Points**:
* Full history sent with each new message
* LLM sees complete context
* Enables natural follow-up questions
* History scoped to session/conversation

**Implementation Details**:
* Store chat history in session state
* Prepend history to new prompts
* May truncate very long histories (token limits)
* Clear when user starts new conversation

**Example Use Case**:
```
User: "Explain quantum computing"
Assistant: [Detailed explanation]
User: "Give me an example"
→ Model understands "example" refers to quantum computing
User: "What are practical applications?"
→ Model maintains quantum computing context
```

#### 4. Response Generator

**Purpose**: Generate tokens iteratively using trained LLM

**Process**:
* Takes enhanced prompt + chat history
* Calls trained LLM repeatedly
* Uses text generation algorithm (Top-P sampling)
* Continues until completion

**Implementation Details**:
* Top-P sampling (discussed earlier)
* Hyperparameters: P value, temperature, max length
* Iterative token-by-token generation
* Stops at end-of-sentence token or max length

**Optimization Considerations**:
* Inference speed critical for user experience
* May use model quantization (reduce precision)
* Batching multiple requests
* GPU/specialized hardware acceleration

#### 5. Output Guardrails (Response Safety)

**Purpose**: Ensure generated response is safe to show user

**What it Checks**:
* Harmful or dangerous content
* Biased or discriminatory language
* Leaked sensitive information
* Incorrect safety-critical information
* Content violating policies

**How it Works**:
* Machine learning classifier
* Analyzes generated response
* Trained on examples of unsafe outputs

**Decision**:
* **If safe**: Show response to user
* **If unsafe**: Route to rejection response
  * Example: "I apologize, I cannot provide that information"
  * May appear to user as if model cannot answer

**Why Needed**:
* LLM may occasionally generate unsafe content
* Even with post-training, edge cases exist
* Extra safety layer for production systems
* Protects users and company

**Trade-off**: May occasionally block legitimate responses (false positives)

### System Design Summary

**Key Insights**:
1. **LLM is not standalone**: Requires extensive supporting infrastructure
2. **Safety is multi-layered**: Input and output guardrails
3. **Quality enhancement**: Pre and post-processing improve results
4. **Context management**: Session handling enables conversations
5. **Real systems more complex**: Production systems have additional components for:
   * Logging and monitoring
   * Rate limiting
   * User authentication
   * Load balancing
   * Caching
   * A/B testing
   * Error handling

**Engineering Reality**:
* Trained LLM = core capability
* Complete chatbot = LLM + extensive supporting systems
* Quality and safety require multiple layers
* User experience depends on entire stack

## Key Insights

### Fundamental Principles

* **LLMs are next-token predictors**: At their core, LLMs predict the next token in a sequence based on statistical patterns learned from training data. This simple objective underlies all their capabilities.

* **Two-stage training is essential**: Pre-training provides world knowledge through exposure to Internet data, while post-training adapts the model for practical use by teaching instruction-following and improving quality/safety.

* **Scale matters but is expensive**: Larger models with more parameters generally perform better, but training them requires massive computational resources (thousands of GPUs, millions of dollars) accessible only to well-funded organizations.

* **Base models vs. fine-tuned models serve different purposes**: Base models excel at text completion and have implicit world knowledge, but only post-trained models can serve as useful chatbots that answer questions helpfully and safely.

### Important Distinctions and Comparisons

* **Verifiable vs. unverifiable tasks** (Critical Distinction):
  * **Verifiable tasks** (math, coding): Can automatically check correctness, enabling automated RL training without human annotation
  * **Unverifiable tasks** (creative writing, brainstorming): Require human feedback and reward models (RLHF) to assess quality
  * This distinction fundamentally changes the training approach and data requirements

* **Word-level vs. character-level vs. subword-level tokenization**:
  * **Word-level**: Huge vocabulary (270K+ tokens), expensive to train - not used in modern LLMs
  * **Character-level**: Small vocabulary but very long sequences, computationally expensive - not used in modern LLMs
  * **Subword-level** (BPE): Optimal balance of vocabulary size and sequence length - industry standard for all modern LLMs

* **Greedy vs. beam search vs. Top-P sampling**:
  * **Greedy search**: Deterministic, always picks highest probability, suffers from repetition - not used in production
  * **Beam search**: Keeps top K paths, explores more options than greedy but still deterministic - not used in production
  * **Top-P (Nucleus) sampling**: Adaptive stochastic sampling that adjusts to model confidence - production standard for all modern LLMs

* **Pre-training data vs. post-training data quality and quantity**:
  * **Pre-training**: Trillions of tokens, low quality (noisy Internet data), high quantity
  * **SFT data**: 10K-100K examples, high quality (expert curated), low quantity
  * Trade-off: Pre-training needs scale, post-training needs quality

* **Deterministic vs. stochastic text generation**:
  * **Deterministic**: Same input always produces same output, no randomness, prone to repetition
  * **Stochastic**: Same input can produce different outputs, introduces diversity, prevents repetition
  * Modern systems use stochastic methods exclusively

### Practical Considerations

* **Tokenization efficiency varies by language**: Non-English languages often require 2-3x more tokens than English for the same text, affecting cost and performance.

* **Context matters in generation**: Session management (appending chat history) enables follow-up questions and maintains conversational context, critical for chatbot usability.

* **Multiple evaluation methods necessary**: No single metric captures all aspects of model quality. Combine automated benchmarks, human evaluation, and production feedback for comprehensive assessment.

* **Data cleaning is crucial but complex**: Raw HTML contains irrelevant tags, duplicated content, and unsafe material. Multi-stage cleaning pipelines (URL filtering, deduplication, PII removal) are essential for quality training data.

* **Safety requires multiple layers**: Both input guardrails (checking user prompts) and output guardrails (checking model responses) are necessary because unsafe content can enter from either direction.

* **Hyperparameter tuning is task-specific**:
  * Code generation needs low Top-P (0.1) for correctness
  * Creative writing needs high Top-P (0.95) for novelty
  * No universal optimal values - experimentation required

* **Production systems are much more complex than trained models**: Complete chatbot requires prompt enhancement, safety filtering, session management, response post-processing, and extensive infrastructure - trained LLM is just one component.

### Common Pitfalls and Misconceptions

* **Base models cannot be directly deployed**: Despite having impressive knowledge, base models only continue text rather than answer questions, making them useless for chatbot applications without post-training.

* **More parameters generally better but not always necessary**: A well-trained smaller model can outperform a poorly-trained larger model. Context and training quality matter significantly.

* **Perplexity is not a useful evaluation metric for modern LLMs**: It measures statistical fit to data, not helpfulness, correctness, or safety - task-specific benchmarks are far more meaningful.

* **RLHF requires human feedback only for reward model training**: Once reward model is trained, the actual RL phase is automated - common misconception that humans rate every response during RL.

* **Character-level tokenization seems simple but is impractical**: Very long sequences make training computationally prohibitive despite small vocabulary.

* **Greedy search is not optimal despite picking highest probability**: Local optimization at each step doesn't guarantee globally optimal sequence, and repetition is a severe problem.

## Quick Recall / Implementation Checklist

* [ ] **Data Preparation**: Understand the three-stage pipeline (crawling, cleaning, tokenization) and that clean datasets like FineWeb can provide good starting points
* [ ] **Tokenization strategy**: Use subword-level tokenizers (BPE) for modern LLMs - word-level and character-level are obsolete
* [ ] **Vocabulary size matters**: Larger vocabularies (100K-200K tokens) are more efficient for newer models than older 50K vocabularies
* [ ] **Distinguish verifiable vs. unverifiable tasks**: This determines whether you need automated verification or RLHF approach for reinforcement learning
* [ ] **Base models require post-training**: Never deploy base models directly - they complete text but don't answer questions
* [ ] **SFT uses same algorithm as pre-training**: Only the data changes (prompt-response pairs vs. raw Internet text) - no code changes needed
* [ ] **Reward models are proxies for human judgment**: Train once with human feedback, then use for automated scoring during RL
* [ ] **Top-P sampling is production standard**: Use Top-P (nucleus sampling) for all text generation - greedy and beam search cause repetition
* [ ] **Hyperparameters are task-specific**: Low Top-P (0.1-0.3) for factual/code tasks, high Top-P (0.9-0.95) for creative tasks
* [ ] **Temperature controls randomness**: Low temperature (0.1-0.5) for deterministic outputs, high temperature (1.5-2.0) for creative diversity
* [ ] **Session management enables conversations**: Append full chat history to each new prompt so LLM maintains context for follow-ups
* [ ] **Multi-layered safety is essential**: Implement both input guardrails (filter unsafe prompts) and output guardrails (verify response safety)
* [ ] **Prompt enhancement improves quality**: Fix typos, grammar, and ambiguity before LLM processing for better responses
* [ ] **Use task-specific benchmarks for evaluation**: Perplexity is obsolete - use domain benchmarks (math, coding, reasoning, knowledge) that matter for your use case
* [ ] **Monitor production with human feedback**: Implement thumbs up/down ratings to track model performance and collect training data
* [ ] **Engineering complexity is in scale, not algorithms**: Training code is straightforward (loss calculation + optimizer), but distributed training across thousands of GPUs requires extensive engineering
* [ ] **Common Crawl provides free web data**: No need to crawl yourself for research/startup use cases - use publicly available cleaned datasets
* [ ] **Model size vs. capability trade-off**: More parameters = more capability but also more cost (training, inference, memory) - choose appropriate size for use case
* [ ] **Pre-training is prohibitively expensive for most**: Hundreds of millions of dollars and months of training - only large companies can afford; most applications should use existing base models
* [ ] **Evaluation should be multi-faceted**: Combine automated benchmarks, human evaluation, and crowdsourcing platforms for comprehensive assessment
* [ ] **TikToken library for practical tokenization**: Use OpenAI's TikToken for fast BPE tokenization with pre-trained vocabularies
* [ ] **SFT data quality over quantity**: 10K-100K high-quality curated examples better than millions of noisy examples
* [ ] **Llama models offer open alternatives**: Open source models like Llama 3 (405B parameters) compete with proprietary models for many use cases
* [ ] **Inference optimization critical for production**: Model quantization, batching, and specialized hardware necessary for acceptable response times
* [ ] **Rankings change frequently**: Companies continuously release improved models - LMSYS leaderboard tracks current state of the art
