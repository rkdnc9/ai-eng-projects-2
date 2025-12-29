# Agents

## Overview

This lecture notes covers the design and operation of AI agents, building upon LLMs to create autonomous systems capable of complex task completion. Rather than limiting AI to simple next-token prediction, agents augment LLMs with tools, planning capabilities, and multi-step reasoning. The focus spans from fundamental concepts through workflow patterns, tool integration, multi-step reasoning frameworks, and multi-agent coordination. Understanding agents requires grasping their spectrum of autonomy levels, from simple processors to fully autonomous multi-agent systems. This progression shows how to systematically add agency to LLMs, enabling them to handle tasks that single LLMs cannot.

## Core Concepts

### What are Agents?

At the most fundamental level, **agents are software systems that augment LLMs with additional capabilities** to make them more powerful. LLMs, by themselves, are static models trained on internet data that excel at predicting the next token but lack autonomy, planning ability, or the capacity to break complex tasks into subtasks.

**The core motivation for building agents:**
- **Problem with LLMs alone:** LLMs cannot access live data (weather, sports scores, current events), perform accurate complex mathematics, retrieve information from internal databases, or handle tasks requiring multiple reasoning steps without significant hallucination.
- **Solution:** Augment LLMs with tools, memory, and orchestration software that coordinates between components.


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 893.25px; background-color: white;" viewBox="0 0 893.25 454" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M515.789,62L515.789,66.167C515.789,70.333,515.789,78.667,515.789,86.333C515.789,94,515.789,101,515.789,104.5L515.789,108" id="L_User_Agent_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_User_Agent_0" data-points="W3sieCI6NTE1Ljc4OTA2MjUsInkiOjYyfSx7IngiOjUxNS43ODkwNjI1LCJ5Ijo4N30seyJ4Ijo1MTUuNzg5MDYyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M453.891,161.015L398.284,170.013C342.677,179.01,231.464,197.005,175.857,211.503C120.25,226,120.25,237,120.25,242.5L120.25,248" id="L_Agent_LLM_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Agent_LLM_0" data-points="W3sieCI6NDUzLjg5MDYyNSwieSI6MTYxLjAxNTQ0NTY5MzE3OTh9LHsieCI6MTIwLjI1LCJ5IjoyMTV9LHsieCI6MTIwLjI1LCJ5IjoyNTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M453.891,184.719L444.626,189.766C435.362,194.813,416.833,204.906,407.569,213.453C398.305,222,398.305,229,398.305,232.5L398.305,236" id="L_Agent_Tools_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Agent_Tools_0" data-points="W3sieCI6NDUzLjg5MDYyNSwieSI6MTg0LjcxOTM3NzU3NjgwNTR9LHsieCI6Mzk4LjMwNDY4NzUsInkiOjIxNX0seyJ4IjozOTguMzA0Njg3NSwieSI6MjQwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M577.688,184.719L586.952,189.766C596.216,194.813,614.745,204.906,624.009,213.453C633.273,222,633.273,229,633.273,232.5L633.273,236" id="L_Agent_Memory_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Agent_Memory_0" data-points="W3sieCI6NTc3LjY4NzUsInkiOjE4NC43MTkzNzc1NzY4MDU0fSx7IngiOjYzMy4yNzM0Mzc1LCJ5IjoyMTV9LHsieCI6NjMzLjI3MzQzNzUsInkiOjI0MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M577.688,164.072L617.88,172.56C658.073,181.048,738.458,198.024,778.651,212.012C818.844,226,818.844,237,818.844,242.5L818.844,248" id="L_Agent_RAG_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Agent_RAG_0" data-points="W3sieCI6NTc3LjY4NzUsInkiOjE2NC4wNzE4OTgxMjA2OTgxfSx7IngiOjgxOC44NDM3NSwieSI6MjE1fSx7IngiOjgxOC44NDM3NSwieSI6MjUyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M120.25,330L120.25,336.167C120.25,342.333,120.25,354.667,170.372,367.423C220.493,380.179,320.736,393.357,370.858,399.946L420.979,406.536" id="L_LLM_Output_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_LLM_Output_0" data-points="W3sieCI6MTIwLjI1LCJ5IjozMzB9LHsieCI6MTIwLjI1LCJ5IjozNjd9LHsieCI6NDI0Ljk0NTMxMjUsInkiOjQwNy4wNTcxMjE0MTI2MjkxfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M398.305,342L398.305,346.167C398.305,350.333,398.305,358.667,407.109,366.73C415.913,374.794,433.521,382.587,442.326,386.484L451.13,390.381" id="L_Tools_Output_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Tools_Output_0" data-points="W3sieCI6Mzk4LjMwNDY4NzUsInkiOjM0Mn0seyJ4IjozOTguMzA0Njg3NSwieSI6MzY3fSx7IngiOjQ1NC43ODc1NjAwOTYxNTM4NywieSI6MzkyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M633.273,342L633.273,346.167C633.273,350.333,633.273,358.667,624.469,366.73C615.665,374.794,598.057,382.587,589.252,386.484L580.448,390.381" id="L_Memory_Output_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Memory_Output_0" data-points="W3sieCI6NjMzLjI3MzQzNzUsInkiOjM0Mn0seyJ4Ijo2MzMuMjczNDM3NSwieSI6MzY3fSx7IngiOjU3Ni43OTA1NjQ5MDM4NDYyLCJ5IjozOTJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M818.844,330L818.844,336.167C818.844,342.333,818.844,354.667,784.132,366.789C749.421,378.912,679.998,390.824,645.287,396.78L610.575,402.736" id="L_RAG_Output_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_RAG_Output_0" data-points="W3sieCI6ODE4Ljg0Mzc1LCJ5IjozMzB9LHsieCI6ODE4Ljg0Mzc1LCJ5IjozNjd9LHsieCI6NjA2LjYzMjgxMjUsInkiOjQwMy40MTI0NjY4MDkzMTE0fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_User_Agent_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Agent_LLM_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Agent_Tools_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Agent_Memory_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Agent_RAG_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_LLM_Output_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Tools_Output_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Memory_Output_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_RAG_Output_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-User-0" transform="translate(515.7890625, 35)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-74.0703125" y="-27" width="148.140625" height="54"/><g class="label" style="" transform="translate(-44.0703125, -12)"><rect/><foreignObject width="88.140625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Prompt</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Agent-1" transform="translate(515.7890625, 151)"><rect class="basic label-container" style="fill:#e6f2ff !important" x="-61.8984375" y="-39" width="123.796875" height="78"/><g class="label" style="" transform="translate(-31.8984375, -24)"><rect/><foreignObject width="63.796875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Agent<br />Software</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-LLM-3" transform="translate(120.25, 291)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-112.25" y="-39" width="224.5" height="78"/><g class="label" style="" transform="translate(-82.25, -24)"><rect/><foreignObject width="164.5" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Language Model<br />(Planning &amp; Reasoning)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Tools-5" transform="translate(398.3046875, 291)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-115.8046875" y="-51" width="231.609375" height="102"/><g class="label" style="" transform="translate(-85.8046875, -36)"><rect/><foreignObject width="171.609375" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Tools<br />(Calculators, APIs,<br />Web Search, Databases)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Memory-7" transform="translate(633.2734375, 291)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-69.1640625" y="-51" width="138.328125" height="102"/><g class="label" style="" transform="translate(-39.1640625, -36)"><rect/><foreignObject width="78.328125" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Memory<br />(Context &amp;<br />History)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-RAG-9" transform="translate(818.84375, 291)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-66.40625" y="-39" width="132.8125" height="78"/><g class="label" style="" transform="translate(-36.40625, -24)"><rect/><foreignObject width="72.8125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Retrieval<br />(Optional)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Output-11" transform="translate(515.7890625, 419)"><rect class="basic label-container" style="fill:#d4edda !important" x="-90.84375" y="-27" width="181.6875" height="54"/><g class="label" style="" transform="translate(-60.84375, -12)"><rect/><foreignObject width="121.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Response to User</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Formal Definition:** An agent is a software system that uses an LLM to pursue goals and complete tasks on behalf of users by combining planning, reasoning, tool calling, and memory management.

### Real-World Agent Applications

Agents are already deployed across many domains:
- **Legal agents:** Automate legal research, contract drafting, and case analysis
- **Sales agents:** Handle prospecting, lead qualification, and customer outreach
- **Phone agents:** Manage end-to-end customer conversations autonomously
- **Computer-using agents:** Access browsers and computers to complete tasks (e.g., OpenAI's Operator)
- **Coding agents:** Perform software development tasks automatically, increasing developer productivity (e.g., OpenAI's Codex)
- **Generic agents:** Handle arbitrary complex tasks through planning and tool use (e.g., ChatGPT's agent mode)

### Why LLMs Fail Without Tools

**Example 1: Live Data**
- Question: "How is the weather in San Francisco today?"
- LLM response (without tools): Either hallucinates or admits it has no access to real-time data
- Solution: Augment with weather API

**Example 2: Complex Mathematics**
- Question: "What is 987654 × 123456?"
- LLM response (without tools): Often incorrect due to reliance on learned weights rather than precise computation
- Solution: Augment with calculator tool

**Example 3: Multi-step Tasks**
- Question: "Write a full report on housing market opportunities"
- LLM response (without tools): Fails due to cognitive overload, context drift, and error propagation across steps
- Solution: Augment with web search, document generation, and information synthesis tools

## Agency Levels Framework

**Agents exist on a spectrum of autonomy**, ranging from simple processors to fully autonomous multi-agent systems. Understanding these levels helps determine which approach is appropriate for a task.


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 951.672px; background-color: white;" viewBox="0 0 951.671875 166" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M172.141,83L176.307,83C180.474,83,188.807,83,196.474,83C204.141,83,211.141,83,214.641,83L218.141,83" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTcyLjE0MDYyNSwieSI6ODN9LHsieCI6MTk3LjE0MDYyNSwieSI6ODN9LHsieCI6MjIyLjE0MDYyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M359.766,83L363.932,83C368.099,83,376.432,83,384.099,83C391.766,83,398.766,83,402.266,83L405.766,83" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzU5Ljc2NTYyNSwieSI6ODN9LHsieCI6Mzg0Ljc2NTYyNSwieSI6ODN9LHsieCI6NDA5Ljc2NTYyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M543.031,83L547.198,83C551.365,83,559.698,83,567.365,83C575.031,83,582.031,83,585.531,83L589.031,83" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTQzLjAzMTI1LCJ5Ijo4M30seyJ4Ijo1NjguMDMxMjUsInkiOjgzfSx7IngiOjU5My4wMzEyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M726.313,83L730.479,83C734.646,83,742.979,83,750.646,83C758.313,83,765.313,83,768.813,83L772.313,83" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NzI2LjMxMjUsInkiOjgzfSx7IngiOjc1MS4zMTI1LCJ5Ijo4M30seyJ4Ijo3NzYuMzEyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(90.0703125, 83)"><rect class="basic label-container" style="fill:#ffcccc !important" x="-82.0703125" y="-63" width="164.140625" height="126"/><g class="label" style="" transform="translate(-52.0703125, -48)"><rect/><foreignObject width="104.140625" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Simple<br />Processor<br /><br />Low Autonomy</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(290.953125, 83)"><rect class="basic label-container" style="fill:#ffe6cc !important" x="-68.8125" y="-63" width="137.625" height="126"/><g class="label" style="" transform="translate(-38.8125, -48)"><rect/><foreignObject width="77.625" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Workflows<br /><br />Predefined<br />Steps</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(476.3984375, 83)"><rect class="basic label-container" style="fill:#ffffcc !important" x="-66.6328125" y="-63" width="133.265625" height="126"/><g class="label" style="" transform="translate(-36.6328125, -48)"><rect/><foreignObject width="73.265625" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Tool<br />Calling<br /><br />With Tools</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(659.671875, 83)"><rect class="basic label-container" style="fill:#e6f2ff !important" x="-66.640625" y="-75" width="133.28125" height="150"/><g class="label" style="" transform="translate(-36.640625, -60)"><rect/><foreignObject width="73.28125" height="120"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Multi-Step<br />Agents<br /><br />Dynamic<br />Planning</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(859.9921875, 83)"><rect class="basic label-container" style="fill:#ccf2ff !important" x="-83.6796875" y="-63" width="167.359375" height="126"/><g class="label" style="" transform="translate(-53.6796875, -48)"><rect/><foreignObject width="107.359375" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Multi-Agent<br />Systems<br /><br />High Autonomy</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


### Level 1: Simple Processor

**Definition:** Software that directly interfaces with an LLM, sending prompts and receiving responses.

**Characteristics:**
- Lowest autonomy level
- No planning or adaptation
- Single LLM call per user prompt
- Handles tokenization, prompt engineering, and response decoding
- Not considered truly "agentic"

**Use case:** Simple question-answering without tool access or multi-step reasoning.

### Level 2: Workflows

**Definition:** Fixed sequence of predefined steps designed by developers to handle complex tasks through orchestration.

**Key characteristic:** Information flows deterministically through predetermined steps; no runtime planning by the agent.

**Advantages:**
- Predictable and consistent outputs
- Better reliability for well-defined problems
- Easier to debug and understand

**Disadvantages:**
- Lacks flexibility for novel or unexpected inputs
- Cannot adapt based on real-time feedback
- Requires developer to know problem structure in advance

### Level 3: Tool Calling

**Definition:** Augmenting LLMs with access to external tools (APIs, databases, code interpreters), allowing LLMs to request tool execution as needed.

**Characteristics:**
- LLM decides when and which tools to use
- Adds live data access and specialized capabilities
- Still primarily follows workflow patterns but with tool augmentation

**Use case:** Questions requiring live data or specialized computation.

### Level 4: Multi-step Agents

**Definition:** Dynamic, iterative systems where agents plan, act, and adapt at runtime based on feedback.

**Characteristics:**
- Agent makes autonomous decisions about steps required
- Continuous feedback loop: Think → Act → Observe → Adapt
- Significantly more flexible than workflows
- Can handle open-ended problems

**Advantages:**
- Handles novel situations
- Adapts to unexpected outcomes
- Suitable for complex, unpredictable tasks

**Disadvantages:**
- Higher costs (more LLM calls)
- Potential for infinite loops or compounding errors
- Less predictable outputs
- Harder to implement reliably

### Level 5: Multi-Agent Systems

**Definition:** Multiple specialized agents coordinate to solve complex problems, validating and improving each other's outputs.

**Characteristics:**
- Central manager agent orchestrates sub-agents
- Each sub-agent is a full multi-step agent itself
- High reliability through validation across agents
- Most autonomous level

**Challenges:**
- Coordination complexity between agents
- Distributed memory management
- Error compounding across agent interactions

## Workflow Patterns

Workflows are the practical foundation for building capable agentic systems. Five primary patterns dominate the landscape, each addressing different task characteristics.


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 955.5px; background-color: white;" viewBox="0 0 955.5 166" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M143.578,83L147.745,83C151.911,83,160.245,83,167.911,83C175.578,83,182.578,83,186.078,83L189.578,83" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTQzLjU3ODEyNSwieSI6ODN9LHsieCI6MTY4LjU3ODEyNSwieSI6ODN9LHsieCI6MTkzLjU3ODEyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M335.313,83L339.479,83C343.646,83,351.979,83,359.646,83C367.313,83,374.313,83,377.813,83L381.313,83" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MzM1LjMxMjUsInkiOjgzfSx7IngiOjM2MC4zMTI1LCJ5Ijo4M30seyJ4IjozODUuMzEyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M527.75,83L531.917,83C536.083,83,544.417,83,552.083,83C559.75,83,566.75,83,570.25,83L573.75,83" id="L_C_D_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C_D_0" data-points="W3sieCI6NTI3Ljc1LCJ5Ijo4M30seyJ4Ijo1NTIuNzUsInkiOjgzfSx7IngiOjU3Ny43NSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M740.641,83L744.807,83C748.974,83,757.307,83,764.974,83C772.641,83,779.641,83,783.141,83L786.641,83" id="L_D_E_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_D_E_0" data-points="W3sieCI6NzQwLjY0MDYyNSwieSI6ODN9LHsieCI6NzY1LjY0MDYyNSwieSI6ODN9LHsieCI6NzkwLjY0MDYyNSwieSI6ODN9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C_D_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_D_E_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(75.7890625, 83)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-67.7890625" y="-75" width="135.578125" height="150"/><g class="label" style="" transform="translate(-37.7890625, -60)"><rect/><foreignObject width="75.578125" height="120"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Prompt<br />Chaining<br /><br />Sequential<br />Steps</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(264.4453125, 83)"><rect class="basic label-container" style="fill:#ffe6cc !important" x="-70.8671875" y="-63" width="141.734375" height="126"/><g class="label" style="" transform="translate(-40.8671875, -48)"><rect/><foreignObject width="81.734375" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Routing<br /><br />Conditional<br />Branching</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(456.53125, 83)"><rect class="basic label-container" style="fill:#ffcccc !important" x="-71.21875" y="-63" width="142.4375" height="126"/><g class="label" style="" transform="translate(-41.21875, -48)"><rect/><foreignObject width="82.4375" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Reflection<br /><br />Iterative<br />Refinement</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-D-5" transform="translate(659.1953125, 83)"><rect class="basic label-container" style="fill:#f0ccff !important" x="-81.4453125" y="-63" width="162.890625" height="126"/><g class="label" style="" transform="translate(-51.4453125, -48)"><rect/><foreignObject width="102.890625" height="96"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Parallelization<br /><br />Concurrent<br />Processing</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-E-7" transform="translate(869.0703125, 83)"><rect class="basic label-container" style="fill:#ccf2ff !important" x="-78.4296875" y="-75" width="156.859375" height="150"/><g class="label" style="" transform="translate(-48.4296875, -60)"><rect/><foreignObject width="96.859375" height="120"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Orchestrator-<br />Workers<br /><br />Distributed<br />Tasks</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


### Pattern 1: Prompt Chaining

**Definition:** Decomposing complex tasks into sequential subtasks, with each LLM call handling a single, simplified objective.

**When to use:**
- Tasks naturally decomposable into sequential steps
- Content generation (outlines → sections → full document)
- Data extraction (raw data → extraction → formatting → validation)
- Information processing (raw data → transformation → analysis → synthesis)

**Trade-offs:**
- **Pro:** Lower cognitive load per LLM call, better accuracy, easier error tracking
- **Con:** Higher latency (sequential calls), increased costs (multiple LLM invocations)

**Example: Market Research Workflow**

Instead of: "Analyze housing market, summarize findings, identify trends, share opportunities"

Break into:
1. Analyze the housing market (single focus)
2. Summarize findings from step 1 (simpler task)
3. Identify trends in summary (narrower scope)
4. Share current opportunities given findings and trends (targeted task)

Each step is significantly simpler, reducing hallucination and error propagation.

### Pattern 2: Routing

**Definition:** Classifying incoming queries by intent and directing them to specialized handlers or models.

**When to use:**
- Distinct task categories exist
- Intent classification is reliable
- Different handlers optimized for different types
- Cost optimization through model selection

**Router Types:**

| Type | Mechanism | Best For |
|------|-----------|----------|
| **Rule-based** | Predefined if-else logic | High-confidence intent patterns |
| **ML-based** | Discriminative classifier | Well-trained classification |
| **Semantic** | Embedding similarity to topic embeddings | Nuanced semantic routing |
| **LLM-based** | LLM performs classification | Complex intent patterns |

**Example: Customer Support**

Router determines intent: "Where is my order?" → Route to shipping specialist
- "Can I get a refund?" → Route to refund specialist
- "The app is crashing" → Route to technical support

**Cost Optimization:** Route simple questions to small, fast, cheap models; complex questions to powerful models.

### Pattern 3: Reflection (Evaluator-Optimizer)

**Definition:** Iterative refinement where a generator produces outputs and an evaluator/critic provides feedback for improvement.

**When to use:**
- Clear evaluation criteria exist
- Iterative refinement helps (writing, planning, code)
- Self-correction benefits the task

**Components:**

- **Generator:** Creates initial solution, remembers past iterations for context
- **Evaluator/Critic:** Examines proposed solution, identifies flaws, provides natural language feedback
- **Feedback Loop:** Generator refines based on evaluator feedback until acceptable

**Best Practice:** Use two specialized LLMs rather than one LLM in both roles to reduce bias and improve robustness.

**Examples:**
- **Code generation:** Evaluator runs tests, identifies bugs; generator fixes them
- **Complex search:** Evaluator decides if more research needed; generator performs additional searches
- **Detailed planning:** Evaluator identifies plan flaws; generator improves plan

### Pattern 4: Parallelization

**Definition:** Simultaneously processing multiple subtasks or multiple attempts, then aggregating results.

**Two Variations:**

**Sectioning:** Divide task into independent subtasks, run in parallel
- Example: "Research GenAI AND recommendation systems" → One agent for GenAI, one for RecSys, aggregate results
- Benefit: Faster processing

**Voting:** Run same task multiple times with different attempts
- Example: "Research image generator state-of-the-art" → Multiple agents independently research, aggregator combines
- Benefit: Robustness through validation (one agent's miss caught by another)

**When to use:**
- Parallel subtasks exist or multiple perspectives help
- Need speed through parallelization
- Robustness from redundant attempts
- Multiple API calls can be parallelized

**Examples:**
- Information gathering (web research)
- Multi-API interactions with latency
- Code review for vulnerabilities
- Research tasks requiring diverse sources

### Pattern 5: Orchestrator-Workers

**Definition:** Central orchestrator coordinates work across multiple specialized worker agents or components.

**Characteristics:**
- Orchestrator determines task breakdown
- Workers execute assigned subtasks independently
- Orchestrator aggregates and validates results

**Less common than other patterns but useful for structured multi-component workflows.**

## Tool Calling and Function Calling

Tool calling is the mechanism enabling LLMs to request execution of external functions, transforming them from pure text predictors into action-capable systems.

### Three-Step Tool Calling Process


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 276px; background-color: white;" viewBox="0 0 276 590" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M138,158L138,162.167C138,166.333,138,174.667,138,182.333C138,190,138,197,138,200.5L138,204" id="L_A_B_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A_B_0" data-points="W3sieCI6MTM4LCJ5IjoxNTh9LHsieCI6MTM4LCJ5IjoxODN9LHsieCI6MTM4LCJ5IjoyMDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M138,358L138,362.167C138,366.333,138,374.667,138,382.333C138,390,138,397,138,400.5L138,404" id="L_B_C_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_B_C_0" data-points="W3sieCI6MTM4LCJ5IjozNTh9LHsieCI6MTM4LCJ5IjozODN9LHsieCI6MTM4LCJ5Ijo0MDh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A_B_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_B_C_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-0" transform="translate(138, 83)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-130" y="-75" width="260" height="150"/><g class="label" style="" transform="translate(-100, -60)"><rect/><foreignObject width="200" height="120"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Step 1: Define Tool<br /><br />def add(x: int, y: int) -&gt; int:<br />    return x + y</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-B-1" transform="translate(138, 283)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-126.40625" y="-75" width="252.8125" height="150"/><g class="label" style="" transform="translate(-96.40625, -60)"><rect/><foreignObject width="192.8125" height="120"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Step 2: Tell LLM About Tool<br /><br />System Prompt:<br />Tool: add<br />Arguments: x(int), y(int)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C-3" transform="translate(138, 495)"><rect class="basic label-container" style="fill:#d4edda !important" x="-130" y="-87" width="260" height="174"/><g class="label" style="" transform="translate(-100, -72)"><rect/><foreignObject width="200" height="144"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>Step 3: Execute When Requested<br /><br />LLM Output: tool_add(5, 3)<br />Software: Parse &amp; Call<br />Return: Result = 8</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


#### Step 1: Define Tools

Tools are implemented as functions with clear specifications:

```python
def add(x: int, y: int) -> int:
    """Add two integers and return their sum."""
    return x + y

def get_forecast(lat: float, lon: float) -> dict:
    """Get weather forecast for coordinates."""
    # Call weather API
    return forecast_data
```

**Requirements:**
- Clear function names describing the tool
- Type hints for arguments and return values
- Descriptive docstrings
- Predictable, reliable execution

#### Step 2: Communicate Tools to LLM

Inform the LLM about available tools via the system prompt:

```
You have access to the following tools:

Tool: add
Description: Add two integers
Arguments: x (int), y (int)
Usage: <tool>add(5, 3)</tool>

Tool: get_forecast
Description: Get weather forecast for given latitude/longitude
Arguments: lat (float), lon (float)
Usage: <tool>get_forecast(37.7749, -122.4194)</tool>
```

**Better Approach:** Use structured prompting format consistently so LLM learns when and how to invoke tools.

#### Step 3: Execute Tools and Return Results

When LLM outputs a tool invocation:

```
User: What is 987654 + 123456?

LLM thinks and outputs:
<tool>add(987654, 123456)</tool>

Software:
1. Parses tool call from LLM output
2. Extracts function name and arguments
3. Calls add(987654, 123456) → 1111110
4. Appends result to context: "Result: 1111110"
5. Passes updated context back to LLM
6. LLM continues generation: "The sum is 1111110"
```

### Tool Calling in Practice: Multiple Tools

As tools increase, manual system prompt management becomes unsustainable:

**Challenge:** Each tool requires English explanation in system prompt; different people explain differently; not scalable.

**Solution: Autoformatting**

Standardize how tools are defined (docstrings, type hints), then automatically extract and format descriptions:

```python
@tool
def calculate_tax(income: float, state: str) -> float:
    """Calculate state income tax given income and state."""
    return income * TAX_RATES[state]
```

Autoformatter extracts: "Tool: calculate_tax, Arguments: income (float), state (str), Returns: float"

This standardization enables scaling to hundreds of tools with consistent formatting.

### Limitations and Solutions

**Problem: Manual Integration**

Different companies write their own wrappers for the same services (Slack API, Gmail, etc.), duplicating effort:

```
n companies × m services = n×m integrations (highly inefficient)
```

**Solution: Model Context Protocol (MCP)**

Rather than each company building Slack integration, Slack itself provides an MCP server. Companies just send requests to the standard interface.

## Model Context Protocol (MCP)

### Definition

**Model Context Protocol** is a universal open standard (introduced by Anthropic, November 25) for connecting AI systems with data sources and tools, eliminating redundant integration work.

### The Integration Problem MCP Solves


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1168.97px; background-color: white;" viewBox="0 0 1168.96875 420" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"/><g class="edgeLabels"/><g class="nodes"><g class="root" transform="translate(0, 36)"><g class="clusters"><g class="cluster" id="After" data-look="classic"><rect style="fill:#ccffcc !important" x="8" y="8" width="648.609375" height="332"/><g class="cluster-label" transform="translate(232.3046875, 8)"><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>AFTER MCP: n + m Integrations</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M184.109,70L190.359,70C196.609,70,209.109,70,226.768,80.385C244.427,90.769,267.245,111.538,278.654,121.923L290.062,132.307" id="L_C4_MCP_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C4_MCP_0" data-points="W3sieCI6MTg0LjEwOTM3NSwieSI6NzB9LHsieCI6MjIxLjYwOTM3NSwieSI6NzB9LHsieCI6MjkzLjAyMDUwNzgxMjUsInkiOjEzNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M184.109,174L190.359,174C196.609,174,209.109,174,220.943,174C232.776,174,243.943,174,249.526,174L255.109,174" id="L_C5_MCP_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C5_MCP_0" data-points="W3sieCI6MTg0LjEwOTM3NSwieSI6MTc0fSx7IngiOjIyMS42MDkzNzUsInkiOjE3NH0seyJ4IjoyNTkuMTA5Mzc1LCJ5IjoxNzR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M184.109,278L190.359,278C196.609,278,209.109,278,226.768,267.615C244.427,257.231,267.245,236.462,278.654,226.077L290.062,215.693" id="L_C6_MCP_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C6_MCP_0" data-points="W3sieCI6MTg0LjEwOTM3NSwieSI6Mjc4fSx7IngiOjIyMS42MDkzNzUsInkiOjI3OH0seyJ4IjoyOTMuMDIwNTA3ODEyNSwieSI6MjEzfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M412.625,174L418.875,174C425.125,174,437.625,174,449.458,174C461.292,174,472.458,174,478.042,174L483.625,174" id="L_MCP_S2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_MCP_S2_0" data-points="W3sieCI6NDEyLjYyNSwieSI6MTc0fSx7IngiOjQ1MC4xMjUsInkiOjE3NH0seyJ4Ijo0ODcuNjI1LCJ5IjoxNzR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_C4_MCP_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C5_MCP_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C6_MCP_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_MCP_S2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-C4-10" transform="translate(114.8046875, 70)"><rect class="basic label-container" style="" x="-69.3046875" y="-27" width="138.609375" height="54"/><g class="label" style="" transform="translate(-39.3046875, -12)"><rect/><foreignObject width="78.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 1</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-MCP-13" transform="translate(335.8671875, 174)"><rect class="basic label-container" style="fill:#e6f2ff !important" x="-76.7578125" y="-39" width="153.515625" height="78"/><g class="label" style="" transform="translate(-46.7578125, -24)"><rect/><foreignObject width="93.515625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>MCP Protocol<br />Standard</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C5-11" transform="translate(114.8046875, 174)"><rect class="basic label-container" style="" x="-69.3046875" y="-27" width="138.609375" height="54"/><g class="label" style="" transform="translate(-39.3046875, -12)"><rect/><foreignObject width="78.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 2</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C6-12" transform="translate(114.8046875, 278)"><rect class="basic label-container" style="" x="-69.3046875" y="-27" width="138.609375" height="54"/><g class="label" style="" transform="translate(-39.3046875, -12)"><rect/><foreignObject width="78.609375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 3</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-S2-14" transform="translate(553.3671875, 174)"><rect class="basic label-container" style="fill:#d4edda !important" x="-65.7421875" y="-39" width="131.484375" height="78"/><g class="label" style="" transform="translate(-35.7421875, -24)"><rect/><foreignObject width="71.484375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Slack MCP<br />Server</p></span></div></foreignObject></g></g></g></g><g class="root" transform="translate(698.609375, 0)"><g class="clusters"><g class="cluster" id="Before" data-look="classic"><rect style="fill:#ffcccc !important" x="8" y="8" width="454.359375" height="404"/><g class="cluster-label" transform="translate(135.1796875, 8)"><foreignObject width="200" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;"><span class="nodeLabel"><p>BEFORE MCP: n × m Integrations</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M226.297,82L232.547,82C238.797,82,251.297,82,270.195,98.307C289.093,114.613,314.389,147.226,327.037,163.533L339.684,179.839" id="L_C1_S1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C1_S1_0" data-points="W3sieCI6MjI2LjI5Njg3NSwieSI6ODJ9LHsieCI6MjYzLjc5Njg3NSwieSI6ODJ9LHsieCI6MzQyLjEzNTk4NjMyODEyNSwieSI6MTgzfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M226.297,210L232.547,210C238.797,210,251.297,210,263.13,210C274.964,210,286.13,210,291.714,210L297.297,210" id="L_C2_S1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C2_S1_0" data-points="W3sieCI6MjI2LjI5Njg3NSwieSI6MjEwfSx7IngiOjI2My43OTY4NzUsInkiOjIxMH0seyJ4IjozMDEuMjk2ODc1LCJ5IjoyMTB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M226.297,338L232.547,338C238.797,338,251.297,338,270.195,321.693C289.093,305.387,314.389,272.774,327.037,256.467L339.684,240.161" id="L_C3_S1_0" class="edge-thickness-normal edge-pattern-dotted edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_C3_S1_0" data-points="W3sieCI6MjI2LjI5Njg3NSwieSI6MzM4fSx7IngiOjI2My43OTY4NzUsInkiOjMzOH0seyJ4IjozNDIuMTM1OTg2MzI4MTI1LCJ5IjoyMzd9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_C1_S1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C2_S1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_C3_S1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-C1-0" transform="translate(135.8984375, 82)"><rect class="basic label-container" style="" x="-90.3984375" y="-39" width="180.796875" height="78"/><g class="label" style="" transform="translate(-60.3984375, -24)"><rect/><foreignObject width="120.796875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 1<br />Slack Integration</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-S1-3" transform="translate(363.078125, 210)"><rect class="basic label-container" style="" x="-61.78125" y="-27" width="123.5625" height="54"/><g class="label" style="" transform="translate(-31.78125, -12)"><rect/><foreignObject width="63.5625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Slack API</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C2-1" transform="translate(135.8984375, 210)"><rect class="basic label-container" style="" x="-90.3984375" y="-39" width="180.796875" height="78"/><g class="label" style="" transform="translate(-60.3984375, -24)"><rect/><foreignObject width="120.796875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 2<br />Slack Integration</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-C3-2" transform="translate(135.8984375, 338)"><rect class="basic label-container" style="" x="-90.3984375" y="-39" width="180.796875" height="78"/><g class="label" style="" transform="translate(-60.3984375, -24)"><rect/><foreignObject width="120.796875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Company 3<br />Slack Integration</p></span></div></foreignObject></g></g></g></g></g></g></g></svg>

</div>


**Before MCP:**
- Company 1 builds Slack wrapper
- Company 2 builds Slack wrapper (duplicate effort)
- Company 3 builds Slack wrapper (duplicate effort)
- Result: n companies × m services = n×m integrations

**After MCP:**
- Slack builds one MCP server
- Companies send requests to Slack's MCP server
- Result: n companies + m services = n+m integrations

### Architecture

**Two-Sided Model:**

**Service Providers (Slack, Google Drive, etc.):**
- Create functions/capabilities
- Expose as MCP servers following standardized protocol
- Handle requests from any client

**Clients (LLMs, IDEs, agents):**
- Create MCP client
- Connect to available MCP servers
- Send requests, receive responses

### Implementation: servers.json

Instead of hardcoding tool integrations, define available servers in a simple configuration:

```json
{
  "servers": [
    {
      "name": "github-mcp",
      "type": "remote",
      "url": "github-mcp.example.com",
      "auth": "token"
    },
    {
      "name": "slack-mcp",
      "type": "remote",
      "url": "slack-mcp.example.com"
    },
    {
      "name": "internal-db",
      "type": "local",
      "command": "python internal_db_server.py"
    }
  ]
}
```

**Workflow:**
1. Software loads servers.json
2. When LLM requests tool usage, software looks up corresponding MCP server
3. Software sends request to MCP server
4. Server executes and returns response
5. Software passes result back to LLM

### Benefits

- **Scalability:** Add tools by updating JSON, no code changes
- **Maintainability:** Each service maintains its own server implementation
- **Standardization:** All interactions follow same protocol
- **Interoperability:** Any LLM can use any MCP server

## Multi-Step Agents

### Motivation: Beyond Workflows

**Workflows Limitations:**
1. Fixed: Predefined execution path cannot adapt to new information
2. No live data: Cannot respond to questions requiring current information (combined with tools, this is addressed)
3. Deterministic: Cannot handle truly open-ended problems

**Multi-step agents solve this** by introducing runtime planning and adaptation.

### Core Principle: Plan-Act-Adapt

Multi-step agents operate on a continuous cycle with three phases:

1. **Understand & Plan:** Agent analyzes situation and goals, determines next steps
2. **Act:** Agent executes planned actions using tools
3. **Observe & Adapt:** Agent incorporates feedback and adjusts strategy

### The Think-Act-Observe Loop


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 295.211px; background-color: white;" viewBox="0 0 295.2109375 852.75" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M174.75,62L174.75,66.167C174.75,70.333,174.75,78.667,174.75,86.333C174.75,94,174.75,101,174.75,104.5L174.75,108" id="L_Start_Think_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Start_Think_0" data-points="W3sieCI6MTc0Ljc1LCJ5Ijo2Mn0seyJ4IjoxNzQuNzUsInkiOjg3fSx7IngiOjE3NC43NSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M133.77,190L129.391,194.167C125.013,198.333,116.257,206.667,111.878,214.333C107.5,222,107.5,229,107.5,232.5L107.5,236" id="L_Think_Act_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Think_Act_0" data-points="W3sieCI6MTMzLjc2OTUzMTI1LCJ5IjoxOTB9LHsieCI6MTA3LjUsInkiOjIxNX0seyJ4IjoxMDcuNSwieSI6MjQwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M107.5,342L107.5,348.167C107.5,354.333,107.5,366.667,107.5,378.333C107.5,390,107.5,401,107.5,406.5L107.5,412" id="L_Act_Observe_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Act_Observe_0" data-points="W3sieCI6MTA3LjUsInkiOjM0Mn0seyJ4IjoxMDcuNSwieSI6Mzc5fSx7IngiOjEwNy41LCJ5Ijo0MTZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M107.5,518L107.5,522.167C107.5,526.333,107.5,534.667,113.332,547.451C119.163,560.235,130.827,577.47,136.659,586.088L142.49,594.705" id="L_Observe_Decision_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Observe_Decision_0" data-points="W3sieCI6MTA3LjUsInkiOjUxOH0seyJ4IjoxMDcuNSwieSI6NTQzfSx7IngiOjE0NC43MzIxODMwNDU3NjE0NSwieSI6NTk4LjAxNzgxNjk1NDIzODZ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M204.768,598.018L210.973,588.848C217.179,579.679,229.589,561.339,235.795,539.503C242,517.667,242,492.333,242,465C242,437.667,242,408.333,242,379C242,349.667,242,320.333,242,293C242,265.667,242,240.333,238.105,223.96C234.209,207.586,226.419,200.172,222.523,196.465L218.628,192.758" id="L_Decision_Think_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Decision_Think_0" data-points="W3sieCI6MjA0Ljc2NzgxNjk1NDIzODU1LCJ5Ijo1OTguMDE3ODE2OTU0MjM4Nn0seyJ4IjoyNDIsInkiOjU0M30seyJ4IjoyNDIsInkiOjQ2N30seyJ4IjoyNDIsInkiOjM3OX0seyJ4IjoyNDIsInkiOjI5MX0seyJ4IjoyNDIsInkiOjIxNX0seyJ4IjoyMTUuNzMwNDY4NzUsInkiOjE5MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M174.75,716.75L174.75,722.917C174.75,729.083,174.75,741.417,174.75,753.083C174.75,764.75,174.75,775.75,174.75,781.25L174.75,786.75" id="L_Decision_End_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Decision_End_0" data-points="W3sieCI6MTc0Ljc1LCJ5Ijo3MTYuNzV9LHsieCI6MTc0Ljc1LCJ5Ijo3NTMuNzV9LHsieCI6MTc0Ljc1LCJ5Ijo3OTAuNzV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_Start_Think_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Think_Act_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Act_Observe_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Observe_Decision_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(242, 379)"><g class="label" data-id="L_Decision_Think_0" transform="translate(-9.3984375, -12)"><foreignObject width="18.796875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>No</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(174.75, 753.75)"><g class="label" data-id="L_Decision_End_0" transform="translate(-11.328125, -12)"><foreignObject width="22.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Yes</p></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-Start-0" transform="translate(174.75, 35)"><rect class="basic label-container" style="fill:#d1ecf1 !important" x="-112.4609375" y="-27" width="224.921875" height="54"/><g class="label" style="" transform="translate(-82.4609375, -12)"><rect/><foreignObject width="164.921875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Start: Understand Goal</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Think-1" transform="translate(174.75, 151)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-108.6484375" y="-39" width="217.296875" height="78"/><g class="label" style="" transform="translate(-78.6484375, -24)"><rect/><foreignObject width="157.296875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>THINK<br />LLM Plans Next Action</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Act-3" transform="translate(107.5, 291)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-85.75" y="-51" width="171.5" height="102"/><g class="label" style="" transform="translate(-55.75, -36)"><rect/><foreignObject width="111.5" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>ACT<br />Execute Tool<br />with Arguments</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Observe-5" transform="translate(107.5, 467)"><rect class="basic label-container" style="fill:#d4edda !important" x="-99.5" y="-51" width="199" height="102"/><g class="label" style="" transform="translate(-69.5, -36)"><rect/><foreignObject width="139" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>OBSERVE<br />Collect Tool Output<br />&amp; Add to Memory</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Decision-7" transform="translate(174.75, 642.375)"><polygon points="74.375,0 148.75,-74.375 74.375,-148.75 0,-74.375" class="label-container" transform="translate(-73.875, 74.375)" style="fill:#fff3cd !important"/><g class="label" style="" transform="translate(-35.375, -24)"><rect/><foreignObject width="70.75" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Goal<br />Achieved?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-End-11" transform="translate(174.75, 817.75)"><rect class="basic label-container" style="fill:#c3e6cb !important" x="-101.265625" y="-27" width="202.53125" height="54"/><g class="label" style="" transform="translate(-71.265625, -12)"><rect/><foreignObject width="142.53125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Return Final Answer</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


The loop operates as follows:

```
Think: LLM plans and decides next action
  ↓
Act: Agent executes tools with arguments
  ↓
Observe: Agent collects results and feedback
  ↓
Think again: LLM adapts strategy based on observations
  (Loop continues until goal achieved)
```

### Detailed Component Breakdown

**Think (Reasoning Phase)**
- LLM analyzes current situation and goal
- Plans sequence of actions to solve task
- Breaks complex problems into manageable steps
- Reflects on past experiences (via context)
- Types of thoughts: planning, decision-making, reflection, goal-setting, prioritization

**Act (Execution Phase)**
- LLM requests specific tool with arguments (special format)
- Agent parses tool request
- Agent calls tool with provided arguments
- Tool executes and returns result

**Observe (Feedback Phase)**
- Collect tool execution result
- Check success or failure
- Append result to context (memory)
- LLM uses result to refine next thought

### System Prompt for Multi-Step Agents

Agents are enabled through carefully crafted system prompts:

```
You are an AI assistant designed to help users efficiently and accurately.

You have access to the following tools:
[tool descriptions]

You should operate as a multi-step agent:
1. Think step-by-step about how to fulfill the objective
2. Structure your reasoning into: Thought | Action | Observation
3. Repeat as needed until task complete

Format:
- Thought: Your reasoning about next step
- Action: tool_name(arg1, arg2)
- Observation: Result from tool

When ready to answer, start with "Final Answer: "
```

### Multi-Step Agent Frameworks

Different papers propose various implementations of the think-act-observe loop:

#### ReAct (Reasoning + Acting)

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 2861.69px; background-color: white;" viewBox="0 0 2861.6875 164" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"/><g class="edgeLabels"/><g class="nodes"><g class="root" transform="translate(0, 0)"><g class="clusters"><g class="cluster" id="ReAct" data-look="classic"><rect style="fill:#e6f2ff !important" x="8" y="8" width="1255.359375" height="148"/><g class="cluster-label" transform="translate(581.28125, 8)"><foreignObject width="108.796875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>ReAct: Optimal</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M144.906,82L151.156,82C157.406,82,169.906,82,181.74,82C193.573,82,204.74,82,210.323,82L215.906,82" id="L_R1_R2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R1_R2_0" data-points="W3sieCI6MTQ0LjkwNjI1LCJ5Ijo4Mn0seyJ4IjoxODIuNDA2MjUsInkiOjgyfSx7IngiOjIxOS45MDYyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M303.609,82L309.859,82C316.109,82,328.609,82,340.443,82C352.276,82,363.443,82,369.026,82L374.609,82" id="L_R2_R3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R2_R3_0" data-points="W3sieCI6MzAzLjYwOTM3NSwieSI6ODJ9LHsieCI6MzQxLjEwOTM3NSwieSI6ODJ9LHsieCI6Mzc4LjYwOTM3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M496.297,82L502.547,82C508.797,82,521.297,82,533.13,82C544.964,82,556.13,82,561.714,82L567.297,82" id="L_R3_R4_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R3_R4_0" data-points="W3sieCI6NDk2LjI5Njg3NSwieSI6ODJ9LHsieCI6NTMzLjc5Njg3NSwieSI6ODJ9LHsieCI6NTcxLjI5Njg3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M670.703,82L676.953,82C683.203,82,695.703,82,707.536,82C719.37,82,730.536,82,736.12,82L741.703,82" id="L_R4_R5_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R4_R5_0" data-points="W3sieCI6NjcwLjcwMzEyNSwieSI6ODJ9LHsieCI6NzA4LjIwMzEyNSwieSI6ODJ9LHsieCI6NzQ1LjcwMzEyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M829.406,82L835.656,82C841.906,82,854.406,82,866.24,82C878.073,82,889.24,82,894.823,82L900.406,82" id="L_R5_R6_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R5_R6_0" data-points="W3sieCI6ODI5LjQwNjI1LCJ5Ijo4Mn0seyJ4Ijo4NjYuOTA2MjUsInkiOjgyfSx7IngiOjkwNC40MDYyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M1022.094,82L1028.344,82C1034.594,82,1047.094,82,1058.927,82C1070.76,82,1081.927,82,1087.51,82L1093.094,82" id="L_R6_Answer3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_R6_Answer3_0" data-points="W3sieCI6MTAyMi4wOTM3NSwieSI6ODJ9LHsieCI6MTA1OS41OTM3NSwieSI6ODJ9LHsieCI6MTA5Ny4wOTM3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_R1_R2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_R2_R3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_R3_R4_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_R4_R5_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_R5_R6_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_R6_Answer3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-R1-8" transform="translate(95.203125, 82)"><rect class="basic label-container" style="" x="-49.703125" y="-27" width="99.40625" height="54"/><g class="label" style="" transform="translate(-19.703125, -12)"><rect/><foreignObject width="39.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R2-9" transform="translate(261.7578125, 82)"><rect class="basic label-container" style="" x="-41.8515625" y="-27" width="83.703125" height="54"/><g class="label" style="" transform="translate(-11.8515625, -12)"><rect/><foreignObject width="23.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R3-10" transform="translate(437.453125, 82)"><rect class="basic label-container" style="" x="-58.84375" y="-27" width="117.6875" height="54"/><g class="label" style="" transform="translate(-28.84375, -12)"><rect/><foreignObject width="57.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Observe</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R4-11" transform="translate(621, 82)"><rect class="basic label-container" style="" x="-49.703125" y="-27" width="99.40625" height="54"/><g class="label" style="" transform="translate(-19.703125, -12)"><rect/><foreignObject width="39.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R5-12" transform="translate(787.5546875, 82)"><rect class="basic label-container" style="" x="-41.8515625" y="-27" width="83.703125" height="54"/><g class="label" style="" transform="translate(-11.8515625, -12)"><rect/><foreignObject width="23.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-R6-13" transform="translate(963.25, 82)"><rect class="basic label-container" style="" x="-58.84375" y="-27" width="117.6875" height="54"/><g class="label" style="" transform="translate(-28.84375, -12)"><rect/><foreignObject width="57.6875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Observe</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Answer3-14" transform="translate(1161.4765625, 82)"><rect class="basic label-container" style="fill:#ccffcc !important" x="-64.3828125" y="-39" width="128.765625" height="78"/><g class="label" style="" transform="translate(-34.3828125, -24)"><rect/><foreignObject width="68.765625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answer<br />(Optimal)</p></span></div></foreignObject></g></g></g></g><g class="root" transform="translate(1305.359375, 0)"><g class="clusters"><g class="cluster" id="PureAct" data-look="classic"><rect style="" x="8" y="8" width="697.46875" height="148"/><g class="cluster-label" transform="translate(315.9375, 8)"><foreignObject width="81.59375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Pure Acting</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M129.203,82L135.453,82C141.703,82,154.203,82,166.036,82C177.87,82,189.036,82,194.62,82L200.203,82" id="L_A1_A2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A1_A2_0" data-points="W3sieCI6MTI5LjIwMzEyNSwieSI6ODJ9LHsieCI6MTY2LjcwMzEyNSwieSI6ODJ9LHsieCI6MjA0LjIwMzEyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M287.906,82L294.156,82C300.406,82,312.906,82,324.74,82C336.573,82,347.74,82,353.323,82L358.906,82" id="L_A2_A3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A2_A3_0" data-points="W3sieCI6Mjg3LjkwNjI1LCJ5Ijo4Mn0seyJ4IjozMjUuNDA2MjUsInkiOjgyfSx7IngiOjM2Mi45MDYyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M446.609,82L452.859,82C459.109,82,471.609,82,483.443,82C495.276,82,506.443,82,512.026,82L517.609,82" id="L_A3_Answer2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A3_Answer2_0" data-points="W3sieCI6NDQ2LjYwOTM3NSwieSI6ODJ9LHsieCI6NDg0LjEwOTM3NSwieSI6ODJ9LHsieCI6NTIxLjYwOTM3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A1_A2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A2_A3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A3_Answer2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A1-4" transform="translate(87.3515625, 82)"><rect class="basic label-container" style="" x="-41.8515625" y="-27" width="83.703125" height="54"/><g class="label" style="" transform="translate(-11.8515625, -12)"><rect/><foreignObject width="23.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-A2-5" transform="translate(246.0546875, 82)"><rect class="basic label-container" style="" x="-41.8515625" y="-27" width="83.703125" height="54"/><g class="label" style="" transform="translate(-11.8515625, -12)"><rect/><foreignObject width="23.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-A3-6" transform="translate(404.7578125, 82)"><rect class="basic label-container" style="" x="-41.8515625" y="-27" width="83.703125" height="54"/><g class="label" style="" transform="translate(-11.8515625, -12)"><rect/><foreignObject width="23.703125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Answer2-7" transform="translate(594.7890625, 82)"><rect class="basic label-container" style="fill:#ffcccc !important" x="-73.1796875" y="-39" width="146.359375" height="78"/><g class="label" style="" transform="translate(-43.1796875, -24)"><rect/><foreignObject width="86.359375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answer<br />(Inefficient)</p></span></div></foreignObject></g></g></g></g><g class="root" transform="translate(2052.828125, 0)"><g class="clusters"><g class="cluster" id="PureThink" data-look="classic"><rect style="" x="8" y="8" width="792.859375" height="148"/><g class="cluster-label" transform="translate(350.2421875, 8)"><foreignObject width="108.375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Pure Reasoning</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M144.906,82L151.156,82C157.406,82,169.906,82,181.74,82C193.573,82,204.74,82,210.323,82L215.906,82" id="L_T1_T2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_T1_T2_0" data-points="W3sieCI6MTQ0LjkwNjI1LCJ5Ijo4Mn0seyJ4IjoxODIuNDA2MjUsInkiOjgyfSx7IngiOjIxOS45MDYyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M319.313,82L325.563,82C331.813,82,344.313,82,356.146,82C367.979,82,379.146,82,384.729,82L390.313,82" id="L_T2_T3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_T2_T3_0" data-points="W3sieCI6MzE5LjMxMjUsInkiOjgyfSx7IngiOjM1Ni44MTI1LCJ5Ijo4Mn0seyJ4IjozOTQuMzEyNSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M493.719,82L499.969,82C506.219,82,518.719,82,530.552,82C542.385,82,553.552,82,559.135,82L564.719,82" id="L_T3_Answer1_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_T3_Answer1_0" data-points="W3sieCI6NDkzLjcxODc1LCJ5Ijo4Mn0seyJ4Ijo1MzEuMjE4NzUsInkiOjgyfSx7IngiOjU2OC43MTg3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_T1_T2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_T2_T3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_T3_Answer1_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-T1-0" transform="translate(95.203125, 82)"><rect class="basic label-container" style="" x="-49.703125" y="-27" width="99.40625" height="54"/><g class="label" style="" transform="translate(-19.703125, -12)"><rect/><foreignObject width="39.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-T2-1" transform="translate(269.609375, 82)"><rect class="basic label-container" style="" x="-49.703125" y="-27" width="99.40625" height="54"/><g class="label" style="" transform="translate(-19.703125, -12)"><rect/><foreignObject width="39.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-T3-2" transform="translate(444.015625, 82)"><rect class="basic label-container" style="" x="-49.703125" y="-27" width="99.40625" height="54"/><g class="label" style="" transform="translate(-19.703125, -12)"><rect/><foreignObject width="39.40625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Answer1-3" transform="translate(666.0390625, 82)"><rect class="basic label-container" style="fill:#ffcccc !important" x="-97.3203125" y="-39" width="194.640625" height="78"/><g class="label" style="" transform="translate(-67.3203125, -24)"><rect/><foreignObject width="134.640625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Answer<br />(Limited Accuracy)</p></span></div></foreignObject></g></g></g></g></g></g></g></svg>

</div>


**Approach:** Simple prompting technique that interleaves reasoning and action at each step.

**Key insight:** Unlike pure reasoning (think only) or pure action (act only), ReAct combines both:
- **Reason only:** Insufficient for tasks requiring external information
- **Act only:** May miss strategic considerations
- **ReAct:** Reason → Act → Observe → Reason → Act...

**Example flow:**
```
Thought: I need to search for information about front row media center.
Action: search("front row media center")
Observation: Could not find information. Let me search more specifically.
Thought: The search failed. I should try "front row software".
Action: search("front row software")
Observation: Front row is a discontinued media center software for Windows.
Thought: Now I have the information. I can provide the answer.
Final Answer: Front row is discontinued media center software...
```

**Why effective:** Balances exploration and reasoning, allowing model to adjust strategy based on feedback.

#### Reflexion

**Paper:** "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)

**Approach:** Improve agent through self-reflection by converting tool feedback into natural language summaries.

**Key difference from ReAct:** Feedback is processed through LLM to generate natural language summaries, which the agent then uses for reflection.

**Process:**
1. Agent attempts task
2. Tool returns feedback (success/failure/error)
3. LLM converts feedback to natural language summary
4. Summary becomes observation for next think phase

**Benefit:** Natural language summaries are often more interpretable and useful than raw error messages.

#### ReWOO (Reason, Worker, Optimizer)

**Paper:** "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models" (2023)

**Approach:** Modular setup with separate components for reasoning, execution, and optimization.

**Components:**
- **Planner:** Generates sequence of actions (reasoning)
- **Worker:** Executes actions and collects results (execution)
- **Solver:** Synthesizes final answer from collected information (optimization)

**Key insight:** Decoupling reasoning from observations allows more efficient planning without per-step feedback loops.

**Comparison with ReAct:** ReWOO plans upfront then executes, while ReAct interleaves planning and execution. ReWOO can be more efficient for tasks with clear structure.

#### Tree Search

**Paper:** "Tree Search for Language Model Agents" (2024)

**Approach:** Use explicit search algorithms (BFS, DFS) to explore different action sequences and choose best path.

**Key idea:** Rather than single linear path of actions, explore multiple branches:
- Consider different action options at each step
- Evaluate which branch seems most promising
- Backtrack if path leads to dead end
- Choose path with highest expected success

**Benefit:** More thorough exploration of solution space, better handling of ambiguous situations.

**Trade-off:** More expensive (multiple branches evaluated) but potentially better quality solutions.

### Workflow vs. Multi-Step Agents Comparison


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 1866.39px; background-color: white;" viewBox="0 0 1866.390625 238.0078125" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"/><g class="edgeLabels"/><g class="nodes"><g class="root" transform="translate(0, 0)"><g class="clusters"><g class="cluster" id="Agent" data-look="classic"><rect style="fill:#ccf2ff !important" x="8" y="8" width="1183.640625" height="222.0078125"/><g class="cluster-label" transform="translate(517.90625, 8)"><foreignObject width="163.828125" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>AGENTS: Dynamic Path</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M229.828,103.205L236.078,101.421C242.328,99.638,254.828,96.071,266.661,94.287C278.495,92.504,289.661,92.504,295.245,92.504L300.828,92.504" id="L_A1_A2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A1_A2_0" data-points="W3sieCI6MjI5LjgyODEyNSwieSI6MTAzLjIwNDYzNTI5NzQxODJ9LHsieCI6MjY3LjMyODEyNSwieSI6OTIuNTAzOTA2MjV9LHsieCI6MzA0LjgyODEyNSwieSI6OTIuNTAzOTA2MjV9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M462.078,92.504L470.216,92.504C478.354,92.504,494.63,92.504,510.24,92.504C525.849,92.504,540.792,92.504,548.263,92.504L555.734,92.504" id="L_A2_A3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A2_A3_0" data-points="W3sieCI6NDYyLjA3ODEyNSwieSI6OTIuNTAzOTA2MjV9LHsieCI6NTEwLjkwNjI1LCJ5Ijo5Mi41MDM5MDYyNX0seyJ4Ijo1NTkuNzM0Mzc1LCJ5Ijo5Mi41MDM5MDYyNX1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M736.453,92.504L742.703,92.504C748.953,92.504,761.453,92.504,776.081,95.683C790.709,98.862,807.464,105.22,815.842,108.399L824.22,111.578" id="L_A3_Loop_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_A3_Loop_0" data-points="W3sieCI6NzM2LjQ1MzEyNSwieSI6OTIuNTAzOTA2MjV9LHsieCI6NzczLjk1MzEyNSwieSI6OTIuNTAzOTA2MjV9LHsieCI6ODI3Ljk1OTg5MTU2NzkyNywieSI6MTEyLjk5NzEzOTY4MjA3Mjk2fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M931.469,129.504L939.285,129.504C947.102,129.504,962.734,129.504,977.701,129.504C992.667,129.504,1006.966,129.504,1014.116,129.504L1021.266,129.504" id="L_Loop_Done_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Loop_Done_0" data-points="W3sieCI6OTMxLjQ2ODc1LCJ5IjoxMjkuNTAzOTA2MjV9LHsieCI6OTc4LjM2NzE4NzUsInkiOjEyOS41MDM5MDYyNX0seyJ4IjoxMDI1LjI2NTYyNSwieSI6MTI5LjUwMzkwNjI1fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M827.96,146.011L818.959,149.426C809.958,152.842,791.955,159.673,761.978,163.088C732,166.504,690.047,166.504,646.206,166.504C602.365,166.504,556.635,166.504,512.529,166.504C468.422,166.504,425.938,166.504,385.341,166.504C344.745,166.504,306.036,166.504,281.073,164.903C256.11,163.303,244.892,160.102,239.284,158.501L233.675,156.901" id="L_Loop_A1_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Loop_A1_0" data-points="W3sieCI6ODI3Ljk1OTg5MTU2NzkyNywieSI6MTQ2LjAxMDY3MjgxNzkyNzA2fSx7IngiOjc3My45NTMxMjUsInkiOjE2Ni41MDM5MDYyNX0seyJ4Ijo2NDguMDkzNzUsInkiOjE2Ni41MDM5MDYyNX0seyJ4Ijo1MTAuOTA2MjUsInkiOjE2Ni41MDM5MDYyNX0seyJ4IjozODMuNDUzMTI1LCJ5IjoxNjYuNTAzOTA2MjV9LHsieCI6MjY3LjMyODEyNSwieSI6MTY2LjUwMzkwNjI1fSx7IngiOjIyOS44MjgxMjUsInkiOjE1NS44MDMxNzcyMDI1ODE4fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_A1_A2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A2_A3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_A3_Loop_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(978.3671875, 129.50390625)"><g class="label" data-id="L_Loop_Done_0" transform="translate(-9.3984375, -12)"><foreignObject width="18.796875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>No</p></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(510.90625, 166.50390625)"><g class="label" data-id="L_Loop_A1_0" transform="translate(-11.328125, -12)"><foreignObject width="22.65625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"><p>Yes</p></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A1-6" transform="translate(137.6640625, 129.50390625)"><rect class="basic label-container" style="" x="-92.1640625" y="-39" width="184.328125" height="78"/><g class="label" style="" transform="translate(-62.1640625, -24)"><rect/><foreignObject width="124.328125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Think:<br />Analyze Situation</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-A2-7" transform="translate(383.453125, 92.50390625)"><rect class="basic label-container" style="" x="-78.625" y="-39" width="157.25" height="78"/><g class="label" style="" transform="translate(-48.625, -24)"><rect/><foreignObject width="97.25" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Act:<br />Execute Tools</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-A3-8" transform="translate(648.09375, 92.50390625)"><rect class="basic label-container" style="" x="-88.359375" y="-39" width="176.71875" height="78"/><g class="label" style="" transform="translate(-58.359375, -24)"><rect/><foreignObject width="116.71875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Observe:<br />Evaluate Results</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Loop-9" transform="translate(871.4609375, 129.50390625)"><polygon points="60.0078125,0 120.015625,-60.0078125 60.0078125,-120.015625 0,-60.0078125" class="label-container" transform="translate(-59.5078125, 60.0078125)"/><g class="label" style="" transform="translate(-21.0078125, -24)"><rect/><foreignObject width="42.015625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Adapt<br />Path?</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Done-15" transform="translate(1089.703125, 129.50390625)"><rect class="basic label-container" style="fill:#d4edda !important" x="-64.4375" y="-27" width="128.875" height="54"/><g class="label" style="" transform="translate(-34.4375, -12)"><rect/><foreignObject width="68.875" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Complete</p></span></div></foreignObject></g></g></g></g><g class="root" transform="translate(1233.640625, 37.00390625)"><g class="clusters"><g class="cluster" id="Workflow" data-look="classic"><rect style="fill:#ffe6cc !important" x="8" y="8" width="616.75" height="148"/><g class="cluster-label" transform="translate(227.703125, 8)"><foreignObject width="177.34375" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>WORKFLOWS: Fixed Path</p></span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M195.109,82L201.359,82C207.609,82,220.109,82,231.943,82C243.776,82,254.943,82,260.526,82L266.109,82" id="L_W1_W2_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_W1_W2_0" data-points="W3sieCI6MTk1LjEwOTM3NSwieSI6ODJ9LHsieCI6MjMyLjYwOTM3NSwieSI6ODJ9LHsieCI6MjcwLjEwOTM3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M401.484,82L407.734,82C413.984,82,426.484,82,438.318,82C450.151,82,461.318,82,466.901,82L472.484,82" id="L_W2_W3_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_W2_W3_0" data-points="W3sieCI6NDAxLjQ4NDM3NSwieSI6ODJ9LHsieCI6NDM4Ljk4NDM3NSwieSI6ODJ9LHsieCI6NDc2LjQ4NDM3NSwieSI6ODJ9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_W1_W2_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_W2_W3_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-W1-0" transform="translate(120.3046875, 82)"><rect class="basic label-container" style="" x="-74.8046875" y="-39" width="149.609375" height="78"/><g class="label" style="" transform="translate(-44.8046875, -24)"><rect/><foreignObject width="89.609375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Step 1:<br />Extract Data</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-W2-1" transform="translate(335.796875, 82)"><rect class="basic label-container" style="" x="-65.6875" y="-39" width="131.375" height="78"/><g class="label" style="" transform="translate(-35.6875, -24)"><rect/><foreignObject width="71.375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Step 2:<br />Transform</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-W3-2" transform="translate(531.8671875, 82)"><rect class="basic label-container" style="" x="-55.3828125" y="-39" width="110.765625" height="78"/><g class="label" style="" transform="translate(-25.3828125, -24)"><rect/><foreignObject width="50.765625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Step 3:<br />Load</p></span></div></foreignObject></g></g></g></g></g></g></g></svg>

</div>


| Aspect | Workflows | Multi-Step Agents |
|--------|-----------|------------------|
| **Predictability** | High - fixed path ensures consistent outputs | Lower - dynamic planning can vary |
| **Adaptability** | Low - cannot adjust to unexpected inputs | High - adapts at runtime |
| **Implementation** | Simpler - predetermined steps | More complex - requires planning |
| **Reliability** | More reliable for well-defined tasks | Less reliable but more flexible |
| **Cost** | Lower - fewer LLM calls typically | Higher - more exploration possible |
| **Best for** | Defined tasks with clear workflows | Open-ended or unpredictable problems |

**Practical guideline:** Use workflows when solution is well understood and repeatable; use agents for open-ended or complex problems.

### When to Use Multi-Step Agents

**Use agents when:**
- Problem is not well-structured
- Multiple possible solution paths exist
- Unexpected inputs may require adaptation
- Complex reasoning and planning required

**Examples:**
- **Software development:** Multiple ways to solve coding problem; agent adapts based on test results
- **Research:** Open-ended exploration of topics
- **Robotics & autonomous navigation:** No predefined path; agent navigates based on observations
- **Structured information synthesis:** Deep research on topics

**Use workflows when:**
- Problem solution is well understood
- Task is repeatable with consistent structure
- Requirements are fixed and predictable

## Multi-Agent Systems

### Motivation

Single agents may:
- Fail or move in incorrect directions
- Miss important information or perspectives
- Lack specialized expertise for different subtasks

**Solution:** Multiple specialized agents coordinate and validate each other's work.

### Architecture


<div class="mermaid-diagram">

<svg id="my-svg" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="flowchart" style="max-width: 848.945px; background-color: white;" viewBox="0 0 848.9453125 478" role="graphics-document document" aria-roledescription="flowchart-v2"><style>#my-svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#my-svg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#my-svg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#my-svg .error-icon{fill:#552222;}#my-svg .error-text{fill:#552222;stroke:#552222;}#my-svg .edge-thickness-normal{stroke-width:1px;}#my-svg .edge-thickness-thick{stroke-width:3.5px;}#my-svg .edge-pattern-solid{stroke-dasharray:0;}#my-svg .edge-thickness-invisible{stroke-width:0;fill:none;}#my-svg .edge-pattern-dashed{stroke-dasharray:3;}#my-svg .edge-pattern-dotted{stroke-dasharray:2;}#my-svg .marker{fill:#333333;stroke:#333333;}#my-svg .marker.cross{stroke:#333333;}#my-svg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#my-svg p{margin:0;}#my-svg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#my-svg .cluster-label text{fill:#333;}#my-svg .cluster-label span{color:#333;}#my-svg .cluster-label span p{background-color:transparent;}#my-svg .label text,#my-svg span{fill:#333;color:#333;}#my-svg .node rect,#my-svg .node circle,#my-svg .node ellipse,#my-svg .node polygon,#my-svg .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#my-svg .rough-node .label text,#my-svg .node .label text,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-anchor:middle;}#my-svg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#my-svg .rough-node .label,#my-svg .node .label,#my-svg .image-shape .label,#my-svg .icon-shape .label{text-align:center;}#my-svg .node.clickable{cursor:pointer;}#my-svg .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#my-svg .arrowheadPath{fill:#333333;}#my-svg .edgePath .path{stroke:#333333;stroke-width:2.0px;}#my-svg .flowchart-link{stroke:#333333;fill:none;}#my-svg .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#my-svg .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#my-svg .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#my-svg .cluster text{fill:#333;}#my-svg .cluster span{color:#333;}#my-svg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#my-svg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#my-svg rect.text{fill:none;stroke-width:0;}#my-svg .icon-shape,#my-svg .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#my-svg .icon-shape p,#my-svg .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#my-svg .icon-shape rect,#my-svg .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#my-svg .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#my-svg .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#my-svg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker id="my-svg_flowchart-v2-pointEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-pointStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="4.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleEnd" class="marker flowchart-v2" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-circleStart" class="marker flowchart-v2" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossEnd" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><marker id="my-svg_flowchart-v2-crossStart" class="marker cross flowchart-v2" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"/></marker><g class="root"><g class="clusters"/><g class="edgePaths"><path d="M486.078,62L486.078,66.167C486.078,70.333,486.078,78.667,486.078,86.333C486.078,94,486.078,101,486.078,104.5L486.078,108" id="L_User_Manager_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_User_Manager_0" data-points="W3sieCI6NDg2LjA3ODEyNSwieSI6NjJ9LHsieCI6NDg2LjA3ODEyNSwieSI6ODd9LHsieCI6NDg2LjA3ODEyNSwieSI6MTEyfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M403.523,163.996L349.526,172.497C295.529,180.998,187.534,197.999,133.536,209.999C79.539,222,79.539,229,79.539,232.5L79.539,236" id="L_Manager_Web_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Manager_Web_0" data-points="W3sieCI6NDAzLjUyMzQzNzUsInkiOjE2My45OTYyOTExMDA1NjMwN30seyJ4Ijo3OS41MzkwNjI1LCJ5IjoyMTV9LHsieCI6NzkuNTM5MDYyNSwieSI6MjQwfV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M403.523,177.585L384.159,183.821C364.794,190.056,326.065,202.528,306.701,212.264C287.336,222,287.336,229,287.336,232.5L287.336,236" id="L_Manager_DB_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Manager_DB_0" data-points="W3sieCI6NDAzLjUyMzQzNzUsInkiOjE3Ny41ODQ2OTI3OTQ1MjgwOH0seyJ4IjoyODcuMzM1OTM3NSwieSI6MjE1fSx7IngiOjI4Ny4zMzU5Mzc1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M568.633,175.154L591.331,181.795C614.029,188.436,659.424,201.718,682.122,211.859C704.82,222,704.82,229,704.82,232.5L704.82,236" id="L_Manager_Analysis_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Manager_Analysis_0" data-points="W3sieCI6NTY4LjYzMjgxMjUsInkiOjE3NS4xNTQwMDU1MDAxOTY0M30seyJ4Ijo3MDQuODIwMzEyNSwieSI6MjE1fSx7IngiOjcwNC44MjAzMTI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M79.539,318L79.539,322.167C79.539,326.333,79.539,334.667,82.077,342.454C84.615,350.241,89.69,357.483,92.228,361.104L94.766,364.724" id="L_Web_WebTools_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Web_WebTools_0" data-points="W3sieCI6NzkuNTM5MDYyNSwieSI6MzE4fSx7IngiOjc5LjUzOTA2MjUsInkiOjM0M30seyJ4Ijo5Ny4wNjE5MzQ2MjE3MTA1MiwieSI6MzY4fV0=" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M287.336,318L287.336,322.167C287.336,326.333,287.336,334.667,300.622,345.22C313.909,355.773,340.482,368.546,353.768,374.932L367.055,381.318" id="L_DB_DBTools_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_DB_DBTools_0" data-points="W3sieCI6Mjg3LjMzNTkzNzUsInkiOjMxOH0seyJ4IjoyODcuMzM1OTM3NSwieSI6MzQzfSx7IngiOjM3MC42NjAxNTYyNSwieSI6MzgzLjA1MTI4ODM4NTk5N31d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M704.82,318L704.82,322.167C704.82,326.333,704.82,334.667,707.356,342.454C709.891,350.241,714.961,357.482,717.497,361.103L720.032,364.723" id="L_Analysis_AnalysisTools_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Analysis_AnalysisTools_0" data-points="W3sieCI6NzA0LjgyMDMxMjUsInkiOjMxOH0seyJ4Ijo3MDQuODIwMzEyNSwieSI6MzQzfSx7IngiOjcyMi4zMjY0ODAyNjMxNTc5LCJ5IjozNjh9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M168.555,368L171.476,363.833C174.396,359.667,180.237,351.333,183.158,336.5C186.078,321.667,186.078,300.333,186.078,279C186.078,257.667,186.078,236.333,221.667,218.074C257.256,199.815,328.434,184.631,364.023,177.039L399.611,169.446" id="L_WebTools_Manager_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_WebTools_Manager_0" data-points="W3sieCI6MTY4LjU1NTI1Mjg3ODI4OTQ4LCJ5IjozNjh9LHsieCI6MTg2LjA3ODEyNSwieSI6MzQzfSx7IngiOjE4Ni4wNzgxMjUsInkiOjI3OX0seyJ4IjoxODYuMDc4MTI1LCJ5IjoyMTV9LHsieCI6NDAzLjUyMzQzNzUsInkiOjE2OC42MTE2NjY2NjY2NjY2OH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M520.238,383.051L534.126,376.376C548.013,369.701,575.788,356.35,589.675,339.009C603.563,321.667,603.563,300.333,603.563,279C603.563,257.667,603.563,236.333,596.499,221.819C589.436,207.305,575.309,199.609,568.246,195.761L561.183,191.914" id="L_DBTools_Manager_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_DBTools_Manager_0" data-points="W3sieCI6NTIwLjIzODI4MTI1LCJ5IjozODMuMDUxMjg4Mzg1OTk3fSx7IngiOjYwMy41NjI1LCJ5IjozNDN9LHsieCI6NjAzLjU2MjUsInkiOjI3OX0seyJ4Ijo2MDMuNTYyNSwieSI6MjE1fSx7IngiOjU1Ny42NzAxNjYwMTU2MjUsInkiOjE5MH1d" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M792.014,368L794.789,363.833C797.565,359.667,803.117,351.333,805.892,336.5C808.668,321.667,808.668,300.333,808.668,279C808.668,257.667,808.668,236.333,769.316,217.859C729.964,199.386,651.26,183.771,611.908,175.964L572.556,168.157" id="L_AnalysisTools_Manager_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_AnalysisTools_Manager_0" data-points="W3sieCI6NzkyLjAxMzcyMzI3MzAyNjQsInkiOjM2OH0seyJ4Ijo4MDguNjY3OTY4NzUsInkiOjM0M30seyJ4Ijo4MDguNjY3OTY4NzUsInkiOjI3OX0seyJ4Ijo4MDguNjY3OTY4NzUsInkiOjIxNX0seyJ4Ijo1NjguNjMyODEyNSwieSI6MTY3LjM3ODM4Mjk2MDE3MzR9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/><path d="M486.078,190L486.078,194.167C486.078,198.333,486.078,206.667,486.078,214.333C486.078,222,486.078,229,486.078,232.5L486.078,236" id="L_Manager_FinalAnswer_0" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" style=";" data-edge="true" data-et="edge" data-id="L_Manager_FinalAnswer_0" data-points="W3sieCI6NDg2LjA3ODEyNSwieSI6MTkwfSx7IngiOjQ4Ni4wNzgxMjUsInkiOjIxNX0seyJ4Ijo0ODYuMDc4MTI1LCJ5IjoyNDB9XQ==" marker-end="url(#my-svg_flowchart-v2-pointEnd)"/></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" data-id="L_User_Manager_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Manager_Web_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Manager_DB_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Manager_Analysis_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Web_WebTools_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_DB_DBTools_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Analysis_AnalysisTools_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_WebTools_Manager_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_DBTools_Manager_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_AnalysisTools_Manager_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" data-id="L_Manager_FinalAnswer_0" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" class="labelBkg" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-User-0" transform="translate(486.078125, 35)"><rect class="basic label-container" style="fill:#d1ecf1 !important" x="-69.5078125" y="-27" width="139.015625" height="54"/><g class="label" style="" transform="translate(-39.5078125, -12)"><rect/><foreignObject width="79.015625" height="24"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>User Query</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Manager-1" transform="translate(486.078125, 151)"><rect class="basic label-container" style="fill:#f8d7da !important" x="-82.5546875" y="-39" width="165.109375" height="78"/><g class="label" style="" transform="translate(-52.5546875, -24)"><rect/><foreignObject width="105.109375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Manager Agent<br />(Orchestrator)</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Web-3" transform="translate(79.5390625, 279)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-71.5390625" y="-39" width="143.078125" height="78"/><g class="label" style="" transform="translate(-41.5390625, -24)"><rect/><foreignObject width="83.078125" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Web Search<br />Sub-Agent</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-DB-5" transform="translate(287.3359375, 279)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-66.2578125" y="-39" width="132.515625" height="78"/><g class="label" style="" transform="translate(-36.2578125, -24)"><rect/><foreignObject width="72.515625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Database<br />Sub-Agent</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-Analysis-7" transform="translate(704.8203125, 279)"><rect class="basic label-container" style="fill:#fff3cd !important" x="-66.2578125" y="-39" width="132.515625" height="78"/><g class="label" style="" transform="translate(-36.2578125, -24)"><rect/><foreignObject width="72.515625" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Analysis<br />Sub-Agent</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-WebTools-9" transform="translate(132.80859375, 419)"><rect class="basic label-container" style="" x="-78.21875" y="-51" width="156.4375" height="102"/><g class="label" style="" transform="translate(-48.21875, -36)"><rect/><foreignObject width="96.4375" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Tools:<br />Bing, Google,<br />DuckDuckGo</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-DBTools-11" transform="translate(445.44921875, 419)"><rect class="basic label-container" style="" x="-74.7890625" y="-51" width="149.578125" height="102"/><g class="label" style="" transform="translate(-44.7890625, -36)"><rect/><foreignObject width="89.578125" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Tools:<br />SQL, NoSQL,<br />Internal APIs</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-AnalysisTools-13" transform="translate(758.0390625, 419)"><rect class="basic label-container" style="" x="-82.90625" y="-51" width="165.8125" height="102"/><g class="label" style="" transform="translate(-52.90625, -36)"><rect/><foreignObject width="105.8125" height="72"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Tools:<br />NLP, Statistics,<br />Visualization</p></span></div></foreignObject></g></g><g class="node default" id="flowchart-FinalAnswer-21" transform="translate(486.078125, 279)"><rect class="basic label-container" style="fill:#d4edda !important" x="-82.484375" y="-39" width="164.96875" height="78"/><g class="label" style="" transform="translate(-52.484375, -24)"><rect/><foreignObject width="104.96875" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>Synthesized<br />Final Response</p></span></div></foreignObject></g></g></g></g></g></svg>

</div>


**Manager Agent** (Orchestrator)
- Central coordinator
- Reasons about which sub-agents to activate
- Synthesizes final result from sub-agents
- Manages communication between agents

**Sub-Agents** (Specialists)
- Autonomous multi-step agents themselves
- Each handles specific domain/task
- Examples: web search agent, database query agent, analysis agent
- Each maintains own memory and planning

### Example: Anthropic's Multi-Agent Research System

**Components:**
- Lead agent: Orchestrates research process
- Search sub-agents: Perform parallel web searches
- Analysis agents: Process and analyze information
- Iterative refinement loop

**Result:** More comprehensive, accurate, and reliable research output through division of expertise.

### Challenges in Multi-Agent Systems

**Coordination Challenge**
- Different agents must share state and synchronize actions
- Deciding when to activate which agents
- Managing hand-offs between agents

**Memory Management**
- Each agent has own memory
- Useful information from one agent's memory may not be visible to others
- Central memory or communication protocol needed

**Compounding Errors**
- Error from one agent propagates to dependent agents
- In loops, errors can amplify
- Multi-agent amplification of errors worse than single-agent

**Complexity**
- Designing reliable multi-agent systems is significantly harder
- Even advanced labs approach with caution
- Requires extensive engineering and testing

## Agent-to-Agent Protocol (A2A)

### Definition

**Agent-to-Agent (A2A) Protocol** is a standardized protocol announced by Google (April 2025) enabling agents to communicate with each other, discover capabilities, and coordinate work.

### Parallel to MCP

Just as **MCP standardizes tool-agent communication**, **A2A standardizes agent-agent communication**:

- **MCP:** Standardizes how agents interact with tools
- **A2A:** Standardizes how agents interact with other agents

### Benefits

- **Interoperability:** Agents from different systems can work together
- **Discovery:** Agents can discover each other's capabilities
- **Scalability:** New agents can be added without modifying existing ones
- **Standardization:** Reduces redundant agent-to-agent integration work

### Design Principles

- Open protocol for agent discovery and communication
- Standardized message format for agent-to-agent requests
- Capability advertisement and negotiation
- Error handling and fallback mechanisms

### Broader Protocol Ecosystem

The field is developing multiple complementary protocols:
- **MCP:** Tool-to-agent communication
- **A2A:** Agent-to-agent communication
- Other emerging proposals for specific coordination patterns

(See survey in references for comprehensive protocol overview)

## Evaluation of Agents

### Key Metrics

**Token Consumption**
- Measures average tokens used per request
- Important because: More tokens = higher cost
- Goal: Maximize capability per token
- Trade-off: Planning tokens vs. solution tokens

**Tool Execution Success Rate**
- Percentage of tool calls that execute successfully
- Indicates: System setup quality and LLM reliability
- Concern: Failed tool calls can derail agent
- Improvement: Better error handling and tool descriptions

**Observability & Debuggability**
- How easily can you determine what went wrong?
- Why it matters: Complex agent behavior hard to debug
- Metrics: Can you trace decisions, identify bottlenecks, find errors?

**Task Success Rate**
- Percentage of tasks completed successfully
- Requires: Good benchmarks with correct solutions
- Domain-specific: Different metrics for different domains

### Agent Leaderboards

Leaderboards exist for evaluating agents across domains:
- Software development benchmarks
- Banking and finance tasks
- Healthcare applications
- Insurance automation
- General reasoning tasks

Leaderboards enable comparison of different agents and tracking progress in the field.

## Implementation Frameworks

In practice, building agents is simpler than the theory suggests due to mature frameworks and libraries.

### LangChain

**Features:**
- Simplifies LLM application development
- Built-in tool management with `@tool` decorator
- Easy tool binding: `llm.bind_tools(tools)`
- Tool invocation: `tool.invoke(arguments)`
- Large collection of pre-built tools (search, code execution, productivity)

**Example:**
```python
from langchain.tools import tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

llm_with_tools = llm.bind_tools([add, multiply])
```

### OpenAI Agents SDK

**Features:**
- Purpose-built for agent implementation
- Simplified agent creation
- Good integration with OpenAI models
- Examples and tutorials for common patterns

### Google Agent SDK

**Features:**
- Agent system implementation
- Integration with Google services
- Specialized tools for Google Cloud

### Key Takeaway

While frameworks exist, the complexity is manageable. Most tools are pre-built, and adding new functions requires only a few lines of code. The barrier to building working agents is lower than theory might suggest.

## Key Insights

1. **Agent Spectrum:** Autonomy exists on a spectrum. Start with workflows for predictable tasks, progress to agents when unpredictability or adaptation is needed.

2. **Tool Access is Transformative:** Adding tool access (calculator, web search, APIs, databases) fundamentally changes what LLMs can accomplish. This is often sufficient for many tasks without full agentic systems.

3. **Standardization Matters:** MCP and A2A show the importance of standardization - eliminating n×m integration problems through agreed protocols.

4. **Trade-offs:** Agents offer flexibility at the cost of predictability, cost, and complexity. Workflows offer predictability and lower cost. Choose based on task requirements.

5. **Multi-step Reasoning:** Think-Act-Observe loops enable agents to plan, adapt, and recover from failures. This pattern (ReAct, Reflexion, ReWOO, Tree Search) is fundamental to all advanced agents.

6. **Multiple Perspectives Help:** Multi-agent systems and voting patterns (in parallelization) show that multiple attempts and perspectives improve robustness and accuracy.

7. **Error Compounding:** In iterative systems, errors compound. In multi-step agents, a single error can propagate. In multi-agent systems, errors can amplify. Plan accordingly with error handling and monitoring.

8. **Evaluation is Challenging:** Unlike LLMs, agent evaluation requires multiple metrics (token consumption, tool success rate, task success rate) and domain-specific benchmarks. Single metrics are insufficient.

9. **Implementation is Practical:** Despite theoretical complexity, mature frameworks make agent implementation straightforward for developers. The gap between theory and practice is manageable.

10. **Domain Matters:** Different domains require different agency levels. Coding benefits from agents; simple FAQs work fine with tools or workflows. Match agency level to problem characteristics.

## Quick Recall / Implementation Checklist

- [ ] **Define the problem:** Is this task well-structured (use workflow) or open-ended (use agent)?

- [ ] **Choose agency level:** Identify minimum autonomy needed - simple processor, workflow with tools, multi-step agent, or multi-agent system?

- [ ] **Tool planning:** List all tools/APIs needed. Check if MCP servers exist; if not, plan wrapper implementation.

- [ ] **Workflow design:** If using workflows, map out sequence of steps and design each component's responsibility.

- [ ] **Prompt engineering:** Write clear system prompt with tool descriptions, expected format, and reasoning guidelines.

- [ ] **Function calling setup:** Ensure tools have type hints, docstrings, and clear descriptions for automatic formatting.

- [ ] **Error handling:** Plan for tool failures, implement retry logic, and add validation.

- [ ] **Memory management:** Design how state/context is maintained across steps, especially for agents.

- [ ] **Testing:** Test each workflow step independently before integration; test agent loops with various inputs.

- [ ] **Evaluation setup:** Define success metrics appropriate for your use case (token consumption, tool success rate, task completion).

- [ ] **Monitoring:** Implement observability to debug agent behavior - log thoughts, actions, observations.

- [ ] **Iteration:** Start simple (workflows), add complexity only when needed; iterate based on performance metrics.

- [ ] **Framework selection:** Choose LangChain, OpenAI SDK, or Google SDK based on your stack and requirements.

- [ ] **MCP adoption:** For tools beyond single company, prefer MCP servers over manual integration.

- [ ] **Scaling plan:** Plan how system grows as new tools/agents added - ensure architecture supports scaling.

- [ ] **Cost optimization:** Track token usage, route simple queries to smaller models, cache common operations.

- [ ] **Reliability requirements:** Higher reliability needs suggest workflows; flexibility needs suggest agents; multi-agent validates results.

- [ ] **Feedback loops:** For agents, implement reflection patterns for iterative improvement based on tool feedback.

---

## References

### Papers

- **ReAct:** "ReAct: Synergizing Reasoning and Acting in Language Models" - https://arxiv.org/abs/2210.03629

- **Reflexion:** "Reflexion: Language Agents with Verbal Reinforcement Learning" - https://arxiv.org/abs/2303.11366

- **ReWOO:** "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models" - https://arxiv.org/abs/2305.18323

- **Tree Search for Language Model Agents:** - https://arxiv.org/abs/2407.01476

- **AI Agents Protocol Survey:** "A survey of AI agents protocol" - https://arxiv.org/abs/2504.16736

### Anthropic Resources

- **Building Effective Agents:** https://www.anthropic.com/engineering/building-effective-agents

- **Building Effective Agents Cookbook:** https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents

- **Model Context Protocol (MCP):** https://www.anthropic.com/news/model-context-protocol

- **Multi-Agent Research System:** https://www.anthropic.com/engineering/multi-agent-research-system

### OpenAI Resources

- **Introducing Operator:** https://openai.com/index/introducing-operator/

- **A Practical Guide to Building Agents:** https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

- **OpenAI Agents SDK:** https://openai.github.io/openai-agents-python/

- **Fine-tuning for Function Calling:** https://cookbook.openai.com/examples/fine_tuning_for_function_calling

### Tool & Framework Documentation

- **LangChain Tools:** https://python.langchain.com/docs/integrations/tools/

- **LangChain Function Calling:** https://python.langchain.com/docs/how_to/function_calling/

- **Google Agent-to-Agent (A2A) Protocol:** https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/

### Agent Evaluation

- **Agent Leaderboard:** https://github.com/rungalileo/agent-leaderboard

### Additional Guides

- **Hugging Face Agent Course:** Agent explanations and tutorials

- **Hugging Face Agents vs Workflows:** Comparison of when to use each approach
