# Agents

## Table of Contents

* [Overview](#overview)
* [Core Concepts](#core-concepts)
  * [What are Agents?](#what-are-agents)
  * [Real-World Agent Applications](#real-world-agent-applications)
  * [Why LLMs Fail Without Tools](#why-llms-fail-without-tools)
* [Agency Levels Framework](#agency-levels-framework)
  * [Level 1: Simple Processor](#level-1-simple-processor)
  * [Level 2: Workflows](#level-2-workflows)
  * [Level 3: Tool Calling](#level-3-tool-calling)
  * [Level 4: Multi-step Agents](#level-4-multi-step-agents)
  * [Level 5: Multi-Agent Systems](#level-5-multi-agent-systems)
* [Workflow Patterns](#workflow-patterns)
  * [Pattern 1: Prompt Chaining](#pattern-1-prompt-chaining)
  * [Pattern 2: Routing](#pattern-2-routing)
  * [Pattern 3: Reflection (Evaluator-Optimizer)](#pattern-3-reflection-evaluator-optimizer)
  * [Pattern 4: Parallelization](#pattern-4-parallelization)
  * [Pattern 5: Orchestrator-Workers](#pattern-5-orchestrator-workers)
* [Tool Calling and Function Calling](#tool-calling-and-function-calling)
  * [Three-Step Tool Calling Process](#three-step-tool-calling-process)
  * [Tool Calling in Practice: Multiple Tools](#tool-calling-in-practice-multiple-tools)
  * [Limitations and Solutions](#limitations-and-solutions)
* [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  * [Definition](#definition)
  * [The Integration Problem MCP Solves](#the-integration-problem-mcp-solves)
  * [Architecture](#architecture)
  * [Implementation: servers.json](#implementation-serversjson)
  * [Benefits](#benefits)
* [Multi-Step Agents](#multi-step-agents)
  * [Motivation: Beyond Workflows](#motivation-beyond-workflows)
  * [Core Principle: Plan-Act-Adapt](#core-principle-plan-act-adapt)
  * [The Think-Act-Observe Loop](#the-think-act-observe-loop)
  * [Detailed Component Breakdown](#detailed-component-breakdown)
  * [System Prompt for Multi-Step Agents](#system-prompt-for-multi-step-agents)
  * [Multi-Step Agent Frameworks](#multi-step-agent-frameworks)
  * [Workflow vs. Multi-Step Agents Comparison](#workflow-vs-multi-step-agents-comparison)
  * [When to Use Multi-Step Agents](#when-to-use-multi-step-agents)
* [Multi-Agent Systems](#multi-agent-systems)
  * [Motivation](#motivation)
  * [Architecture](#architecture-1)
  * [Example: Anthropic's Multi-Agent Research System](#example-anthropics-multi-agent-research-system)
  * [Challenges in Multi-Agent Systems](#challenges-in-multi-agent-systems)
* [Agent-to-Agent Protocol (A2A)](#agent-to-agent-protocol-a2a)
  * [Definition](#definition-1)
  * [Parallel to MCP](#parallel-to-mcp)
  * [Benefits](#benefits-1)
  * [Design Principles](#design-principles)
  * [Broader Protocol Ecosystem](#broader-protocol-ecosystem)
* [Evaluation of Agents](#evaluation-of-agents)
  * [Key Metrics](#key-metrics)
  * [Agent Leaderboards](#agent-leaderboards)
* [Implementation Frameworks](#implementation-frameworks)
  * [LangChain](#langchain)
  * [OpenAI Agents SDK](#openai-agents-sdk)
  * [Google Agent SDK](#google-agent-sdk)
  * [Key Takeaway](#key-takeaway)
* [Key Insights](#key-insights)
* [Quick Recall / Implementation Checklist](#quick-recall--implementation-checklist)
* [References](#references)

---

**TL;DR**: Agents augment LLMs with tools, planning, and multi-step reasoning. Five agency levels: simple processor → workflows → tool calling → multi-step agents → multi-agent systems. Five workflow patterns: prompt chaining, routing, reflection, parallelization, orchestrator-workers. Tool calling via MCP (Model Context Protocol) eliminates n×m integration problems. Multi-step agents use Think-Act-Observe loops (ReAct, Reflexion, ReWOO, Tree Search) for dynamic planning. Multi-agent systems coordinate specialists for validation. Trade-off: workflows for predictability, agents for flexibility. Production uses LangChain/OpenAI SDK, evaluates on token consumption, tool success rate, and task completion.

---

## Overview

This lecture notes covers the design and operation of AI agents, building upon LLMs to create autonomous systems capable of complex task completion. Rather than limiting AI to simple next-token prediction, agents augment LLMs with tools, planning capabilities, and multi-step reasoning. The focus spans from fundamental concepts through workflow patterns, tool integration, multi-step reasoning frameworks, and multi-agent coordination. Understanding agents requires grasping their spectrum of autonomy levels, from simple processors to fully autonomous multi-agent systems. This progression shows how to systematically add agency to LLMs, enabling them to handle tasks that single LLMs cannot.

## Core Concepts

### What are Agents?

At the most fundamental level, **agents are software systems that augment LLMs with additional capabilities** to make them more powerful. LLMs, by themselves, are static models trained on internet data that excel at predicting the next token but lack autonomy, planning ability, or the capacity to break complex tasks into subtasks.

**The core motivation for building agents:**
* **Problem with LLMs alone:** LLMs cannot access live data (weather, sports scores, current events), perform accurate complex mathematics, retrieve information from internal databases, or handle tasks requiring multiple reasoning steps without significant hallucination.
* **Solution:** Augment LLMs with tools, memory, and orchestration software that coordinates between components.

![Diagram 1](https://kroki.io/mermaid/svg/eNpVj02LwjAQhu_-ihDPZcGDB1mEfggeXJBV8FA8dOOkDTvNlCQi_fe2Y9U4h8DwzPPOpHZV14jd70wMlZbyYNoO4fvPfa33jhR4T447fnZ0E-k1kKW2l2eRJGuRlfJE7l8j3fx7cO_gAtpYuHB7CNB5eeYlGWt5KY9EyDSvEI2t3_bJhEaM-Onk7BSl_LliMMkYx4NpDTZEa4veVq1RjxOwsnaInSIKjtg8I9h83Nb7AG0UsjV1E_1yxroPPYJIhTaIq7nWaqgIZC8Ayw-Qv8DoRKCYACz1QusIbCag1Ajuw4KBqg==)

**Formal Definition:** An agent is a software system that uses an LLM to pursue goals and complete tasks on behalf of users by combining planning, reasoning, tool calling, and memory management.

### Real-World Agent Applications

Agents are already deployed across many domains:
* **Legal agents:** Automate legal research, contract drafting, and case analysis
* **Sales agents:** Handle prospecting, lead qualification, and customer outreach
* **Phone agents:** Manage end-to-end customer conversations autonomously
* **Computer-using agents:** Access browsers and computers to complete tasks (e.g., OpenAI's Operator)
* **Coding agents:** Perform software development tasks automatically, increasing developer productivity (e.g., OpenAI's Codex)
* **Generic agents:** Handle arbitrary complex tasks through planning and tool use (e.g., ChatGPT's agent mode)

### Why LLMs Fail Without Tools

**Example 1: Live Data**
* Question: "How is the weather in San Francisco today?"
* LLM response (without tools): Either hallucinates or admits it has no access to real-time data
* Solution: Augment with weather API

**Example 2: Complex Mathematics**
* Question: "What is 987654 × 123456?"
* LLM response (without tools): Often incorrect due to reliance on learned weights rather than precise computation
* Solution: Augment with calculator tool

**Example 3: Multi-step Tasks**
* Question: "Write a full report on housing market opportunities"
* LLM response (without tools): Fails due to cognitive overload, context drift, and error propagation across steps
* Solution: Augment with web search, document generation, and information synthesis tools

## Agency Levels Framework

**Agents exist on a spectrum of autonomy**, ranging from simple processors to fully autonomous multi-agent systems. Understanding these levels helps determine which approach is appropriate for a task.

> **Key Insight**: Agency exists on a spectrum, not as binary. Match autonomy level to task characteristics: simple processor for basic Q&A, workflows for predictable tasks, tool calling for live data needs, multi-step agents for open-ended problems, multi-agent systems for complex validation. Higher autonomy = more flexibility but less predictability and higher costs.

| Level | Autonomy | Planning | Tool Access | Adaptability | Cost | Reliability | Best For |
|-------|----------|----------|-------------|--------------|------|-------------|----------|
| **1. Simple Processor** | None | None | ❌ No | None | Lowest | Highest | Basic Q&A |
| **2. Workflows** | Low | Fixed | ⚠️ Optional | None | Low | High | Repeatable tasks |
| **3. Tool Calling** | Low | Fixed | ✅ Yes | Low | Medium | High | Live data access |
| **4. Multi-step Agents** | High | Dynamic | ✅ Yes | High | High | Medium | Open-ended problems |
| **5. Multi-Agent Systems** | Highest | Distributed | ✅ Yes | Highest | Highest | Variable | Complex validation |

### Agency Level Decision Tree

![Agency Level Decision Tree](https://kroki.io/mermaid/svg/eNp9klFr2zAUhd_3K-77asbW7SWMljZZ16xZGaRQRsiDIl_bl1xLniTHCfb-eyXZxikb9ZslnU_nnqPciKqAp8U78N_N5rkQDtypQtAZOGH311tIkiu4bZc2_kODzIl1ppauNph-3ZkPVw25AvZKNwqs5tqRVlAJV1z_jdjbgOh-o72AyktIOrFj7GDePiKmwHRASIUTkaUN4NGhUYLBac32FeRRX4CuUCWoUkw7WLRrUjkjSF1WjMfoceQY_FOTQQsHweQv8LY8LNLmA62_ApT3EXDfNis8IMPHGawpAOGX0RKt1WaE9gc-zeBZm33GutmeAeOMAdZzO7gLuTXDyYgwWKGIAYxe7iZpRkevtQ4rr_2--ecueA9PAfzazOUsrsJcMPs0tmfUkFd6UqIkOWLvB-znGfys2VES1uEmR-W2vaFFlP4n2A6W7VwoEOGwj1fqA_bBZEaXUJJ1Yu_zFqwVDr31sFj0VEMHPwYXX0YX0QCsT95NGZG2QkleYZ31Y0_aweRySs0iZ4mts4wkeUgHD2-NOIlDOKEsC3nt4Ur6N7lq7ykv_GxMYkdM7jS0Fp9S6QljbavJgDTkSAr2Y51tBbyQEqvYNhgKAT68AGDgI7g=)

![Diagram 2](https://kroki.io/mermaid/svg/eNp1kUFrwkAQhe_-imELUkEptKUtpQhSoS0oilp6CB5GMxsD627YnbT13zeZrBAr5pDZzDfvZV6SeSx2sBp3oLo-A_lE1XeYe7cvWK1hMBjCKCPLiZLysvE3w6XT_IOe1LojQiEyOplMEzVBm5WYEUxdSkYU13OD1uY2gy4sCIOrz71KfypfOWdCoqQ0ulc029IgOx_6MJp_hL70v2gDS0K_3fVhjIwbDBTO_aa0d_6QqKZGR2eZfhm68vieh8r7cC5djN4StSD2OX1jDDErOHcWTe8YvIorw7OSi5Lr-VA4GwjYydeMrpKnNSjNZqf_3eq17Zb0Ah8MxdV0bszzFT3oW61bUH5aw7TWd9u0xRqrSNN7SlNs0TpCFD6ljyeoWfsCjOtfoHWMNvoD0sO9AQ==)

### Level 1: Simple Processor

**Definition:** Software that directly interfaces with an LLM, sending prompts and receiving responses.

**Characteristics:**
* Lowest autonomy level
* No planning or adaptation
* Single LLM call per user prompt
* Handles tokenization, prompt engineering, and response decoding
* Not considered truly "agentic"

**Use case:** Simple question-answering without tool access or multi-step reasoning.

### Level 2: Workflows

**Definition:** Fixed sequence of predefined steps designed by developers to handle complex tasks through orchestration.

**Key characteristic:** Information flows deterministically through predetermined steps; no runtime planning by the agent.

**Advantages:**
* Predictable and consistent outputs
* Better reliability for well-defined problems
* Easier to debug and understand

**Disadvantages:**
* Lacks flexibility for novel or unexpected inputs
* Cannot adapt based on real-time feedback
* Requires developer to know problem structure in advance

### Level 3: Tool Calling

**Definition:** Augmenting LLMs with access to external tools (APIs, databases, code interpreters), allowing LLMs to request tool execution as needed.

**Characteristics:**
* LLM decides when and which tools to use
* Adds live data access and specialized capabilities
* Still primarily follows workflow patterns but with tool augmentation

**Use case:** Questions requiring live data or specialized computation.

### Level 4: Multi-step Agents

**Definition:** Dynamic, iterative systems where agents plan, act, and adapt at runtime based on feedback.

**Characteristics:**
* Agent makes autonomous decisions about steps required
* Continuous feedback loop: Think → Act → Observe → Adapt
* Significantly more flexible than workflows
* Can handle open-ended problems

**Advantages:**
* Handles novel situations
* Adapts to unexpected outcomes
* Suitable for complex, unpredictable tasks

**Disadvantages:**
* Higher costs (more LLM calls)
* Potential for infinite loops or compounding errors
* Less predictable outputs
* Harder to implement reliably

### Level 5: Multi-Agent Systems

**Definition:** Multiple specialized agents coordinate to solve complex problems, validating and improving each other's outputs.

**Characteristics:**
* Central manager agent orchestrates sub-agents
* Each sub-agent is a full multi-step agent itself
* High reliability through validation across agents
* Most autonomous level

**Challenges:**
* Coordination complexity between agents
* Distributed memory management
* Error compounding across agent interactions

## Workflow Patterns

Workflows are the practical foundation for building capable agentic systems. Five primary patterns dominate the landscape, each addressing different task characteristics.

![Diagram 3](https://kroki.io/mermaid/svg/eNplkVtLAzEQhd_9FUME34oURaVIZW3XC_Yi7SrI0odtMtsNpklJsr1g_e9mp1tZMS-BOec7M5MsbLYqIOmfQDhTn1mfMro68KYFWuczLeDRZIrNoNXqQlJI_Zmy5Ol59HI7t-fdwWAIryrTDka49RBxL41mMwokM2GhnLKolxASb5GXHiExRlFhI30BkV2US9Te1XBACB3PHdo1pmx8P40n7zERPaMUBkMVAePSr0pP9TOIhABvYIhLY3d1VB1BcX3k0oURv1i1FUERLySuUdyxb7IfLZV_PzL7wx7_pQ90e4i1SNkEfWk1PEidKYi026ANrYlwfqewfolcKtU5zW_EtcgaYrVpLeX5BRcN6Tj5QRaXKP6QoXkt8Qu84vOGRL945NrI83ZD_F2j2fYHPEilQA==)

### Pattern 1: Prompt Chaining

**Definition:** Decomposing complex tasks into sequential subtasks, with each LLM call handling a single, simplified objective.

**When to use:**
* Tasks naturally decomposable into sequential steps
* Content generation (outlines → sections → full document)
* Data extraction (raw data → extraction → formatting → validation)
* Information processing (raw data → transformation → analysis → synthesis)

**Trade-offs:**
* **Pro:** Lower cognitive load per LLM call, better accuracy, easier error tracking
* **Con:** Higher latency (sequential calls), increased costs (multiple LLM invocations)

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
* Distinct task categories exist
* Intent classification is reliable
* Different handlers optimized for different types
* Cost optimization through model selection

**Router Types:**

| Type | Mechanism | Best For |
|------|-----------|----------|
| **Rule-based** | Predefined if-else logic | High-confidence intent patterns |
| **ML-based** | Discriminative classifier | Well-trained classification |
| **Semantic** | Embedding similarity to topic embeddings | Nuanced semantic routing |
| **LLM-based** | LLM performs classification | Complex intent patterns |

**Example: Customer Support**

Router determines intent: "Where is my order?" → Route to shipping specialist
* "Can I get a refund?" → Route to refund specialist
* "The app is crashing" → Route to technical support

**Cost Optimization:** Route simple questions to small, fast, cheap models; complex questions to powerful models.

### Pattern 3: Reflection (Evaluator-Optimizer)

**Definition:** Iterative refinement where a generator produces outputs and an evaluator/critic provides feedback for improvement.

**When to use:**
* Clear evaluation criteria exist
* Iterative refinement helps (writing, planning, code)
* Self-correction benefits the task

**Components:**

* **Generator:** Creates initial solution, remembers past iterations for context
* **Evaluator/Critic:** Examines proposed solution, identifies flaws, provides natural language feedback
* **Feedback Loop:** Generator refines based on evaluator feedback until acceptable

**Best Practice:** Use two specialized LLMs rather than one LLM in both roles to reduce bias and improve robustness.

**Examples:**
* **Code generation:** Evaluator runs tests, identifies bugs; generator fixes them
* **Complex search:** Evaluator decides if more research needed; generator performs additional searches
* **Detailed planning:** Evaluator identifies plan flaws; generator improves plan

### Pattern 4: Parallelization

**Definition:** Simultaneously processing multiple subtasks or multiple attempts, then aggregating results.

**Two Variations:**

**Sectioning:** Divide task into independent subtasks, run in parallel
* Example: "Research GenAI AND recommendation systems" → One agent for GenAI, one for RecSys, aggregate results
* Benefit: Faster processing

**Voting:** Run same task multiple times with different attempts
* Example: "Research image generator state-of-the-art" → Multiple agents independently research, aggregator combines
* Benefit: Robustness through validation (one agent's miss caught by another)

**When to use:**
* Parallel subtasks exist or multiple perspectives help
* Need speed through parallelization
* Robustness from redundant attempts
* Multiple API calls can be parallelized

**Examples:**
* Information gathering (web research)
* Multi-API interactions with latency
* Code review for vulnerabilities
* Research tasks requiring diverse sources

### Pattern 5: Orchestrator-Workers

**Definition:** Central orchestrator coordinates work across multiple specialized worker agents or components.

**Characteristics:**
* Orchestrator determines task breakdown
* Workers execute assigned subtasks independently
* Orchestrator aggregates and validates results

**Less common than other patterns but useful for structured multi-component workflows.**

## Tool Calling and Function Calling

Tool calling is the mechanism enabling LLMs to request execution of external functions, transforming them from pure text predictors into action-capable systems.

### Three-Step Tool Calling Process

![Diagram 4](https://kroki.io/mermaid/svg/eNpdj1FLwzAUhd_9FYcKsuGG6BRH0EG7-TZxbAMfZEi23GyFrK3pDbb_3iSrOMxDLpdzOd85eyurA9azC_iXfiQrpgq3AjPSeUFYl6V52tqbSfwUaUileo1AXvAAbZx9DCdhingTfCyxswUaXKNNNhh6Peus7wTWZAzm81ek29LxP8SqrZmOWNjyWHWO4UAEbtxSu3dHKrgWaHqB7mPEmWxihyziph1uJPDS0M4x4f1ABZb05cgT1B8xJHlzXDkWYI_6DA0fBhj1T4FKzd_SksBC2ppwhak0p8DLWFN409oZxjPGPkMMUXNrCCl0boy41GP1qOSZkP0KWo926kyYdoK6J6XkD8Q6fpw=)

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
* Clear function names describing the tool
* Type hints for arguments and return values
* Descriptive docstrings
* Predictable, reliable execution

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

![Diagram 5](https://kroki.io/mermaid/svg/eNqFks1qg0AUhfd5iotdBlsyWhehBFQMZFEqMTvpYuLMpCFmJkymhTxJH6gv1utPyLVCcxcDnvOp5x5mZ_npAzbJBHDOn9td-5xIZawsvSRbvq0zeE3zOWj4-YYjrLSTCLm90WfvvX2tmXRWeqk5nri-wOxla58WRc2rA8UpzW40u08HNzq4SxeYpPPjfIU6iQj-o79AgOQYS8FAklpMht3EyklbevFyk62vzUz_KSYkxVD9mVRA9YgsS3T8U-nhAbk1zlSm7opwXAtuBd2fXfdHuoOk_ZJ2UEUIPu6IAMkzlqKR1ARotIL9acddatlfG1D7up4_KFXhELctrjerqrGJ2Xy4s2SkmFLEKljviFAKwX8BfzK13A==)

**Before MCP:**
* Company 1 builds Slack wrapper
* Company 2 builds Slack wrapper (duplicate effort)
* Company 3 builds Slack wrapper (duplicate effort)
* Result: n companies × m services = n×m integrations

**After MCP:**
* Slack builds one MCP server
* Companies send requests to Slack's MCP server
* Result: n companies + m services = n+m integrations

> **Key Insight**: Model Context Protocol (MCP) eliminates the n×m integration problem. Instead of every company building wrappers for every service, each service provides one MCP server that all companies use. This standardization reduces integration work from n×m to n+m - exponential to linear savings. Enables scaling to hundreds of tools without code changes.

### Architecture

**Two-Sided Model:**

**Service Providers (Slack, Google Drive, etc.):**
* Create functions/capabilities
* Expose as MCP servers following standardized protocol
* Handle requests from any client

**Clients (LLMs, IDEs, agents):**
* Create MCP client
* Connect to available MCP servers
* Send requests, receive responses

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

* **Scalability:** Add tools by updating JSON, no code changes
* **Maintainability:** Each service maintains its own server implementation
* **Standardization:** All interactions follow same protocol
* **Interoperability:** Any LLM can use any MCP server

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

![Diagram 6](https://kroki.io/mermaid/svg/eNpVj8FqwzAQRO_5CuGcTUsKPZQSiO0cCoGWtNCDyUFZr-IliuSu5EL79d2qbmzrIBi9mR3tiXXXqt1-oeRs6uyF_aWLj0e-WZetJkfulES6XvGjRxdJ2z8ZsQvZQeX5WhV1tvd9nNlL7xqK5N3gL1g7aMWSHVJdkZKlJNFYhF_nGH6KyDrSJyYlDnJ4kfIhW6ZsJR_WrK1FS996PkDaoWeWSJKyGGAIY3uVJmzr7JmhxRClzXOevO-ez8hhnFWRcDr2EZuk33Q4y-aLNCjEL4tqowxZ-7A0xtxBMwHFFeA9wASUVwAwA9U_uAUwZgK2A5DnlTE_TPGNiQ==)

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
* LLM analyzes current situation and goal
* Plans sequence of actions to solve task
* Breaks complex problems into manageable steps
* Reflects on past experiences (via context)
* Types of thoughts: planning, decision-making, reflection, goal-setting, prioritization

**Act (Execution Phase)**
* LLM requests specific tool with arguments (special format)
* Agent parses tool request
* Agent calls tool with provided arguments
* Tool executes and returns result

**Observe (Feedback Phase)**
* Collect tool execution result
* Check success or failure
* Append result to context (memory)
* LLM uses result to refine next thought

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
* Thought: Your reasoning about next step
* Action: tool_name(arg1, arg2)
* Observation: Result from tool

When ready to answer, start with "Final Answer: "

```

### Multi-Step Agent Frameworks

Different papers propose various implementations of the think-act-observe loop:

#### ReAct (Reasoning + Acting)

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)

![Diagram 7](https://kroki.io/mermaid/svg/eNqFUlFLwzAQft-vCPVFoUOGgiIy6CiOwZybnfpQ9pC21y4YkpFLke7Xm8R0tNZhoUl63_fdfb1LpehhT7bxiJjnDUGlgV3JpgbVBDsyHk_JMxW0sog_kKgCoR8zdT29fFH5HlArqqW6CnYjl6flWfEHZGlgFpIANVynSups7HIYwW9-PEuDmGqaUYR_yZGgvEGGadCeBhKnsfW9ma2U3PDd9uDYMyaqkMylrDiELhLX-ad959LXjGfe21CdbJYhWUm7ue-F0KCMGRKtF-jlrbme52Gq1XIdkkRTzVCzHH_yvTOsKWdHE5Wi8z9O1x2PN_p3vFe0hw56-sQMNxL4ZSeeNEKb-bIjFM6NA8kr4EEKhNYO6obDKUnJOH-4KO-Lu4J2YDsDD5XlTV50INPeM8ipc2fwjllPKW6h6NV119ljE8jLyTebb-zR)

**Approach:** Simple prompting technique that interleaves reasoning and action at each step.

**Key insight:** Unlike pure reasoning (think only) or pure action (act only), ReAct combines both:
* **Reason only:** Insufficient for tasks requiring external information
* **Act only:** May miss strategic considerations
* **ReAct:** Reason → Act → Observe → Reason → Act...

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
* **Planner:** Generates sequence of actions (reasoning)
* **Worker:** Executes actions and collects results (execution)
* **Solver:** Synthesizes final answer from collected information (optimization)

**Key insight:** Decoupling reasoning from observations allows more efficient planning without per-step feedback loops.

**Comparison with ReAct:** ReWOO plans upfront then executes, while ReAct interleaves planning and execution. ReWOO can be more efficient for tasks with clear structure.

#### Tree Search

**Paper:** "Tree Search for Language Model Agents" (2024)

**Approach:** Use explicit search algorithms (BFS, DFS) to explore different action sequences and choose best path.

**Key idea:** Rather than single linear path of actions, explore multiple branches:
* Consider different action options at each step
* Evaluate which branch seems most promising
* Backtrack if path leads to dead end
* Choose path with highest expected success

**Benefit:** More thorough exploration of solution space, better handling of ambiguous situations.

**Trade-off:** More expensive (multiple branches evaluated) but potentially better quality solutions.

### Workflow vs. Multi-Step Agents Comparison

![Diagram 8](https://kroki.io/mermaid/svg/eNp9kVFPwyAQx9_3KUh90ZjFSHUPi1lSEx-WGGewb8se6O3oiIwulKr79lJKtbTRe-H-_O-4H1AafjqQ_HFGXNRNUXr92hjMD1K_b5M2JQx5XWmpy4fC3KxeKvL0ZdForshaiyrZ-e428ttt4huTHZnPVySnI53GOtP1JxrX1CX--MtneZQW9yQDaAyH81UYgHo_m3JmYAOlywaIb9Zwi6WEX_oBaNbOBNtj0EilkfJkNEZcaxRCgkRt_6ZjHZtffNsPCLkOsAMiNno6FjExx7QpajQf2O_cjervo_rFuL7DT-N7bE5WHrma3sGeFfa_Q4RUankhBLiY2PR_Ow02QFswsP2zBBMXggrxDbncw1A=)

| Aspect | Workflows | Multi-Step Agents |
|--------|-----------|------------------|
| **Predictability** | High - fixed path ensures consistent outputs | Lower - dynamic planning can vary |
| **Adaptability** | Low - cannot adjust to unexpected inputs | High - adapts at runtime |
| **Implementation** | Simpler - predetermined steps | More complex - requires planning |
| **Reliability** | More reliable for well-defined tasks | Less reliable but more flexible |
| **Cost** | Lower - fewer LLM calls typically | Higher - more exploration possible |
| **Best for** | Defined tasks with clear workflows | Open-ended or unpredictable problems |

**Practical guideline:** Use workflows when solution is well understood and repeatable; use agents for open-ended or complex problems.

> **Key Insight**: Workflows vs. agents is a fundamental trade-off. Workflows offer predictability, lower cost, easier debugging, and higher reliability for well-defined tasks. Agents offer flexibility, runtime adaptation, and ability to handle novel situations at the cost of higher latency, increased costs, and less predictable outputs. Choose workflows unless adaptability is essential.

### When to Use Multi-Step Agents

**Use agents when:**
* Problem is not well-structured
* Multiple possible solution paths exist
* Unexpected inputs may require adaptation
* Complex reasoning and planning required

**Examples:**
* **Software development:** Multiple ways to solve coding problem; agent adapts based on test results
* **Research:** Open-ended exploration of topics
* **Robotics & autonomous navigation:** No predefined path; agent navigates based on observations
* **Structured information synthesis:** Deep research on topics

**Use workflows when:**
* Problem solution is well understood
* Task is repeatable with consistent structure
* Requirements are fixed and predictable

## Multi-Agent Systems

### Motivation

Single agents may:
* Fail or move in incorrect directions
* Miss important information or perspectives
* Lack specialized expertise for different subtasks

**Solution:** Multiple specialized agents coordinate and validate each other's work.

### Architecture

![Diagram 9](https://kroki.io/mermaid/svg/eNp1Ud9rwjAQft9fcWQPOlC3arUqwyETh6Aic8OHIiNNrj9YbaSJDFn3v69J203ZzEO4u---L3dfgpTuQ3iZXEF-5vOFS-Y0CQ40QFgIjvG9l96OPiIVwvooFe5glYrdXpEtNJsjmCCLZCSST7IJI5brCFEwlogc-QP5ujLCVZ8mZQuqwgweacxcou9DTJVIf7mU8_qg7_S6dgOsdsfu9m7I9q_MBnMdTDMoA5eUAYxXMyMUoHrzRYqMSlXvOC3HsQcNaFrtdsu2BvYFVU8relrNgzXSlIVGTJqwXhvPIDcnUbL2L38pEsxgGiU0dskTJphShUVuZJ5R7kUiMecasjbAWDnNHfMoe3eJNgLyvkOshoZjmXNXPldteUoqAe-8aKpVZqD8g4uinsdUXqV2Tt_VRFIdY9Sd4EdxPLz2-9zh9AT62bfEfb_D-AluViowbiM_41bDX4S9S1AxcgGyDvaY9w1QINTr)

**Manager Agent** (Orchestrator)
* Central coordinator
* Reasons about which sub-agents to activate
* Synthesizes final result from sub-agents
* Manages communication between agents

**Sub-Agents** (Specialists)
* Autonomous multi-step agents themselves
* Each handles specific domain/task
* Examples: web search agent, database query agent, analysis agent
* Each maintains own memory and planning

### Example: Anthropic's Multi-Agent Research System

**Components:**
* Lead agent: Orchestrates research process
* Search sub-agents: Perform parallel web searches
* Analysis agents: Process and analyze information
* Iterative refinement loop

**Result:** More comprehensive, accurate, and reliable research output through division of expertise.

### Challenges in Multi-Agent Systems

**Coordination Challenge**
* Different agents must share state and synchronize actions
* Deciding when to activate which agents
* Managing hand-offs between agents

**Memory Management**
* Each agent has own memory
* Useful information from one agent's memory may not be visible to others
* Central memory or communication protocol needed

**Compounding Errors**
* Error from one agent propagates to dependent agents
* In loops, errors can amplify
* Multi-agent amplification of errors worse than single-agent

**Complexity**
* Designing reliable multi-agent systems is significantly harder
* Even advanced labs approach with caution
* Requires extensive engineering and testing

## Agent-to-Agent Protocol (A2A)

### Definition

**Agent-to-Agent (A2A) Protocol** is a standardized protocol announced by Google (April 2025) enabling agents to communicate with each other, discover capabilities, and coordinate work.

### Parallel to MCP

Just as **MCP standardizes tool-agent communication**, **A2A standardizes agent-agent communication**:

* **MCP:** Standardizes how agents interact with tools
* **A2A:** Standardizes how agents interact with other agents

### Benefits

* **Interoperability:** Agents from different systems can work together
* **Discovery:** Agents can discover each other's capabilities
* **Scalability:** New agents can be added without modifying existing ones
* **Standardization:** Reduces redundant agent-to-agent integration work

### Design Principles

* Open protocol for agent discovery and communication
* Standardized message format for agent-to-agent requests
* Capability advertisement and negotiation
* Error handling and fallback mechanisms

### Broader Protocol Ecosystem

The field is developing multiple complementary protocols:
* **MCP:** Tool-to-agent communication
* **A2A:** Agent-to-agent communication
* Other emerging proposals for specific coordination patterns

(See survey in references for comprehensive protocol overview)

## Evaluation of Agents

### Key Metrics

**Token Consumption**
* Measures average tokens used per request
* Important because: More tokens = higher cost
* Goal: Maximize capability per token
* Trade-off: Planning tokens vs. solution tokens

**Tool Execution Success Rate**
* Percentage of tool calls that execute successfully
* Indicates: System setup quality and LLM reliability
* Concern: Failed tool calls can derail agent
* Improvement: Better error handling and tool descriptions

**Observability & Debuggability**
* How easily can you determine what went wrong?
* Why it matters: Complex agent behavior hard to debug
* Metrics: Can you trace decisions, identify bottlenecks, find errors?

**Task Success Rate**
* Percentage of tasks completed successfully
* Requires: Good benchmarks with correct solutions
* Domain-specific: Different metrics for different domains

### Agent Leaderboards

Leaderboards exist for evaluating agents across domains:
* Software development benchmarks
* Banking and finance tasks
* Healthcare applications
* Insurance automation
* General reasoning tasks

Leaderboards enable comparison of different agents and tracking progress in the field.

## Implementation Frameworks

In practice, building agents is simpler than the theory suggests due to mature frameworks and libraries.

### LangChain

**Features:**
* Simplifies LLM application development
* Built-in tool management with `@tool` decorator
* Easy tool binding: `llm.bind_tools(tools)`
* Tool invocation: `tool.invoke(arguments)`
* Large collection of pre-built tools (search, code execution, productivity)

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
* Purpose-built for agent implementation
* Simplified agent creation
* Good integration with OpenAI models
* Examples and tutorials for common patterns

### Google Agent SDK

**Features:**
* Agent system implementation
* Integration with Google services
* Specialized tools for Google Cloud

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

* [ ] **Define the problem:** Is this task well-structured (use workflow) or open-ended (use agent)?

* [ ] **Choose agency level:** Identify minimum autonomy needed - simple processor, workflow with tools, multi-step agent, or multi-agent system?

* [ ] **Tool planning:** List all tools/APIs needed. Check if MCP servers exist; if not, plan wrapper implementation.

* [ ] **Workflow design:** If using workflows, map out sequence of steps and design each component's responsibility.

* [ ] **Prompt engineering:** Write clear system prompt with tool descriptions, expected format, and reasoning guidelines.

* [ ] **Function calling setup:** Ensure tools have type hints, docstrings, and clear descriptions for automatic formatting.

* [ ] **Error handling:** Plan for tool failures, implement retry logic, and add validation.

* [ ] **Memory management:** Design how state/context is maintained across steps, especially for agents.

* [ ] **Testing:** Test each workflow step independently before integration; test agent loops with various inputs.

* [ ] **Evaluation setup:** Define success metrics appropriate for your use case (token consumption, tool success rate, task completion).

* [ ] **Monitoring:** Implement observability to debug agent behavior - log thoughts, actions, observations.

* [ ] **Iteration:** Start simple (workflows), add complexity only when needed; iterate based on performance metrics.

* [ ] **Framework selection:** Choose LangChain, OpenAI SDK, or Google SDK based on your stack and requirements.

* [ ] **MCP adoption:** For tools beyond single company, prefer MCP servers over manual integration.

* [ ] **Scaling plan:** Plan how system grows as new tools/agents added - ensure architecture supports scaling.

* [ ] **Cost optimization:** Track token usage, route simple queries to smaller models, cache common operations.

* [ ] **Reliability requirements:** Higher reliability needs suggest workflows; flexibility needs suggest agents; multi-agent validates results.

* [ ] **Feedback loops:** For agents, implement reflection patterns for iterative improvement based on tool feedback.

---

## References

### Papers

* **ReAct:** ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629)

* **Reflexion:** ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366)

* **ReWOO:** ["ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models"](https://arxiv.org/abs/2305.18323)

* **Tree Search for Language Model Agents:** [Paper](https://arxiv.org/abs/2407.01476)

* **AI Agents Protocol Survey:** ["A survey of AI agents protocol"](https://arxiv.org/abs/2504.16736)

### Anthropic Resources

* [**Building Effective Agents**](https://www.anthropic.com/engineering/building-effective-agents)

* [**Building Effective Agents Cookbook**](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents)

* [**Model Context Protocol (MCP)**](https://www.anthropic.com/news/model-context-protocol)

* [**Multi-Agent Research System**](https://www.anthropic.com/engineering/multi-agent-research-system)

### OpenAI Resources

* [**Introducing Operator**](https://openai.com/index/introducing-operator/)

* [**A Practical Guide to Building Agents**](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

* [**OpenAI Agents SDK**](https://openai.github.io/openai-agents-python/)

* [**Fine-tuning for Function Calling**](https://cookbook.openai.com/examples/fine_tuning_for_function_calling)

### Tool & Framework Documentation

* [**LangChain Tools**](https://python.langchain.com/docs/integrations/tools/)

* [**LangChain Function Calling**](https://python.langchain.com/docs/how_to/function_calling/)

* [**Google Agent-to-Agent (A2A) Protocol**](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)

### Agent Evaluation

* [**Agent Leaderboard**](https://github.com/rungalileo/agent-leaderboard)

### Additional Guides

* **Hugging Face Agent Course:** Agent explanations and tutorials

* **Hugging Face Agents vs Workflows:** Comparison of when to use each approach
