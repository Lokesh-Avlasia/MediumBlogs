# The Handoff Pattern: Teaching AI Agents to "Transfer the Call"

*A practical guide to building multi-agent systems — no framework needed*

---

If you've ever called customer support and heard "Let me transfer you to a specialist," you already understand the handoff pattern. In this tutorial, we'll build exactly that — but with AI agents.

By the end, you'll understand:
- Why a single AI agent fails at multi-domain tasks
- What the handoff pattern is and when you need it
- How to build it from scratch with Python and the OpenAI API

No frameworks. No magic. Just clean Python so you learn the **pattern**, not a library.

---

## Section 1: The Single Agent That Tries Too Hard

Let's start with what most people build first — one agent that handles everything.

Here's a customer support bot that does billing AND tech support:

```python
SYSTEM_PROMPT = """You are a customer support agent. You handle ALL types of issues:

BILLING ISSUES:
- Help with invoices, charges, refunds, payments
- Always ask for account email and invoice number
- Refunds take 5-7 business days
- Pricing: Basic $29.99/mo, Pro $59.99/mo, Enterprise $99.99/mo

TECHNICAL ISSUES:
- Help with login problems, app crashes, bugs, errors
- Always ask what device, browser, and app version
- Common fixes: clear cache, update app, try incognito mode
- Known bug: v3.2.1 crashes on iOS, fixed in v3.2.2
"""
```

Now try this conversation:

```
You: I was charged twice for my subscription last month

Agent: I'm sorry about the double charge! Could you please provide
       your account email and the approximate date? Also, what
       device are you using to access your account?
```

See the problem? It's asking for your **device** when you have a **billing** issue. The billing and tech support instructions are bleeding into each other.

This gets worse as you add more domains. A single agent with 20 responsibilities and 15 tools will start guessing which tool to use and mixing up its instructions. It's like hiring one person to be your accountant, your IT help desk, and your receptionist — simultaneously.

**The core problem:** One system prompt cannot carry deep expertise in multiple domains without losing focus.

---

## Section 2: What If Agents Could Transfer the Call?

Think about how a real company works:

1. You call and reach a **receptionist** (triage)
2. The receptionist asks "What's this about?"
3. They **transfer you** to the billing department or tech support
4. That specialist helps you — they know their domain deeply
5. If you bring up a different topic, they transfer you back

This is the **handoff pattern**:

> One agent transfers control of the conversation to another agent that is better suited for the task.

Here are the key concepts:

- **Agent**: An LLM call with a specific system prompt and set of tools
- **Active Agent**: Which agent is currently "on the phone" with the user — only one at a time
- **Handoff**: The active agent decides to transfer control to another agent
- **Conversation History**: The shared memory that travels with the user across handoffs — so they never repeat themselves

**Important:** A handoff is NOT a tool call that returns data. It's a **transfer of control**. The old agent goes to sleep, the new agent wakes up, and the new agent talks to the user directly.

---

## Section 3: Building the Agents

Each agent is just a dictionary with three things: a name, a system prompt, and the tools it can call.

### Agent 1: Triage

```python
triage_agent = {
    "name": "Triage Agent",
    "system_prompt": (
        "You are a customer support triage agent. "
        "Your ONLY job is to understand the customer's issue "
        "and route them to the right specialist. "
        "You do NOT solve problems yourself.\n\n"
        "Hand off to:\n"
        "- Billing specialist: for invoices, charges, refunds\n"
        "- Tech support: for login issues, app crashes, bugs\n"
    ),
    "tools": [handoff_to_billing_tool, handoff_to_tech_support_tool],
}
```

Notice: **no billing knowledge, no tech knowledge**. Triage only knows how to listen and route.

### Agent 2: Billing Specialist

```python
billing_agent = {
    "name": "Billing Agent",
    "system_prompt": (
        "You are a billing specialist. "
        "You handle invoices, charges, refunds, payments.\n\n"
        "Ask specific follow-up questions:\n"
        "- Account email address\n"
        "- Invoice number or approximate date of charge\n"
        "- Payment method used\n\n"
        "If the customer asks about something OUTSIDE billing, "
        "hand them back to triage. Do NOT help with tech issues."
    ),
    "tools": [handoff_to_triage_tool],
}
```

Notice: **deep billing knowledge, zero tech knowledge**. And it can only hand off **back to triage** — it doesn't know other agents exist. This keeps agents loosely coupled.

### Agent 3: Tech Support

```python
tech_support_agent = {
    "name": "Tech Support Agent",
    "system_prompt": (
        "You are a technical support specialist. "
        "You handle login issues, app crashes, bugs, errors.\n\n"
        "Ask diagnostic questions:\n"
        "- What device (iPhone, Android, desktop)?\n"
        "- What browser or app version?\n"
        "- When did the issue start?\n\n"
        "If the customer asks about something OUTSIDE tech support, "
        "hand them back to triage. Do NOT help with billing."
    ),
    "tools": [handoff_to_triage_tool],
}
```

**Compare this to the single agent approach.** Each system prompt is small, focused, and carries only the knowledge it needs. The billing agent will never accidentally ask about your device. The tech agent will never mention refund policies.

---

## Section 4: The Handoff Mechanism

Now for the interesting part — how does the transfer actually work?

### The Handoff Tools

The handoff "tools" are not real tools that fetch data or call APIs. They're **signals**. When an agent returns a `handoff_to_billing` tool call, it's saying: "I want to transfer this conversation."

```python
handoff_to_billing_tool = {
    "type": "function",
    "function": {
        "name": "handoff_to_billing",
        "description": "Transfer the customer to the billing specialist.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief summary of why you are transferring",
                }
            },
            "required": ["reason"],
        },
    },
}
```

### Calling an Agent

This is the core function. Pay attention to one detail:

```python
def call_agent(agent, conversation_history):
    messages = [
        {"role": "system", "content": agent["system_prompt"]},  # ← THIS changes
        *conversation_history,                                    # ← THIS stays the same
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=agent["tools"],
    )

    return response.choices[0].message
```

**This is the entire handoff mechanism.** When you switch agents, you don't start a new conversation. You keep the same history but swap the system prompt. The new agent "sees" everything that was discussed but through its own specialist lens.

### The Chat Loop

The main loop has an **inner loop** that handles handoffs:

```python
def run_chat_loop():
    active_agent = triage_agent
    conversation_history = []

    while True:
        user_input = input("You: ")
        conversation_history.append({"role": "user", "content": user_input})

        # Inner loop: keep going until an agent gives a TEXT response
        while True:
            response = call_agent(active_agent, conversation_history)

            if response.tool_calls:
                # Agent wants to hand off — switch agents
                tool_call = response.tool_calls[0]
                target_key, reason = process_tool_call(tool_call)

                # Add the tool call and result to history (required by API)
                conversation_history.append(...)  # tool call message
                conversation_history.append(...)  # tool result message

                # Switch!
                active_agent = agents[target_key]
                continue  # ← Don't wait for user, let new agent speak

            # Agent gave text — show it and wait for next user message
            print(f"{active_agent['name']}: {response.content}")
            conversation_history.append({"role": "assistant", "content": response.content})
            break
```

**The critical detail:** After a handoff, we do NOT wait for the user to type again. We `continue` the inner loop, which immediately calls the new agent. This is what lets the billing agent say "Hi, I'm the billing specialist. I can see you had a question about a charge..." without the user having to repeat themselves.

---

## Section 5: Watch It Work

Here's a real conversation with the system. I've added `[debug]` annotations showing what happens behind the scenes.

```
============================================================
Customer Support Bot (Handoff Pattern)
============================================================
[Connected to: Triage Agent]

You: Hi, I was charged twice for my subscription last month

  [debug] Triage Agent receives the message
  [debug] Tool call: handoff_to_billing(reason="duplicate charge on subscription")
  [Handoff: Triage Agent → Billing Agent]

Billing Agent: Hi there! I'm sorry to hear about the duplicate
charge. I'd like to look into this for you. Could you please
provide:
1. The email address associated with your account?
2. The approximate date of the charges?
```

The billing agent asked for **email and date** — billing-specific questions. Not your device or browser.

```
You: Sure, it's jane@example.com and the charges were around Feb 15

Billing Agent: Thank you, Jane. I can see two charges of $29.99
on February 15th. This appears to be a system error during
payment processing. I'll initiate a refund for the duplicate
charge — you should see it back on your card within 5-7
business days. Is there anything else I can help with?
```

Now watch what happens when the topic changes:

```
You: Actually yes — my app has been crashing on my phone lately

  [debug] Billing Agent receives the message
  [debug] Tool call: handoff_to_triage(reason="customer reporting app crash, not a billing issue")
  [Handoff: Billing Agent → Triage Agent]

  [debug] Triage Agent receives the message
  [debug] Tool call: handoff_to_tech_support(reason="app crashing on mobile device")
  [Handoff: Triage Agent → Tech Support Agent]

Tech Support Agent: Hey! I can see you've been experiencing app
crashes on your phone. Let me help troubleshoot that. A few
questions:
1. What type of phone are you using (iPhone or Android)?
2. What version of our app do you have installed?
3. When did the crashing start?
```

Look at the difference. Tech support asked for **phone type, app version, and timeline** — completely different questions than billing asked. Same system, same conversation, but a different specialist is now in charge.

```
You: iPhone 14, latest version, started about a week ago

Tech Support Agent: That timing matches a known bug in version
3.2.1 that affected iOS devices. We released a fix in version
3.2.2 yesterday. Could you try updating the app from the App
Store? If the crash persists after updating, try clearing the
app cache: Settings → Apps → [Our App] → Clear Cache.
```

Two handoffs happened in the background between agents, but the user experienced one seamless conversation.

---

## Section 6: Why This Matters Beyond the Demo

### You Don't Need a Framework for This

What we built is ~200 lines of Python. The handoff pattern is not a framework feature — it's an **architectural idea**: same conversation history, different system prompt, and a mechanism to switch.

Frameworks like OpenAI Swarm and LangGraph implement this same pattern with nicer abstractions. But the core idea is what you just saw.

### When You Actually Need Handoff

You need handoff when **all three** of these are true:

1. **Domain expertise matters for the conversation itself** — the specialist asks smarter questions than a generalist could relay
2. **The conversation goes deep** — multiple turns of back-and-forth, not just a single Q&A
3. **The flow is unpredictable** — each user answer changes what the agent asks next

If any of these are false, a simpler pattern (single agent with tools, or a supervisor orchestrating sub-agents) works fine.

### Scaling Up

The beauty of this pattern:

- **Add a new agent** (e.g., Sales) without touching existing agents — just add a new tool to triage
- **Test independently** — each agent can be tested in isolation with mock conversations
- **Use different models** — GPT-4o for complex billing, GPT-4o-mini for simple triage
- **Change one agent's behavior** without risking regressions in others

### Try It Yourself

1. **Add a Sales agent** that handles pricing questions and upsells
2. **Add memory** — let agents store notes that persist across handoffs
3. **Use a cheaper model** for triage (it only routes, doesn't need to be smart)
4. **Add guardrails** — what if the billing agent refuses to hand back? Add a max-turn check

---

## The Full Code

The complete working implementation is in `handoff_tutorial.py` — a single file you can run with:

```bash
pip install openai
export OPENAI_API_KEY="your-key-here"
python handoff_tutorial.py
```

And the "before" example showing the single-agent problem is in `single_agent_problem.py`.

---

## When Should You Use the Handoff Pattern? (The Decision Guide)

Before you add handoff to your system, ask yourself these questions. If you can't say "yes" to the handoff side — you don't need it.

### Handoff vs. Single Agent With Tools

Ask yourself these three questions. If the answer is "yes" to any of them — handoff might be your fix.

**"Is my system prompt getting too bloated to maintain?"**
YES → Handoff. Split domains into separate agents.
NO → Single agent is fine.

**"Are instructions from different domains bleeding into each other?"**
YES → Handoff. Isolated system prompts solve this.
NO → Single agent is fine.

**"Does the agent pick the wrong tool because it has too many?"**
YES → Handoff. Each agent only sees its own tools.
NO → Single agent is fine.

**Bottom line:** If your single agent works well with its tools — don't add handoff. It's over-engineering. Handoff solves the problem of **one agent trying to be an expert in everything**.

### Handoff vs. Supervisor + Agent-as-Tool

This is the harder decision. A smart supervisor CAN relay messages between the user and a sub-agent. So when is handoff actually better?

**"Does the sub-agent need to ask domain-specific follow-ups that the supervisor can't rephrase well?"**
YES → Handoff. The billing agent knows to ask "gross or net revenue?" — a supervisor would just parrot "they need more info."
NO → Supervisor + Tool. The supervisor relays the question just fine.

**"Is the sub-agent conversation long (5+ turns) and would it clutter the supervisor's context?"**
YES → Handoff. 15 turns of billing back-and-forth wastes the supervisor's context window.
NO → Supervisor + Tool. 1-2 turns, supervisor handles it easily.

**"Do you care about cost and latency?"**
YES → Handoff. Every turn through a supervisor = extra LLM call. Handoff removes the middleman.
NO → Supervisor + Tool. Cost doesn't matter for your use case.

**Bottom line:** If the supervisor can relay the conversation just as well as the sub-agent would handle it directly — use supervisor + tool. It's simpler. Switch to handoff only when you **feel the pain**.

### The 3 Reasons That Actually Matter

After all the theory, it comes down to exactly three reasons to use handoff:

**1. The sub-agent talks to the user BETTER than a supervisor can relay**

The specialist asks smarter, more contextual follow-up questions. A billing agent that sees your payment data can ask "I see your gross revenue is up but net is down because chargebacks doubled — which one concerns you?" A supervisor relaying messages can never do this.

**2. You want to save cost and reduce latency**

With a supervisor as middleman: `User → Supervisor (LLM call) → Sub-agent (LLM call) → Supervisor (LLM call) → User` = 3 LLM calls per turn.

With handoff: `User → Sub-agent (LLM call) → User` = 1 LLM call per turn.

Over a 10-turn conversation, that's **20 extra LLM calls saved**.

**3. Long sub-conversations would pollute the supervisor's memory**

If billing needs 15 turns with the user, all that back-and-forth sits in the supervisor's context. When the user later asks about something else, the supervisor is dragging around 15 turns of irrelevant billing detail — wasting tokens and confusing the model.

**If none of these three reasons apply to you — you don't need handoff. Use a simpler pattern.**

### Real-World Problems That Need Handoff

These are the startup use cases where handoff genuinely shines and no other pattern does it as well:

**Customer support with deep troubleshooting**
Tech support needs to diagnose step-by-step: "try this" → "didn't work" → "try that" → each answer changes the next question. A supervisor can't drive this diagnostic conversation.

**Multi-stage sales** (qualify → demo → close)
Each stage needs a different personality and expertise. Sales is warm, technical demo is precise, closing is persuasive. One supervisor can't play all three roles.

**Medical / Legal / Financial intake**
A tax agent asks follow-ups based on previous answers ("LLC? Which state? Payroll or income tax?"). Getting the sequence wrong has real consequences. The specialist MUST drive the conversation.

**Multi-department internal copilot**
HR policies, compensation rules, IT help — each domain has complex rules that change based on context. A 15-turn HR conversation about FMLA leave shouldn't clutter the main agent.

### The Quick Decision Flowchart

```
Is your single agent struggling with too many domains?
    │
    NO → Keep your single agent. You're fine.
    │
    YES → Can a supervisor relay the conversation just as well?
              │
              YES → Use supervisor + agent-as-tool. Simpler.
              │
              NO → Does the sub-agent need to:
                    - Ask smart, domain-specific follow-ups? → HANDOFF
                    - Have long (5+ turn) conversations?     → HANDOFF
                    - Save you cost on LLM calls?            → HANDOFF
```

**Start simple. Add complexity only when you feel the pain.**

---

*The handoff pattern is just one of several multi-agent patterns. In future posts, we'll explore the orchestrator pattern (a supervisor that delegates sub-tasks) and the pipeline pattern (agents chained in sequence). Each has its place — but handoff is where most teams should start when their single agent outgrows its system prompt.*
