# User Personas

## Purpose

This document defines the primary users Vanguard is being built for.

Rather than designing features around technology, Vanguard is designed around the people who use local AI and the problems they encounter.

These personas should guide future product decisions, feature prioritization, and user experience design.

---

# Primary Users

These users represent Vanguard's primary audience and should receive the highest priority when making engineering and product decisions.

---

## Persona 1 — AI Enthusiast

### Who Are They?

AI enthusiasts enjoy experimenting with open-source language models, testing new releases, comparing performance, and exploring what local AI can do.

They often own gaming GPUs and are comfortable installing software but are not experts in AI infrastructure.

### Goals

- Experiment with the latest open-source AI models.
- Compare different models.
- Run AI locally for privacy.
- Learn and explore the local AI ecosystem.

### Pain Points

- Unsure which model fits their hardware.
- Confusing quantization options.
- Multiple runtimes with different setup processes.
- Difficulty understanding hardware requirements.

### Technical Skill

Intermediate

Comfortable with computers and software installation but not necessarily experienced with AI infrastructure.

### Why Vanguard Helps

Vanguard automatically determines the best way to run a chosen model while hiding unnecessary infrastructure complexity.

---

## Persona 2 — Computer Science Student

### Who Are They?

Students studying Computer Science, Artificial Intelligence, Machine Learning, or Software Engineering who want hands-on experience with local AI.

### Goals

- Learn AI systems.
- Experiment with models.
- Build AI-powered applications.
- Understand how modern AI infrastructure works.

### Pain Points

- Spending more time configuring tools than learning.
- Unsure which runtime to use.
- Hardware limitations.
- Fragmented documentation across many projects.

### Technical Skill

Intermediate to Advanced

Comfortable programming but still learning AI infrastructure.

### Why Vanguard Helps

Vanguard allows students to spend more time learning AI concepts instead of troubleshooting setup issues.

---

## Persona 3 — Software Developer

### Who Are They?

Developers building applications that integrate local AI models into their products or workflows.

### Goals

- Prototype quickly.
- Integrate local AI into applications.
- Achieve reliable performance.
- Spend time building products rather than configuring infrastructure.

### Pain Points

- Too many deployment options.
- Runtime selection is confusing.
- Performance tuning requires significant experimentation.
- Hardware compatibility is difficult to predict.

### Technical Skill

Advanced

Comfortable writing software but not necessarily specializing in AI infrastructure.

### Why Vanguard Helps

Vanguard automates deployment, optimization, and runtime selection while allowing developers to focus on building applications.

---

## Persona 4 — AI Researcher

### Who Are They?

Researchers, graduate students, and advanced users who regularly evaluate models, benchmarks, and inference techniques.

### Goals

- Evaluate new models quickly.
- Compare inference performance.
- Experiment with different optimization strategies.
- Maintain control over advanced settings.

### Pain Points

- Repetitive setup.
- Constant environment configuration.
- Benchmarking different runtimes takes time.
- Manual optimization for each machine.

### Technical Skill

Advanced

Highly technical users who understand AI infrastructure but still value automation for repetitive tasks.

### Why Vanguard Helps

Vanguard accelerates experimentation while preserving complete access to advanced configuration when needed.

---

# Secondary Users

These users may benefit from Vanguard but are not the project's primary focus.

- Home lab enthusiasts
- Small AI startups
- Technical content creators
- Hobbyists exploring local AI

---

# Users Vanguard Is Not Designed For

At this stage, Vanguard is not intended for:

- General consumers
- Users who only use cloud AI services such as ChatGPT or Claude
- People with no interest in running AI locally
- Enterprise-scale deployments

Future versions may expand into these areas, but they are not part of Vanguard's initial mission.

---

# Design Philosophy

Every feature added to Vanguard should improve the experience of one or more primary user personas.

If a feature does not solve a meaningful problem for these users, it should be reconsidered before implementation.
