# Evaluation of Multi-Agent Systems

**Codelab**: [From "vibe checks" to data-driven Agent Evaluation](https://codelabs.developers.google.com/codelabs/production-ready-ai-roadshow/2-evaluating-multi-agent-systems/evaluating-multi-agent-systems).

**This is the starter version of the lab. The final version of the repository is available in the [`main` branch](https://github.com/vladkol/agent-evaluation-lab/tree/main).**

This project implements a **Continuous Evaluation Pipeline** for a multi-agent system built with Google Agent Development Kit (ADK) and Agent-to-Agent (A2A) protocol on [Cloud Run](https://docs.cloud.google.com/run/docs?utm_campaign=CDR_0xc245fc42_default_b473562939&utm_medium=external&utm_source=blog). It features a team of microservice agents that research, judge, and build content, orchestrated to deliver high-quality results.

The goal of this project is to demonstrate **Agentic Engineering** practices: safely deploying agents to shadow revisions, running automated evaluation suites using Vertex AI, and making data-driven decisions on agent improvements.

[Vertex AI Gen AI Evaluation Service](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview?utm_campaign=CDR_0xc245fc42_default_b473562939&utm_medium=external&utm_source=blog) provides enterprise-grade tools for objective, data-driven assessment of generative AI models and agents.
