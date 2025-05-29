# Engagement-Driven Content Generation

## 🧠 Abstract

This repository supports the paper *"Engagement-Driven Content Generation with Large Language Models"* accepted to **KDD 2025**. We present a framework for fine-tuning Large Language Models (LLMs) to generate text that maximizes user engagement within social networks. Our work introduces a reinforcement learning-based solution that uses simulated feedback, offering efficient and adaptive training.

---

## Cite us
If you use our code, please cite us:

```bibtex
@inproceedings{coppolillo2025engagement,
  title={Engagement-Driven Content Generation with Large Language Models},
  author={Coppolillo, Erica and Cinus, Federico and Minici, Marco and Bonchi, Francesco and Manco, Giuseppe},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)},
  year={2025},
  doi={10.1145/3711896.3736932},
  url={https://github.com/mminici/Engagement-Driven-Content-Generation}
}
```

---

## 🛠️ Summary of the Approach

The pipeline uses **Reinforcement Learning from Simulated Feedback (RLSF)** to train LLMs to generate content that spreads effectively in a network. The process is composed of:

1. **Query Completion**: The LLM receives a topic query to complete.
2. **Simulation**: The produced post is injected into a synthetic or real social network.
3. **Engagement Model**: The content propagates on the network.
4. **Reward Signal**: The number of activated users and the content's fluency are combined to compute a reward.
5. **Fine-tuning**: The LLM is updated via PPO using this reward.

The engagement model used is based on a modified Independent Cascade model with a Bounded Confidence filter for sentiment alignment. The whole procedure is model-agnostic, modular, and enables controlling variables like user opinion distribution, source node position, and community structure.

---

## 🚀 Usage: Starting the RL process

Launch `src/synthetic_config_pipeline.sh` to run the experiments on the **synthetic** network. You can adjust the underlying user opinion, the LLM starting position, and the modularity and homophily of the network.

Launch `src/real_config_pipeline.sh` to run the experiments on the **real-world** data. You can specify "brexit" or "referendum" for processing the Brexit and Italian Referendum datasets, respectively, "positive" and "negative" for the network opinion distribution, and "low-centrality" or "high-centrality" for the LLM position in the graph.

---

### 🔧 Requirements

Ensure you have the following packages installed:

```
conda env create -f environment_core.yml
conda activate socLLM
sh install_packages.sh
```
