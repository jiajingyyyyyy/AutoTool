# AutoTool: Efficient Tool Selection for Large Language Model Agents

[![arXiv](https://img.shields.io/badge/arXiv-2511.14650-b31b1b.svg)](https://arxiv.org/abs/2511.14650)
[![Status](https://img.shields.io/badge/Status-Under_Code_Review-yellow)](https://github.com/your-username/AutoTool)

## Overview

**AutoTool** is a graph-based and lightweight tool selection framework designed to optimize Large Language Model (LLM) agents.

Instead of passively following observed tool usage inertia, AutoTool manages it as an active behavior. By integrating statistical structures into the agent design, AutoTool leverages usage inertia to effectively address high latency and resource consumption common in existing multi-step tool selection frameworks.

![AutoTool Workflow](./assets/workflow.png)

---

## 🚀 Installation

### 1. Setup Conda Environment

First, create the environment and install the core `autool` package.

```bash
conda create --name autotool python=3.10
conda activate autotool
pip install -e .
```

### 2. Environment Preparation (AgentBoard)

AutoTool relies on the [AgentBoard](https://github.com/hkust-nlp/AgentBoard) environment (specifically for Alfworld, ScienceWorld, and ToolQuery-Academic). We highly recommend using **Docker**.

<details>
<summary><strong>Click to expand Docker setup instructions (~12GB)</strong></summary>


**Step 1: Pull and Run Docker**
Replace `/your/local/path` with your actual workspace paths.

```bash
docker pull zzh202121/agentboard:0117
docker run -itd \
    --network host \
    --name autotool \
    --shm-size 64gb \
    -v /path/to/model_download:/model_download \
    -v /path/to/AutoTool_Repo:/data \
    zzh202121/agentboard:0117 \
    /bin/bash

docker attach autotool
```

**Step 2: Activate Internal Environment**
Inside the container:

```bash
echo 'export PROJECT_PATH=/data' >> ~/.bashrc
source ~/.bashrc
conda activate agentboard
```

**Step 3: Download Data**

```bash
git clone https://github.com/hkust-nlp/AgentBoard.git
cd AgentBoard
mkdir data
wget https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz
tar -zxvf data.tar.gz
```

</details>

### 3. Download Embedding Model

AutoTool uses SimCSE for embeddings. Download the model to your specified directory:

```bash
cd /data/model_download
huggingface-cli download princeton-nlp/sup-simcse-roberta-base \
    --local-dir models--princeton-nlp--sup-simcse-roberta-base/snapshots/4bf73c6b5df517f74188c5e9ec159b2208c89c08 \
    --local-dir-use-symlinks False
```

## ⚙️ Configuration

### API and Paths

Copy the example configuration and update it with your credentials:

```bash
cd autool
cp .env.example .env
```

**Edit `.env` file:**

*   Configure API Base URL.
*   Configure API Key.
*   Set the Embedding Model path.
*   Set the Tool Description file path.

## 📂 Project Structure & Core Components

The project is organized into two main parts: the **AutoTool** core package and the **AgentBoard** evaluation framework.

```text
.
├── autool/                         # 📦 Core AutoTool Framework
│   ├── core/
│   │   ├── tool_predict/           # Graph-based tool selection logic
│   │   ├── param_completion/       # Parameter dependency handling & completion
│   └── utils/                      # Utilities for embeddings (SimCSE) and parsing
│
├── agentboard/                     # 🧪 Evaluation Platform (Modified)
│   ├── agents/                     # Agent Implementations (Entry Points)
│   │   ├── test_agent.py           # 👉 AutoTool + ReAct (Main Implementation)
│   │   ├── test_agent2.py          # 👉 Ngram + ReAct (Baseline)
│   ├── prompts/                    # System prompts for Alfworld, ScienceWorld, etc.
│   └── eval_main.py                # 🚀 Main evaluation script
│
└── eval_configs/
    └── main_results_all_tasks.yaml # Global configuration for evaluation
```

### Core Components & Hyperparameters

Key agent implementations are located in `agentboard/agents`:

*   `test_agent.py`: **AutoTool + ReAct**
*   `test_agent2.py`: **AutoTool + Reflexion**

> **Note:** You can adjust hyperparameters at the end of each python file.

To select the agent type for evaluation, modify `eval_configs/main_results_all_tasks.yaml`.

## 🏃 Quick Start

Run the evaluation script (e.g., for AlfWorld):

```bash
python agentboard/eval_main.py \
    --cfg-path eval_configs/main_results_all_tasks.yaml \
    --tasks alfworld \
    --model DeepSeekV3 \
    --log_path ./results/alfworld_quick_start \
    --project_name evaluate_reflection \
    --baseline_dir ./data/baseline_results
```

**Important Note regarding `--model`:**
Although we use API-based inference (not local models), the AgentBoard framework requires the `--model` argument to be present. You can specify any string (e.g., `qwen`), but do not omit the flag.

## 📖 Citation

If you find AutoTool useful for your research, please cite our paper:

```bibtex
@article{jia2025autotool,
  title={AutoTool: Efficient Tool Selection for Large Language Model Agents},
  author={Jia, Jingyi and Li, Qinbin},
  journal={arXiv preprint arXiv:2511.14650},
  year={2025}
}
```

## ✉️ Contact

For questions or feedback, feel free to contact: **jingyijia@hust.edu.cn**
