# Multi-Talker ASR with Large Language Models

This repository contains code for **large language model (LLM)-based multi-talker automatic speech recognition (ASR)** with **serialized output training** on mixture speech.

The implementation is based on the following papers:

- Hao Shi, Yusuke Fujita, Tomoya Mizumoto, Lianbo Liu, Atsushi Kojima, and Yui Sudo,  
  **“Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition”**,  
  in Proc. IEEE ASRU, 2025 (Accepted). \[PDF\] \[BibTeX\]

- Hao Shi, Yuan Gao, Zhaoheng Ni, and Tatsuya Kawahara,  
  **“Serialized Speech Information Guidence with Overlapped Encoding Separation for Multi-Speaker Automatic Speech Recognition”**,  
  in Proc. IEEE SLT, 2024, pp. 198–204. \[PDF\] \[BibTeX\]

If you use this repository in your research, please consider citing these works.

---

## Overview

This project focuses on **multi-talker ASR** using LLMs as the backend decoder.  
It combines:

- **Serialized Output Training (SOT)** for multi-talker transcriptions, and  
- **Serialized CTC training** to better utilize CTC-style supervision in a serialized framework.

The current implementation targets **LibriMix** mixtures and supports **2-speaker (2mix)** and **3-speaker (3mix)** settings.

---

## Supported Models and Datasets

### LLM Backends

The following LLMs are currently supported:

- **LLaMA**: 1B, 3B, 8B  
- **LLaMA-Instruct**: 1B, 3B, 8B  

(Adjust the exact model names/paths according to your local setup.)

### Datasets

- **LibriMix**  
  - 2-speaker mixtures (**2mix**)  
  - 3-speaker mixtures (**3mix**)

You can plug in other datasets with multi-speaker mixtures by following the same serialized-output and serialized-CTC interface.

---

## Training Modes

This repository supports two main training paradigms:

1. **SOT-based training (Serialized Output Training)**  
   - Outputs all speakers’ transcriptions as a single serialized sequence.  
   - Suitable for LLM-based decoders that generate text autoregressively.

2. **Serialized CTC training**  
   - Uses **serialized CTC** objectives to align multi-talker transcriptions.  
   - Combines CTC-style supervision with serialized multi-speaker outputs.

Both training modes are designed for **multi-talker ASR** with LLM backends and can be used separately or in combination depending on your experimental setting.

---

## Getting Started

### 1. Environment
Python: 3.10.16

```bash
git clone https://github.com/<your-username>/Multi-talker-ASR-with-LLMs.git
cd Multi-talker-ASR-with-LLMs

# (optional) create and activate a virtual environment
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt

