📌 LoRA Fine-Tuning GPT-2 on Custom Medical Dataset

This project demonstrates the use of Lightweight Low-Rank Adaptation (LoRA) to fine-tune the GPT-2 language model on a custom medical dialogue dataset (dr_patient.txt). By applying LoRA to select attention layers, this approach enables parameter-efficient fine-tuning, allowing the model to learn doctor–patient conversational patterns without needing to retrain all GPT-2 parameters. The goal is to create a lightweight yet effective medical-aware conversational model capable of generating contextually relevant and coherent medical responses.

🚀 Features

Loads and processes a custom text dataset using Hugging Face Datasets

Tokenizes the data using the GPT-2 tokenizer with padding and truncation

Applies LoRA to the c_attn layers for efficient adaptation

Fine-tunes GPT-2 using the Hugging Face Trainer API

Saves the optimized LoRA-based GPT-2 model

Generates medical-style responses using the trained model

🛠️ Tech Stack

Python

Hugging Face Transformers

Datasets

PEFT (LoRA)

Accelerate

PyTorch

📂 Workflow Summary

Load the medical dialogue dataset (dr_patient.txt)

Tokenize all text samples and prepare labels for causal language modeling

Apply a LoRA configuration (r=8, alpha=64, dropout=0.05) targeting GPT-2’s c_attn module

Train the LoRA-wrapped GPT-2 model for 5 epochs using the Trainer API

Save the fine-tuned model to an output directory

Perform inference with a medical prompt such as:
“Doctor: I have headache”
to generate a contextual doctor-like response
