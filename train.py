import os
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="C:/Users/J Bernardo/Desktop/tp_ia/phi-2")
    parser.add_argument("--dataset_path", type=str, default="C:/Users/J Bernardo/Desktop/tp_ia/data/train.json")
    parser.add_argument("--output_dir", type=str, default="C:/Users/J Bernardo/Desktop/tp_ia/phi-2-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    return parser.parse_args()

def clean_text(text):
    text = text.replace('\t', ' ')
    text = ' '.join(text.split())
    return text

def format_example(example):
    messages = example.get("messages", [])
    prompt = ""
    for m in messages:
        role = m.get("role", "")
        content = clean_text(m.get("content", ""))
        tag = "<|user|>" if role == "user" else "<|assistant|>"
        prompt += f"{tag}\n{content}\n"
    return {"text": prompt.strip()}

def main():
    args = parse_args()
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.train()

    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    formatted_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    formatted_dataset = formatted_dataset.filter(lambda x: x["text"] and len(x["text"].strip()) > 10)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_attention_mask=True
        )

    tokenized_dataset = formatted_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if len(tokenized_dataset) < 10:
        raise ValueError("El dataset es muy pequeño o fue filtrado completamente. Verifica el contenido.")

    split = tokenized_dataset.train_test_split(test_size=min(0.1, 1000/len(tokenized_dataset)))
    train_dataset = split["train"]
    eval_dataset = split["test"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        save_steps=100,
        save_total_limit=2,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=100,
    )

    def custom_collator(features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collator
    )

    # Test corto antes de entrenar
    sample = tokenizer("¿Qué son los números naturales?", return_tensors="pt", padding=True, truncation=True)
    print("input_ids:", sample["input_ids"])
    print("decoded:", tokenizer.decode(sample["input_ids"][0]))
    print("vocab size:", tokenizer.vocab_size)
    print("max token id:", sample["input_ids"].max().item())

    sample = train_dataset[0]
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0),
        "attention_mask": sample["attention_mask"].unsqueeze(0),
        "labels": sample["input_ids"].unsqueeze(0)
    }
    out = model(**inputs)
    print("Loss inicial:", out.loss.item())

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
