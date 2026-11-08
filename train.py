"""Fine-tune a small causal LM (distilgpt2) to generate short texts conditioned on an emotion label.
Each training example is formatted as: "joy: text<|endoftext|>" (prefix conditioning)."""
import argparse, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--n_per_label", type=int, default=400)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--out", default="model")
    a = p.parse_args()
    random.seed(0); torch.manual_seed(0)
    ds = load_dataset("dair-ai/emotion", split="train")
    by = {i: [] for i in range(6)}
    for ex in ds: by[ex["label"]].append(ex["text"])
    rows = [f"{LABELS[l]}: {t}" for l in by for t in random.sample(by[l], min(a.n_per_label, len(by[l])))]
    random.shuffle(rows)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token  # control = plain-text prefix "joy: ..." (learns faster than new tokens)
    model = AutoModelForCausalLM.from_pretrained(a.model)
    enc = tok([r + tok.eos_token for r in rows], truncation=True, max_length=48)
    data = [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])]
    args = TrainingArguments(a.out, num_train_epochs=a.epochs, per_device_train_batch_size=16, learning_rate=5e-4,
                             logging_steps=25, save_strategy="no", report_to=[], use_cpu=not torch.cuda.is_available())
    Trainer(model=model, args=args, train_dataset=data,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)

if __name__ == "__main__":
    main()
