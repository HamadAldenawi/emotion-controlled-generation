"""Evaluate controllability (does the text express the requested emotion?) and diversity.
Judge = TF-IDF + logistic regression trained on the dair-ai/emotion train split, scored on its test split first."""
import json, torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate import generate
from train import LABELS

def distinct_n(texts, n):
    grams = [tuple(t.split()[i:i+n]) for t in texts for i in range(len(t.split()) - n + 1)]
    return len(set(grams)) / max(1, len(grams))

def main(model_dir="model", n=20):
    tr, te = load_dataset("dair-ai/emotion", split="train"), load_dataset("dair-ai/emotion", split="test")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2); X = vec.fit_transform(tr["text"])
    clf = LogisticRegression(max_iter=2000).fit(X, tr["label"])
    judge_acc = (clf.predict(vec.transform(te["text"])) == te["label"]).mean()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_dir); m = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    res, all_txt, rejected = {}, [], 0
    for i, lab in enumerate(LABELS):
        txt, tries = generate(m, tok, lab, n); rejected += tries - len(txt)
        res[lab] = round(float((clf.predict(vec.transform(txt)) == i).mean()), 3); all_txt += txt
    report = {"judge_accuracy_on_test": round(float(judge_acc), 3), "control_accuracy_per_label": res,
              "control_accuracy_mean": round(sum(res.values()) / len(res), 3),
              "distinct_1": round(distinct_n(all_txt, 1), 3), "distinct_2": round(distinct_n(all_txt, 2), 3),
              "generations_rejected_by_safety_filter": rejected, "samples_per_label": n}
    json.dump(report, open("results.json", "w"), indent=2); print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
