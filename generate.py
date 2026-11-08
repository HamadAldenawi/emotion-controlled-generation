import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safety import is_safe

def generate(model, tok, label, n=5, max_tries=4):
    out, tries = [], 0
    ids = tok(f"{label}:", return_tensors="pt").input_ids
    while len(out) < n and tries < n * max_tries:
        tries += 1
        g = model.generate(ids, do_sample=True, top_p=0.92, temperature=0.9, max_new_tokens=30,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
        if text and is_safe(text): out.append(text)
    return out, tries

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="model"); p.add_argument("--label", default="joy"); p.add_argument("--n", type=int, default=5)
    a = p.parse_args(); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForCausalLM.from_pretrained(a.model).eval()
    for t in generate(m, tok, a.label, a.n)[0]: print("-", t)
