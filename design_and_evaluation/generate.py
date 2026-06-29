import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from models.generator import BDEConditionedGPT
from transformers import AutoTokenizer, GPT2Config

SAVE_DIR = "generator/bde_conditioned_gpt"

# load saved model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = BDEConditionedGPT.from_pretrained(SAVE_DIR).to(device)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    bde_value,
    max_new_tokens=80,
    temperature=1.2,
    top_p=0.99,
    top_k=50,
    num_return_sequences=1,
):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    attention_mask = torch.ones_like(input_ids)
    bde_tensor = torch.tensor([bde_value], dtype=torch.float32, device=device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bde=bde_tensor,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts[0]


def run(bde_values, n_samples, out_path, temperature=1.2, top_p=0.99):
    """Generate `n_samples` fragment pairs at each target in `bde_values` and save to `out_path`."""
    gen_f1, gen_f2, gen_bde = [], [], []
    for bde in bde_values:
        for _ in tqdm(range(n_samples), desc=f"BDE {bde}"):
            try:
                out = generate(
                    model=model,
                    tokenizer=tokenizer,
                    bde_value=float(bde),
                    temperature=temperature,
                    top_p=top_p,
                )
                f1, f2 = out.split("  ")
                gen_f1.append(f1)
                gen_f2.append(f2)
                gen_bde.append(float(bde))
            except Exception:
                continue

    saved_df = pd.DataFrame({"fragment1": gen_f1, "fragment2": gen_f2, "bde": gen_bde})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    saved_df.to_csv(out_path, index=False)
    return saved_df


# OPTION A — run interactively / in a notebook:
#   edit the values below and call run(...) directly.

BDE_VALUES = list(range(55, 126, 10))   # target BDE values (kcal/mol)
N_SAMPLES  = 1000                       # number of pairs per target
OUT_PATH   = "SAVED_GENERATED_PAIRS/generated_radical_pairs.csv"
TEMPERATURE = 1.2
TOP_P       = 0.99

# Uncomment to run directly (e.g. in a notebook or by executing this block):
# saved_df = run(BDE_VALUES, N_SAMPLES, OUT_PATH, TEMPERATURE, TOP_P)


# OPTION B — run from the command line:
#   python generate.py --bde 90 --n 50 --out my_pairs.csv
# Defaults reproduce the manuscript settings.

def _in_notebook():
    return "ipykernel" in sys.modules

if __name__ == "__main__" and not _in_notebook():
    parser = argparse.ArgumentParser(
        description="Generate radical fragment pairs at user-specified target BDE values."
    )
    parser.add_argument("--bde", type=float, nargs="+", default=BDE_VALUES,
                        help="Target BDE value(s) in kcal/mol (default: 55 65 ... 125).")
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help="Number of fragment pairs per target (default: 1000).")
    parser.add_argument("--out", default=OUT_PATH, help="Output CSV path.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    args = parser.parse_args()
    run(args.bde, args.n, args.out, args.temperature, args.top_p)
