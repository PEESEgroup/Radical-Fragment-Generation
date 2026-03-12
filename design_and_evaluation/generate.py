import pandas as pd
import torch
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

    input_ids = torch.tensor(
        [[tokenizer.bos_token_id]],
        device=device
    )

    attention_mask = torch.ones_like(input_ids)

    bde_tensor = torch.tensor(
        [bde_value],
        dtype=torch.float32,
        device=device
    )

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

BDE_VALUES = list(range(55, 126, 10))
# → [55, 65, 75, 85, 95, 105, 115, 125]

from tqdm import tqdm
import pandas as pd

N_SAMPLES = 1000

gen_f1 = []
gen_f2 = []
gen_bde = []

for bde in BDE_VALUES:
    for _ in tqdm(range(N_SAMPLES), desc=f"BDE {bde}"):
        try:
            out = generate(
                model=model,
                tokenizer=tokenizer,
                bde_value=float(bde),
                temperature=1.2,
                top_p=0.99,
            )

            f1, f2 = out.split("  ")

            gen_f1.append(f1)
            gen_f2.append(f2)
            gen_bde.append(float(bde))

        except Exception:
            continue

saved_df = pd.DataFrame({
    "fragment1": gen_f1,
    "fragment2": gen_f2,
    "bde": gen_bde
})

saved_df.to_csv("SAVED_GENERATED_PAIRS/generated_radical_pairs.csv", index=False)