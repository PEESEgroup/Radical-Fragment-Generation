import pandas as pd
from design_and_evaluation.combine_radicals import combine_fragments

bde2_v3 = pd.read_csv("data/bde_rdf_with_multi_halo_cfc_model_3.csv")
train_df = bde2_v3[bde2_v3["set"] == "train"]

NUM_TO_SAMPLE_PER_TARGET_BDE = 20

# Need to have run "design_and_evaluation/screener_prediction.py" first to create the "generated_radical_pairs_with_screener_predictions.csv" file.
preds_frags = pd.read_csv("SAVED_GENERATED_PAIRS/generated_radical_pairs_with_screener_predictions.csv")

def canon_pair(a, b):
    return tuple(sorted([a, b]))

train_pairs = set(
    canon_pair(r.fragment1, r.fragment2)
    for _, r in train_df.iterrows()
)

print(f"Training unique fragment pairs: {len(train_pairs)}")


gen = preds_frags.copy()

gen["pair_key"] = gen.apply(
    lambda r: canon_pair(r.fragment1, r.fragment2),
    axis=1,
)

gen["is_novel"] = ~gen["pair_key"].isin(train_pairs)


print("Novel fraction:", gen["is_novel"].mean())


TARGETS = sorted(gen["bde"].unique())

novel_examples = []

for t in TARGETS:
    sub = gen[(gen["bde"] == t) & (gen["is_novel"])]

    if len(sub) == 0:
        continue

    # sample up to 20 clean examples
    pick = sub.sample(min(NUM_TO_SAMPLE_PER_TARGET_BDE, len(sub)), random_state=42)

    novel_examples.append(
        pick[["bde", "fragment1", "fragment2", "pred_bde", "pair_key"]]
    )

novel_examples = pd.concat(novel_examples).reset_index(drop=True)

tqdm.pandas(desc="Recombining fragments")

novel_examples['parent'] = novel_examples.progress_apply(
    lambda row: combine_fragments(row.fragment1, row.fragment2),
    axis=1
)

novel_examples.to_csv("SAVED_GENERATED_PAIRS/novel_examples_for_dft.csv", index=False)