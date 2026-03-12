import os
from tqdm.auto import tqdm

# After running orca structure relaxations, the original INP_DIR (now SRC_DIR) will be filled with optimized XYZ geometries, which will be used to write the SCF and Freq files 
SRC_DIR = "ORCA_RELAX_INPUTS/orca_inp"
OUT_DIR = "/content/drive/MyDrive/Molecular design/PFAS/Generated/orca_sp_freq_inp"

NPROCS = 8
CHARGE = 0

METHOD_SCF  = "! wB97X-D3BJ def2-TZVP TightSCF DefGrid3"
METHOD_FREQ = "! wB97X-D3BJ def2-TZVP TightSCF DefGrid3 Freq"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Extract coords from INP
# -----------------------------
def extract_xyz_from_inp(inp_file):
    with open(inp_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    start = None
    for i, line in enumerate(lines):
        if line.startswith("* xyz"):
            start = i + 1
            break

    if start is None:
        return None

    coords = []
    for j in range(start, len(lines)):
        if lines[j].startswith("*"):
            break
        coords.append(lines[j])

    return coords


def write_orca_input(coords, out_path, method_line, mult):
    with open(out_path, "w") as f:
        f.write(f"{method_line}\n\n")
        f.write("%pal\n")
        f.write(f"  nprocs {NPROCS}\n")
        f.write("end\n\n")
        f.write(f"* xyz {CHARGE} {mult}\n")
        for line in coords:
            f.write(line + "\n")
        f.write("*\n")


inp_files = [f for f in os.listdir(SRC_DIR) if f.endswith(".inp")]

for filename in tqdm(inp_files):

    base = filename.replace(".inp", "")
    inp_path = os.path.join(SRC_DIR, filename)

    coords = extract_xyz_from_inp(inp_path)

    if coords is None:
        print(f"Skipping {base} (no coordinates found)")
        continue

    if "_parent" in base:
        mult = 1
    elif "_frag" in base:
        mult = 2
    else:
        print(f"Unknown type: {base}")
        continue

    write_orca_input(
        coords,
        os.path.join(OUT_DIR, f"{base}_sp.inp"),
        METHOD_SCF,
        mult
    )

    write_orca_input(
        coords,
        os.path.join(OUT_DIR, f"{base}_freq.inp"),
        METHOD_FREQ,
        mult
    )

print("All SCF and FREQ files created.")