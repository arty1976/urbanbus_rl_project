from pathlib import Path

path = Path("build_gatv2_dataset.py")
text = path.read_text(encoding="utf-8")

old = "y[node_indices] = torch.from_numpy(y_vals)"
new = "y[node_indices_t] = torch.tensor(y_vals.tolist(), dtype=torch.float32)"

if old not in text:
    raise SystemExit("Target y-line not found.")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

print("OK: patched y assignment")
