from pathlib import Path

path = Path("build_gatv2_dataset.py")
text = path.read_text(encoding="utf-8")

old = "x[node_indices] = torch.tensor(x_vals.tolist(), dtype=torch.float32)"
new = "node_indices_t = torch.tensor(node_indices.tolist(), dtype=torch.long)\n    x[node_indices_t] = torch.tensor(x_vals.tolist(), dtype=torch.float32)"

if old not in text:
    raise SystemExit("Target line not found. Restore backup first or check current file.")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

print("OK: patched build_gatv2_dataset.py")
