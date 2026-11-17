import json

with open("notebooks/AIML506_GraphRAG_vs_VectorRAG.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip() if c['source'] else "(empty)"
    print(f"Cell {i:02d} [{c['cell_type']:8s}]: {first_line[:80]}")
