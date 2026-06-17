"""
Strip markdown explanations from all notebooks (outputs are preserved).
- Keeps heading lines (# H1 / ## H2 / ### H3) and horizontal rules (---)
- Keeps markdown cells that are already ≤ 2 lines
- Trims longer markdown cells to heading lines only
- Drops long markdown cells with no headings
"""
import json, glob, re

NOTEBOOKS = sorted(glob.glob("*.ipynb"))


def trim_markdown(source: str) -> str | None:
    """Return trimmed markdown source, or None to drop the cell."""
    lines = source.split("\n")

    # Collect heading lines, ignoring content inside fenced code blocks
    heading_lines = []
    in_fence = False
    for l in lines:
        if l.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^#{1,6}\s", l) or l.strip() == "---":
            heading_lines.append(l)

    # Short cell with no headings: keep as-is
    if not heading_lines:
        non_empty = [l for l in lines if l.strip()]
        if len(non_empty) <= 2:
            return source.rstrip()
        # Long cell with no headings: drop it
        return None

    # Has headings: keep only headings
    result = "\n".join(heading_lines).strip()
    return result if result else None


total_cells = 0
trimmed = 0
dropped = 0

for nb_path in NOTEBOOKS:
    with open(nb_path) as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb["cells"]:
        total_cells += 1

        if cell["cell_type"] != "markdown":
            new_cells.append(cell)
            continue

        src = "".join(cell["source"])
        result = trim_markdown(src)
        if result is None:
            dropped += 1
            continue
        if result != src.rstrip():
            trimmed += 1
        cell["source"] = result
        new_cells.append(cell)

    nb["cells"] = new_cells

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"  {nb_path}")

print(f"\nDone: {total_cells} cells total")
print(f"  Markdown trimmed : {trimmed}")
print(f"  Markdown dropped : {dropped}")
