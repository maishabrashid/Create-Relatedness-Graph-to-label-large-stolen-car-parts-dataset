import pandas as pd
import networkx as nx
import ast
import re
import json
from typing import Optional, Tuple, Dict, Set, List
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def safe_eval(val):
    """Parse NER or JSON-like columns safely."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, dict):
            return list(parsed.keys())
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()]
        return []
    except Exception:
        return [str(val).strip()]

_WORD2DIG = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "for": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_WORD_RE = re.compile("|".join(sorted(_WORD2DIG, key=len, reverse=True)))

def words_to_digits(s: str) -> str:
    """Convert spelled-out or glued words into digits."""
    if not s:
        return ""
    t = s.lower()

    def _repeat_repl(m):
        mult = 2 if m.group(1) == "double" else 3
        d = _WORD2DIG.get(m.group(2), "")
        return d * mult

    t = re.sub(
        r"\b(double|triple)\s+(zero|oh|o|one|two|three|four|for|five|six|seven|eight|nine)\b",
        _repeat_repl,
        t,
    )
    t = _WORD_RE.sub(lambda m: _WORD2DIG[m.group(0)], t)
    return re.sub(r"\D", "", t)

def normalize_phone(s: str) -> Optional[str]:
    """Normalize any phone string to only digits."""
    digits = words_to_digits(s)
    return digits if digits else None

def extract_image_hash(url: str) -> Optional[str]:
    """
    Craigslist URL looks like:
      .../00f0f_3yHRC9QHgPK_600x450.jpg
    We extract the substring between the first '_' and second '_':
      3yHRC9QHgPK
    """
    if not url or not isinstance(url, str):
        return None
    # capture between first and second underscore:
    # something like _3yHRC9QHgPK_
    m = re.search(r"_([A-Za-z0-9]+)_", url)
    if m:
        return m.group(1)
    return None

def parse_pictures_urls(pic_data) -> List[str]:
    """Return a list of URL strings from the pictures column."""
    if pic_data is None or (isinstance(pic_data, float) and pd.isna(pic_data)):
        return []

    # If it's already a dict
    if isinstance(pic_data, dict):
        return [str(k) for k in pic_data.keys()]

    # If it's a string
    if isinstance(pic_data, str):
        s = pic_data.strip()
        if s == "" or s == "{}":
            return []

        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return [str(k) for k in parsed.keys()]
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return [str(k) for k in parsed.keys()]
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

        # fallback: treat as single URL-like string
        return [s]

    # fallback
    return safe_eval(pic_data)

# ----------------------------
# Build Relatedness Graph (posts only)
# ----------------------------
def build_relatedness_graph(csv_path: str, nrows: int = 5000) -> Tuple[nx.DiGraph, pd.DataFrame]:
    df = pd.read_csv(csv_path, sep=None, engine="python").head(nrows)

    post_meta: Dict[str, Dict] = {}
    post_phones: Dict[str, Set[str]] = {}
    post_hashes: Dict[str, Set[str]] = {}

    # Collect identifiers per post
    for idx, row in df.iterrows():
        pid = f"post:{idx}"
        post_meta[pid] = {
            "title": row.get("title"),
            "post": row.get("post"),
        }

        # Phones
        phones = set()
        for ph in safe_eval(row.get("ner_phone")):
            phc = normalize_phone(ph)
            if phc:
                phones.add(phc)
        post_phones[pid] = phones

        # Image hashes
        hashes = set()
        urls = parse_pictures_urls(row.get("pictures"))
        for url in urls:
            h = extract_image_hash(url)
            if h:
                hashes.add(h)
        post_hashes[pid] = hashes

    # Keep posts with at least one phone or image hash
    kept = [pid for pid in post_meta if post_phones[pid] or post_hashes[pid]]
    dropped = len(post_meta) - len(kept)
    if dropped:
        print(f"Dropped {dropped} posts with neither phone nor image.")

    # Directed graph (so SCCs exist)
    G = nx.DiGraph(name="relatedness_graph")

    # Add post nodes
    for pid in kept:
        G.add_node(pid, entity_type="post", **post_meta[pid])

    # Build maps: identifier -> list of posts
    phone_map: Dict[str, List[str]] = {}
    image_map: Dict[str, List[str]] = {}

    for pid in kept:
        for ph in post_phones[pid]:
            phone_map.setdefault(ph, []).append(pid)
        for h in post_hashes[pid]:
            image_map.setdefault(h, []).append(pid)

    # Connect posts sharing identifiers (both directions)
    def connect_shared(mapping: Dict[str, List[str]], rel: str):
        for posts in mapping.values():
            if len(posts) > 1:
                for i in range(len(posts)):
                    for j in range(i + 1, len(posts)):
                        u, v = posts[i], posts[j]
                        for a, b in [(u, v), (v, u)]:
                            if G.has_edge(a, b):
                                existing = G[a][b].get("relation")
                                if existing and rel not in existing.split(","):
                                    G[a][b]["relation"] = existing + "," + rel
                                elif not existing:
                                    G[a][b]["relation"] = rel
                            else:
                                G.add_edge(a, b, relation=rel)

    connect_shared(phone_map, "phone")
    connect_shared(image_map, "image")

    return G, df

# ----------------------------
# SCC selection: one SCC size=11 and one SCC size=15
# ----------------------------
def extract_specific_scc_sizes(G: nx.DiGraph, target_sizes=(12,)) -> nx.DiGraph:
    sccs = list(nx.strongly_connected_components(G))
    sizes = [len(c) for c in sccs]
    print(f"Total SCCs: {len(sccs)}")
    if sizes:
        print(f"Top SCC sizes: {sorted(sizes, reverse=True)[:20]}")

    selected = {}
    for comp in sccs:
        sz = len(comp)
        if sz in target_sizes and sz not in selected:
            selected[sz] = comp
        if len(selected) == len(target_sizes):
            break

    missing = [s for s in target_sizes if s not in selected]
    if missing:
        uniq_sizes = sorted(set(sizes), reverse=True)
        print(f"\nCould NOT find SCC sizes: {missing}")
        print(f"Available SCC sizes (top 50 unique): {uniq_sizes[:50]}")
        raise ValueError("Requested SCC sizes not found in this graph.")

    nodes = set().union(*selected.values())
    H = G.subgraph(nodes).copy()

    for sz, comp in selected.items():
        for n in comp:
            H.nodes[n]["scc_size"] = int(sz)

    print("\nSelected SCCs:")
    for sz in target_sizes:
        print(f"  SCC size {sz}: {len(selected[sz])} nodes")

    return H


# ----------------------------
# Plot and export
# ----------------------------
def plot_scc_subgraph(H: nx.DiGraph, out_png="scc_12.png"):
    pos = nx.spring_layout(H, seed=42)

    node_colors = []
    for n in H.nodes():
        sz = H.nodes[n].get("scc_size")
        if sz == 12:
            node_colors.append("tab:blue")
        # elif sz == 12:
        #     node_colors.append("tab:orange")
        # else:
        #     node_colors.append("gray")

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=350, alpha=0.9)
    nx.draw_networkx_edges(H, pos, edge_color="gray", alpha=0.5, arrows=False)
    # If you want labels:
    # nx.draw_networkx_labels(H, pos, font_size=7)

    plt.title("SCC of size 12")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    CSV_PATH = "/data/rashidm/car-parts/carparts_alig_no_dup_images_nodup_ner.csv"
    NROWS = 1000000  # adjust if needed

    # 1) Build full graph
    G, df = build_relatedness_graph(CSV_PATH, nrows=NROWS)

    # 2) Extract SCC size 11 and 15 subgraph
    H = extract_specific_scc_sizes(G, target_sizes=(12,))

    # 3) Export for Gephi + plot png
    # nx.write_gexf(H, "scc_size_11_and_15.gexf")
    # print("Saved: scc_size_11_and_15.gexf (open in Gephi)")

    plot_scc_subgraph(H, out_png="scc_11_15.png")
