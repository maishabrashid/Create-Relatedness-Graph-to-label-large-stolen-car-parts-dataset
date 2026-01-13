import pandas as pd
import networkx as nx
import ast, re, json

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
    """Convert spelled-out or glued words into digits"""
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

def normalize_phone(s: str) -> str | None:
    digits = words_to_digits(s)
    return digits if digits else None

# Extract image hash between first and second underscore
def extract_image_hash(url: str) -> str | None:
    m = re.search(r"_[A-Za-z0-9]+_", url)
    if m:
        return m.group(0).strip("_")
    return None

# ----------------------------
# Build Relatedness Graph
# ----------------------------
def build_relatedness_graph(csv_path, nrows=1000000):
    df = pd.read_csv(csv_path, sep=None, engine="python").head(nrows)

    post_meta   = {}   # post_id -> dict(title, post)
    post_phones = {}   # post_id -> set(phones)
    post_hashes = {}   # post_id -> set(image hashes)

    for idx, row in df.iterrows():
        pid = f"post:{idx}"
        post_meta[pid] = {"title": row.get("title"), "post": row.get("post")}

        # --- Phones ---
        phones = set()
        for ph in safe_eval(row.get("ner_phone")):
            phc = normalize_phone(ph)
            if phc:
                phones.add(phc)
        post_phones[pid] = phones

        # --- Images ---
        hashes = set()
        pic_data = row.get("pictures")
        try:
            # try to parse JSON directly if string looks like dict
            if isinstance(pic_data, str):
                parsed = json.loads(pic_data.replace("'", "\"")) if "{" in pic_data else ast.literal_eval(pic_data)
                if isinstance(parsed, dict):
                    urls = list(parsed.keys())
                elif isinstance(parsed, list):
                    urls = parsed
                else:
                    urls = [pic_data]
            elif isinstance(pic_data, dict):
                urls = list(pic_data.keys())
            else:
                urls = []
        except Exception:
            urls = safe_eval(pic_data)

        for url in urls:
            if isinstance(url, str):
                h = extract_image_hash(url)
                if h:
                    hashes.add(h)
        post_hashes[pid] = hashes

    # Drop posts with neither phone nor image
    kept = [pid for pid in post_meta if post_phones[pid] or post_hashes[pid]]
    dropped = len(post_meta) - len(kept)
    if dropped:
        print(f"Dropped {dropped} posts with neither phone nor image.")

    # Build directed graph
    G = nx.DiGraph(name="relatedness_graph")

    for pid in kept:
        G.add_node(pid, entity_type="post", **post_meta[pid])

    # Index identifiers → posts
    phone_map, image_map = {}, {}
    for pid in kept:
        for ph in post_phones[pid]:
            phone_map.setdefault(ph, []).append(pid)
        for h in post_hashes[pid]:
            image_map.setdefault(h, []).append(pid)

    # Connect posts that share identifiers (both directions)
    def connect_shared(mapping, rel):
        for posts in mapping.values():
            if len(posts) > 1:
                for i in range(len(posts)):
                    for j in range(i + 1, len(posts)):
                        u, v = posts[i], posts[j]
                        # add both directions
                        for (a,b) in [(u,v),(v,u)]:
                            if G.has_edge(a,b):
                                existing = G[a][b].get("relation")
                                if existing and rel not in existing.split(","):
                                    G[a][b]["relation"] = existing + "," + rel
                            else:
                                G.add_edge(a,b, relation=rel)

    connect_shared(phone_map, "phone")
    connect_shared(image_map, "image")

    return G, post_phones, post_hashes

# ----------------------------
# SCC computation & Export
# ----------------------------
# ----------------------------
# SCC computation & Export
# ----------------------------
def export_scc_csvs(G, post_phones, post_hashes,
                    scc_summary_path="scc_summary.csv",
                    scc_members_path="scc_members.csv"):
    # Compute strongly connected components
    sccs = list(nx.strongly_connected_components(G))

    # Print number and size of SCCs
    print(f"\nTotal Strongly Connected Components (SCCs): {len(sccs)}")
    sizes = [len(c) for c in sccs]
    sizes_sorted = sorted(sizes, reverse=True)
    print(f"SCC sizes (sorted, top 20 shown): {sizes_sorted[:20]}")
    print(f"Largest SCC size: {max(sizes)}")
    print(f"Smallest SCC size: {min(sizes)}\n")

    # Assign each node to its SCC id
    comp_id_of = {}
    for cid, comp in enumerate(sccs, start=1):
        for n in comp:
            comp_id_of[n] = cid

    # Members CSV
    members = []
    for n in G.nodes():
        members.append({
            "component_id": comp_id_of.get(n),
            "post_id": n,
            "title": G.nodes[n].get("title"),
            "degree_out": G.out_degree(n),
            "degree_in": G.in_degree(n),
            "phones": "|".join(sorted(post_phones.get(n, []))),
            "image_hashes": "|".join(sorted(post_hashes.get(n, []))),
        })
    #pd.DataFrame(members).sort_values(["component_id","post_id"]).to_csv(scc_members_path, index=False)

    # Summary CSV
    summary = []
    for cid, comp in enumerate(sccs, start=1):
        comp = list(comp)
        phones_in, imgs_in = [], []
        for n in comp:
            phones_in.extend(list(post_phones.get(n, [])))
            imgs_in.extend(list(post_hashes.get(n, [])))
        phone_counts = pd.Series(phones_in).value_counts() if phones_in else pd.Series(dtype=int)
        img_counts   = pd.Series(imgs_in).value_counts()   if imgs_in   else pd.Series(dtype=int)

        summary.append({
            "component_id": cid,
            "size": len(comp),
            "sample_posts": ", ".join(comp[:5]),
            "shared_phones": "; ".join([f"{k}:{v}" for k,v in phone_counts[phone_counts>=2].head(10).items()]),
            "shared_images": "; ".join([f"{k}:{v}" for k,v in img_counts[img_counts>=2].head(10).items()]),
        })
    #pd.DataFrame(summary).sort_values("size", ascending=False).to_csv(scc_summary_path, index=False)

    print(f"SCCs: {len(sccs)}  |  Saved {scc_summary_path} and {scc_members_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    CSV_PATH = "/data/rashidm/car-parts/carparts_alig_no_dup_images_nodup_ner.csv"
    NROWS = 1000000

    G, post_phones, post_hashes = build_relatedness_graph(CSV_PATH, nrows=NROWS)

    export_scc_csvs(G, post_phones, post_hashes,
                    scc_summary_path="scc_summary.csv",
                    scc_members_path="scc_members.csv")

