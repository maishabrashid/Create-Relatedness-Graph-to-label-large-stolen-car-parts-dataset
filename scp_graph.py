import pandas as pd
import networkx as nx
import ast, re

# =========================
# Helpers
# =========================
def safe_eval(val):
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list): return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str):  return [parsed.strip()]
        return []
    except Exception:
        return [str(val).strip()]

_WORD2DIG = {
    "zero":"0","oh":"0","o":"0",
    "one":"1","two":"2","three":"3","four":"4","for":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
}
_WORD_RE = re.compile("|".join(sorted(_WORD2DIG, key=len, reverse=True)))
def words_to_digits(s: str) -> str:
    if not s: return ""
    t = s.lower()
    def _repeat(m):
        mult = 2 if m.group(1) == "double" else 3
        return _WORD2DIG.get(m.group(2), "") * mult
    t = re.sub(r"\b(double|triple)\s+(zero|oh|o|one|two|three|four|for|five|six|seven|eight|nine)\b", _repeat, t)
    t = _WORD_RE.sub(lambda m: _WORD2DIG[m.group(0)], t)
    return re.sub(r"\D", "", t)

def normalize_phone(s: str) -> str | None:
    d = words_to_digits(s)
    return d if d else None

def norm_location(x):
    return str(x).strip().lower() if x and str(x).strip() else None

# =========================
# Build Relatedness Graph (Directed)
# =========================
def build_relatedness_graph(csv_path: str, nrows: int = 5000, use_locations: bool = False):
    df = pd.read_csv(csv_path, sep=None, engine="python").head(nrows)

    # Keep per-post metadata and identifiers so we can report them later
    post_meta   = {}   # post_id -> dict(title, post)
    post_phones = {}   # post_id -> set(phones)
    post_locs   = {}   # post_id -> set(locs)

    # Normalize identifiers per row
    for idx, row in df.iterrows():
        pid = f"post:{idx}"
        post_meta[pid] = {"title": row.get("title"), "post": row.get("post")}

        phones = set()
        for ph in safe_eval(row.get("ner_phone")):
            phc = normalize_phone(ph)
            if phc: phones.add(phc)
        post_phones[pid] = phones

        locs_raw = safe_eval(row.get("ner_location"))
        if not locs_raw and row.get("location"):  # fallback
            locs_raw = [row.get("location")]
        locs = set()
        for lc in locs_raw:
            nl = norm_location(lc)
            if nl: locs.add(nl)
        post_locs[pid] = locs

    # Drop posts with neither phone nor (if enabled) location
    kept = []
    for pid in post_meta.keys():
        if use_locations:
            if post_phones[pid] or post_locs[pid]:
                kept.append(pid)
        else:
            if post_phones[pid]:
                kept.append(pid)

    # Build directed graph; edges are symmetric (both directions) so SCCs make sense
    G = nx.DiGraph(name="relatedness_graph")

    for pid in kept:
        G.add_node(pid, entity_type="post", **post_meta[pid])

    # Indexes: identifier -> posts
    phone_map, loc_map = {}, {}
    for pid in kept:
        for ph in post_phones[pid]:
            phone_map.setdefault(ph, []).append(pid)
        if use_locations:
            for lc in post_locs[pid]:
                loc_map.setdefault(lc, []).append(pid)

    def connect_shared(mapping, rel):
        for posts in mapping.values():
            if len(posts) > 1:
                for i in range(len(posts)):
                    for j in range(i+1, len(posts)):
                        u, v = posts[i], posts[j]
                        # add both directions; relation kept as comma string (GEXF-safe)
                        if G.has_edge(u, v):
                            G[u][v]["relation"] = ",".join(sorted(set(G[u][v]["relation"].split(",")) | {rel}))
                        else:
                            G.add_edge(u, v, relation=rel)
                        if G.has_edge(v, u):
                            G[v][u]["relation"] = ",".join(sorted(set(G[v][u]["relation"].split(",")) | {rel}))
                        else:
                            G.add_edge(v, u, relation=rel)

    connect_shared(phone_map, "phone")
    if use_locations:
        connect_shared(loc_map, "location")

    return G, post_phones, post_locs

# =========================
# SCC computation + CSV export
# =========================
def export_scc_csvs(G: nx.DiGraph, post_phones: dict, post_locs: dict,
                    scc_summary_path="scc_summary.csv", scc_members_path="scc_members.csv"):
    # List SCCs (each is a set of post ids)
    sccs = list(nx.strongly_connected_components(G))
    # Assign component ids
    comp_id_of = {}
    for cid, comp in enumerate(sccs, start=1):
        for n in comp:
            comp_id_of[n] = cid

    # Members CSV: one row per post
    rows = []
    for n in G.nodes():
        rows.append({
            "component_id": comp_id_of.get(n),
            "post_id": n,
            "title": G.nodes[n].get("title"),
            "degree_out": G.out_degree(n),
            "degree_in": G.in_degree(n),
            "phones": "|".join(sorted(post_phones.get(n, []))),
            "locations": "|".join(sorted(post_locs.get(n, []))),
        })
    pd.DataFrame(rows).sort_values(["component_id","post_id"]).to_csv(scc_members_path, index=False)

    # Summary CSV: one row per SCC
    summary = []
    for cid, comp in enumerate(sccs, start=1):
        comp = list(comp)
        # compute identifiers that appear in >= 2 posts inside this SCC (useful signal)
        phones_in = []
        locs_in = []
        for n in comp:
            phones_in.extend(list(post_phones.get(n, [])))
            locs_in.extend(list(post_locs.get(n, [])))
        # counts
        phone_counts = pd.Series(phones_in).value_counts() if phones_in else pd.Series(dtype=int)
        loc_counts   = pd.Series(locs_in).value_counts()   if locs_in   else pd.Series(dtype=int)

        summary.append({
            "component_id": cid,
            "size": len(comp),
            "sample_posts": ", ".join(comp[:5]),
            "shared_phones": "; ".join([f"{k}:{v}" for k,v in phone_counts[phone_counts>=2].head(10).items()]),
            "shared_locations": "; ".join([f"{k}:{v}" for k,v in loc_counts[loc_counts>=2].head(10).items()]),
        })
    pd.DataFrame(summary).sort_values("size", ascending=False).to_csv(scc_summary_path, index=False)

    print(f"SCCs: {len(sccs)}  |  Saved {scc_summary_path} and {scc_members_path}")

if __name__ == "__main__":
    CSV_PATH = "/data/rashidm/car-parts/carparts_alig_no_dup_images_nodup_ner.csv"  # <-- update path
    NROWS = 1000000
    USE_LOCATIONS = False   # set True to include location-based edges as well

    G, post_phones, post_locs = build_relatedness_graph(CSV_PATH, nrows=NROWS, use_locations=USE_LOCATIONS)

    # Export SCCs to CSV
    export_scc_csvs(G, post_phones, post_locs,
                    scc_summary_path="scc_summary.csv",
                    scc_members_path="scc_members.csv")

