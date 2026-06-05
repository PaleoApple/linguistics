import ijson  # <-- Crucial for streaming JSON without loading it entirely into RAM
import re
import networkx as nx
from pyvis.network import Network

# --- CONFIGURATION & CONFIG ---
DATA_PATH = "wiktionary_etymology_graph.json"
OUTPUT_PATH = "multi_branch_cognate_tree.html"
TARGET_WORD = "машина (Russian)"
MAX_DISTANCE = 4.0  # Budget for Dijkstra path finding

COLOR_MAP = {
    "Proto-Indo-European": "#ef4444",
    "Proto-": "#f97316",
    "Latin": "#eab308", "Greek": "#eab308",
    "English": "#10b981", "German": "#10b981", "Dutch": "#10b981",
    "French": "#06b6d4", "Spanish": "#06b6d4", "Italian": "#06b6d4",
    "Sanskrit": "#a855f7", "Hindi": "#a855f7"
}

# Pre-compile regex for performance
affix_regex = re.compile(r"(^-$|^-\w+|\w+-$|^\*?-\w+|^-\(\)$)")

def is_noisy_node(node_id):
    """Fast check for noisy affix nodes before adding them to the graph."""
    pure_word = node_id.split(" (")[0].strip().replace(" ", "")
    return bool(affix_regex.search(pure_word))

# --- 1. STREAMING GRAPH CONSTRUCTION ---
print("Streaming etymology data into a lightweight graph structure...")
G = nx.DiGraph()

# Open the JSON using ijson to stream nodes and links one by one
with open(DATA_PATH, "rb") as f:
    # 1a. Stream and filter edges first (requires less memory than storing all nodes first)
    print(" -> Processing edges...")
    edges_stream = ijson.items(f, "links.item")
    for edge in edges_stream:
        source = edge.get("source")
        target = edge.get("target")
        
        # Early rejection of noisy affix nodes
        if is_noisy_node(source) or is_noisy_node(target):
            continue
            
        rel = edge.get("relation", "")
        weight = 1.0 if rel == "inh" else 2.0
        
        G.add_edge(source, target, relation=rel, weight=weight)

    # 1b. Stream node attributes, but ONLY for nodes that made it into our edge list
    print(" -> Processing node attributes...")
    f.seek(0) # Reset file pointer to read nodes
    nodes_stream = ijson.items(f, "nodes.item")
    for node in nodes_stream:
        node_id = node.get("id")
        if node_id in G:
            G.nodes[node_id]["glosses"] = node.get("glosses", [])
            G.nodes[node_id]["pos"] = node.get("pos", [])

if TARGET_WORD not in G:
    print(f"Error: Target node '{TARGET_WORD}' not found in the parsed network.")
    exit()

# --- 2. COGNATE MESH EXTRACTION ---
print("Calculating local neighborhood...")
# Create a lightweight undirected view for distance calculation (saves memory over to_undirected().copy())
G_undirected_view = G.to_undirected(as_view=True)

distances = nx.single_source_dijkstra_path_length(
    G_undirected_view, source=TARGET_WORD, cutoff=MAX_DISTANCE, weight="weight"
)

# Extract subgraph and immediately drop the giant graph to free memory
sub_G = G.subgraph(distances.keys()).copy()
del G 
print(f"Extracted clean mesh of {sub_G.number_of_nodes()} nodes.")

# --- 3. VISUALIZATION GENERATION ---
net = Network(height="850px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)
centrality = nx.betweenness_centrality(sub_G)

for node in sub_G.nodes:
    size = 50 if node == TARGET_WORD else 15 + (centrality.get(node, 0) * 300)
    
    color = "#94a3b8"
    for key, hex_color in COLOR_MAP.items():
        if key in node:
            color = hex_color
            break
            
    node_attr = sub_G.nodes[node]
    gloss_list = node_attr.get("glosses", [])
    pos_list = node_attr.get("pos", [])
    
    pos_str = f" [{', '.join(pos_list)}]" if pos_list else ""
    
    if gloss_list:
        hover_text = "\n".join(f"• {g}" for g in gloss_list[:5])
    else:
        hover_text = "• Definition unavailable."
        
    hover_title = f"{node}{pos_str}\n{'=' * len(node)}\n{hover_text}"
    net.add_node(node, label=node, color=color, size=size, title=hover_title)

for source, target, data in sub_G.edges(data=True):
    rel = data.get("relation", "")
    is_borrowing = rel == "bor"
    
    net.add_edge(
        source, 
        target, 
        color="#f43f5e" if is_borrowing else "#4b5563",
        title=f"Relation: {rel}", 
        style="dashed" if is_borrowing else "continuous",
        smooth={"type": "cubicBezier"}
    )

net.set_options("""
{
  "physics": {
    "forceAtlas2Based": { "gravitationalConstant": -120, "centralGravity": 0.01, "springLength": 140, "springStrength": 0.08 },
    "solver": "forceAtlas2Based"
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200
  }
}
""")

net.write_html(OUTPUT_PATH)
print(f"\n✨ Refactor complete! Open '{OUTPUT_PATH}' to evaluate semantic drift interactively.")
