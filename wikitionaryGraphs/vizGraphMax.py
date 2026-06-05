import json
import networkx as nx
from networkx.readwrite import json_graph
from pyvis.network import Network

print("Loading graph...")
with open("wiktionary_etymology_graph.json", "r", encoding="utf-8") as f:
    graph_data = json.load(f)
G = json_graph.node_link_graph(graph_data)

# Target a modern word to find its distant global cousins
target_word = "mead (English)"

if target_word not in G:
    print(f"Could not find exact node '{target_word}'.")
    exit()

# --- THE COUSIN CALCULATOR (k-Hop Neighborhood) ---
# We temporarily treat the graph as undirected to find relatives 'up and across' branches
G_undirected = G.to_undirected()

# nx.single_source_shortest_path_length finds all nodes within 'cutoff' steps
# Setting cutoff=2 or 3 lets us go up to the Proto-language and back down to other branches
max_steps = 3
distances = nx.single_source_shortest_path_length(G_undirected, source=target_word, cutoff=max_steps)
cognate_mesh_nodes = list(distances.keys())

# Extract the directed subgraph containing this entire multi-branch family mesh
sub_G = G.subgraph(cognate_mesh_nodes).copy()
print(f"Extracted a family mesh of {sub_G.number_of_nodes()} related words.")

# --- THE INTERACTIVE GRAPH LAYOUT ---
net = Network(height="850px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)

# Calculate betweenness centrality to naturally highlight the major ancestral hubs (like PIE roots)
centrality = nx.betweenness_centrality(sub_G)

for node in sub_G.nodes:
    # Size nodes dynamically by their structural importance
    node_size = 15 + (centrality.get(node, 0) * 300)
    if node == target_word:
        node_size = 50  # Make your search anchor stand out
        
    # Visual profiling by language branch
    color = "#94a3b8"  # Default gray
    if "Proto-Indo-European" in node:
        color = "#ef4444"  # Ultimate Grandfather Root (Crimson)
    elif "Proto-" in node:
        color = "#f97316"  # Intermediate Proto-Families like Proto-Germanic/Proto-Italic (Orange)
    elif "Latin" in node or "Greek" in node:
        color = "#eab308"  # Classical Branches (Gold)
    elif "English" in node or "German" in node or "Dutch" in node:
        color = "#10b981"  # Germanic Cousins (Emerald Green)
    elif "French" in node or "Spanish" in node or "Italian" in node:
        color = "#06b6d4"  # Romance Cousins (Cyan)
    elif "Sanskrit" in node or "Hindi" in node:
        color = "#a855f7"  # Indo-Aryan Cousins (Purple)

    net.add_node(node, label=node, color=color, size=node_size)

# Add edges with distinct styles for pure inheritance vs lateral borrowing
for source, target, data in sub_G.edges(data=True):
    edge_color = "#4b5563"
    edge_style = "continuous"
    relation = data.get("relation", "")
    
    if relation == "bor":
        edge_color = "#f43f5e"  # Highlight loanwords running across branches
        edge_style = "dashed"
        
    net.add_edge(source, target, color=edge_color, title=relation, smooth={"type": "cubicBezier"})

# Configure structural physics for a beautiful tree distribution
net.set_options("""
var options = {
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -100,
      "centralGravity": 0.01,
      "springLength": 120,
      "springStrength": 0.08
    },
    "solver": "forceAtlas2Based"
  }
}
""")

net.write_html("multi_branch_cognate_tree.html")
print("Complete multi-branch cognate tree generated! Open 'multi_branch_cognate_tree.html'.")
