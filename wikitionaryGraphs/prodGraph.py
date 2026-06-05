import gzip
import json
import networkx as nx
from networkx.readwrite import json_graph

input_file = "raw-wiktextract-data.jsonl.gz"
G = nx.DiGraph()

# Quick normalization map for common ancestral languages in Wiktionary
# This ensures "ine-pro" in a template matches "Proto-Indo-European" as a main language entry
LANG_MAP = {
    "en": "English",
    "la": "Latin",
    "grc": "Ancient Greek",
    "fr": "French",
    "enm": "Middle English",
    "ang": "Old English",
    "gmw-pro": "Proto-West Germanic",
    "gem-pro": "Proto-Germanic",
    "ine-pro": "Proto-Indo-European",
    "itc-pro": "Proto-Italic",
    "de": "German",
    "es": "Spanish",
    "it": "Italian"
}

def clean_lang(lang_code_or_name):
    """Normalize language codes to their full names to ensure nodes snap together."""
    return LANG_MAP.get(lang_code_or_name, lang_code_or_name)

def make_node_id(word, lang):
    if not word or not lang:
        return None
    # Strip reconstruction asterisks to make matching cleaner across templates
    clean_word = word.strip().lstrip("*")
    clean_language = clean_lang(lang.strip())
    return f"{clean_word} ({clean_language})"

print("Processing JSONL stream with language normalization...")

with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        word = data.get("word")
        lang = data.get("lang")
        if not word or not lang:
            continue
            
        current_node = make_node_id(word, lang)
        
        # 1. Past Ancestry
        if "etymology_templates" in data:
            for template in data["etymology_templates"]:
                name = template.get("name")
                args = template.get("args", {})
                
                if name in ["inh", "der", "bor"]:
                    parent_lang = args.get("2")
                    parent_word = args.get("3")
                    
                    if parent_word and parent_lang:
                        parent_node = make_node_id(parent_word, parent_lang)
                        G.add_edge(parent_node, current_node, relation=name)

        # 2. Future Descendants
        if "descendants" in data:
            depth_stack = {0: current_node}
            for line_item in data["descendants"]:
                depth = line_item.get("depth", 1)
                for template in line_item.get("templates", []):
                    if template.get("name") in ["desc", "desctree"]:
                        args = template.get("args", {})
                        desc_lang = args.get("1")
                        desc_word = args.get("2")
                        
                        if desc_word and desc_lang:
                            desc_node = make_node_id(desc_word, desc_lang)
                            
                            potential_parent_depth = depth - 1
                            while potential_parent_depth >= 0 and potential_parent_depth not in depth_stack:
                                potential_parent_depth -= 1
                            
                            ancestor_node = depth_stack.get(potential_parent_depth, current_node)
                            G.add_edge(ancestor_node, desc_node, relation="descendant")
                            depth_stack[depth] = desc_node

        if line_num % 200000 == 0 and line_num > 0:
            print(f"Processed {line_num} lines... Current Graph: {G.number_of_nodes()} nodes.")

# Save unified JSON
print("Saving unified graph...")
with open("wiktionary_etymology_graph.json", "w", encoding="utf-8") as f:
    json.dump(json_graph.node_link_data(G), f, ensure_ascii=False)
print("Done!")
