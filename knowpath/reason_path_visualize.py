import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set global font and size
rcParams['font.size'] = 14 
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# get sub graph from KnowPath
reasoning_paths = [
    ["2003 Football League Cup Final", "->", "sports.sports_championship_event.season", 
     "->", "2002–03 Football League Cup", "->", "time.event.instance_of_recurring_event", "->", "Football League Cup"],
    ["Liverpool F.C.", "->", "sports.sports_team.championships", 
     "->", "2002–03 Football League Cup", "->", "time.event.instance_of_recurring_event", "->", "Football League Cup"]
]
reasoning_paths = [
				["Tao Te Ching", "->", "religion.religious_text.religious_text_of", "->", "Taoism", "->", "religion.religion.notable_figures", "->", "Zhang Jue"],
				["Daozang", "->", "religion.religious_text.religious_text_of", "->", "Taoism", "->", "religion.religion.notable_figures", "->", "Zhang Jue"],
				["Zhuang Zhou", "->", "religion.religious_text.religious_text_of", "->", "Taoism", "->", "religion.religion.notable_figures", "->", "Zhang Jue"],
				["I Ching", "->", "religion.religious_text.religious_text_of", "->", "Taoism", "->", "religion.religion.notable_figures", "->", "Zhang Jue"]
			]
# Creating a directed graph
G = nx.DiGraph()

# Adding edges and nodes
for path in reasoning_paths:
    for i in range(0, len(path)-1, 2):
        source = path[i]
        target = path[i+2]
        G.add_edge(source, target, label=path[i+1])  # Save Arrow Label

# Use hierarchical layout (for directed acyclic graphs)
try:
    pos = nx.multipartite_layout(G, subset_key="layer")  # Try a layered layout
except:
    pos = nx.spring_layout(G, k=1.5, iterations=100)  # Alternative flex layout

# Drawing a shape
plt.figure(figsize=(16, 12))

# Drawing nodes and edges
nx.draw_networkx_nodes(
    G, pos, 
    node_size=10000,     
    node_color="lightblue",
    edgecolors="black",
    linewidths=2,
    alpha=0.9
)

nx.draw_networkx_edges(
    G, pos, 
    arrowstyle="->", 
    arrowsize=25,        
    width=2, 
    edge_color="gray"
)

# Draw node labels (automatically wrap long text)
labels = {node: '\n'.join(node.split(' ')) for node in G.nodes()}
nx.draw_networkx_labels(
    G, pos, 
    labels=labels,
    font_size=20,
    font_family="sans-serif"
)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(
    G, pos, 
    edge_labels=edge_labels,
    font_color="red",
    font_size=20,
    rotate=False
)


plt.axis('off')

# Save high-quality images
plt.savefig("reasoning_paths.png", format="PNG", dpi=300)
plt.savefig("reasoning_paths.pdf", format="PDF")
plt.show()