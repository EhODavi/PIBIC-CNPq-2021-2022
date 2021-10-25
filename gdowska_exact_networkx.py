import networkx as nx
import matplotlib.pyplot as plt
import gdowska_exact

n, x, y, amin = gdowska_exact.main()

G = nx.Graph()

for i in range(n + 1):
    if i in amin:
        G.add_node(i, color='yellow', pos=(x[i], y[i]))
    else:
        G.add_node(i, color='blue', pos=(x[i], y[i]))

colors = [node[1]['color'] for node in G.nodes(data=True)]
pos = nx.get_node_attributes(G, 'pos')

nx.draw(G, pos=pos, node_color=colors, with_labels=True, font_color='black')

plt.show()
