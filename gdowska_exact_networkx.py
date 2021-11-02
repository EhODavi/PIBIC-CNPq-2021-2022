import networkx as nx
import matplotlib.pyplot as plt
import gdowska_exact

n, x, y, amin, modelAll, modelA = gdowska_exact.main()

G = nx.DiGraph()

G.add_node(0, color='green', pos=(x[0], y[0]))

for i in range(1, n + 1):
    if i in amin:
        G.add_node(i, color='red', pos=(x[i], y[i]))
    else:
        G.add_node(i, color='blue', pos=(x[i], y[i]))


modelAll.optimize()
modelA.optimize()

w1, x1, y1, z1 = modelAll.__data
w2, x2, y2, z2 = modelA.__data

for (i, j) in x1:
    if x1[i, j].X > .5:
        G.add_edge(i, j, color='black', weight=1)

for (i, j) in x2:
    if x2[i, j].X > .5:
        G.add_edge(i, j, color='red', weight=1)

edge_colors = nx.get_edge_attributes(G, 'color').values()
weights = nx.get_edge_attributes(G, 'weight').values()
colors = [node[1]['color'] for node in G.nodes(data=True)]
pos = nx.get_node_attributes(G, 'pos')

nx.draw(G, pos=pos, edge_color=edge_colors, node_color=colors, with_labels=True, font_color='black', width=list(weights))

plt.show()
