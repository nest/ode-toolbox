import sympy


class DependencyGraphPlotter:

    @classmethod
    def plot_graph(cls, shapes, dependency_edges, node_is_lin, fn=None):

        from graphviz import Digraph

        E = [ (str(sym1).replace("__d", "'"), str(sym2).replace("__d", "'")) for sym1, sym2 in dependency_edges ]

        dot = Digraph(comment="Dependency graph", engine="dot", format="pdf")
        dot.attr(compound="true")
        nodes = []
        for shape in shapes:
            if node_is_lin[shape.symbol]:
                style = "filled"
                colour = "chartreuse"
            else:
                style = "rounded"
                colour = "black"
            if shape.order > 1:
                with dot.subgraph(name="cluster_" + str(shape.symbol)) as sg:
                    nodes.append("cluster_" + str(shape.symbol))
                    #print("appending node " + "cluster_" + str(shape.symbol))
                    #sg.attr(label=str(shape.symbol))
                    for i in range(shape.order):
                        sg.node(str(shape.symbol) + i * "'", style=style, color=colour)#, str(shape.symbol) + str(i))
                        #print("Creating sg node for " + str(shape.symbol) + i * "'" + ", colour = " + str(colour))
            else:
                dot.node(str(shape.symbol), style=style, color=colour)
                nodes.append(str(shape.symbol))
                #print("Creating order 1 node for " + str(shape.symbol) + ", colour = " + str(colour))

        for e in E:
            prefer_connections_to_clusters = False
            if prefer_connections_to_clusters:
                e = list(e)
                i = 0
                if "cluster_" + str(e[i]) in nodes:
                    #print("Rewriting " + str(e[i]) + " to " + str("cluster_" + str(e[i])))
                    e[i] = "cluster_" + str(e[i])

            #print("Edge from " + str(e[0]) + " to " + str(e[1]))
            dot.edge(str(e[0]), str(e[1]))
        #dot.view()
        if not fn is None:
            print("Saving to " + fn)
            dot.render(fn)
