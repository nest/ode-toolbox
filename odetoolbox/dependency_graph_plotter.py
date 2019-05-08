from .shapes import Shape

import sympy


class DependencyGraphPlotter:

    @classmethod
    def plot_graph(cls, shapes, shape_sys, fn):

        dependency_edges = shape_sys.get_dependency_edges()
        node_is_lin = shape_sys.get_lin_cc_symbols(dependency_edges)

        from graphviz import Digraph
        
        E = [ (str(sym1).replace("__d", "'"), str(sym2).replace("__d", "'")) for sym1, sym2 in dependency_edges ]
        
        #for shape1 in shapes:
            #for shape2 in shapes:

                ## check if symb1 occurs in the expression for symb2
                #shape2_depends_on_shape1 = shape2.diff_rhs_derivatives.has(shape1.symbol)

                #if not shape2_depends_on_shape1:
                    #for derivative_factor in shape2.derivative_factors:
                        #if derivative_factor.has(shape1.symbol):
                            ## shape 2 depends on shape 1
                            #shape2_depends_on_shape1 = True
                            #break

                #if shape2_depends_on_shape1:
                    #E.append((str(shape2.symbol), str(shape1.symbol)))

        dot = Digraph(comment="Dependency graph", engine="dot", format="pdf")#, engine="fdp")#, format="pdf")
        dot.attr(compound="true")
        nodes = []
        for shape in shapes:
            #if shape.is_lin_const_coeff(shapes):
            if node_is_lin[shape.symbol]:
                style = "filled"
                colour = "chartreuse"
            else:
                style = "rounded"
                colour = "black"
            if shape.order > 1:
                with dot.subgraph(name="cluster_" + str(shape.symbol)) as sg:
                    nodes.append("cluster_" + str(shape.symbol))
                    sg.attr(label=str(shape.symbol))
                    for i in range(shape.order):
                        sg.node(str(shape.symbol) + i * "'", style=style, color=colour)#, str(shape.symbol) + str(i))
                        print("Creating sg node for " + str(shape.symbol) + i * "'" + ", colour = " + str(colour))
            else:
                dot.node(str(shape.symbol), style=style, color=colour)
                nodes.append(str(shape.symbol))
                print("Creating order 1 node for " + str(shape.symbol) + ", colour = " + str(colour))

        for e in E:
            prefer_connections_to_clusters = False
            if prefer_connections_to_clusters:
                e = list(e)
                for i in range(2):
                    if "cluster_" +e[i] in nodes:
                        e[i] = "cluster_" + e[i]

            #print("Edge from " + str(e[0]) + " to " + str(e[1]))
            dot.edge(str(e[0]), str(e[1]))
        #dot.view()
        print("Saving to " + fn)
        dot.render(fn)
