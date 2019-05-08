from .shapes import Shape

import sympy


class DependencyGraphPlotter:

    @classmethod
    def plot_graph(cls, shapes, shape_sys, fn):

        E = []
        
        # XXX: TODO: this dependency analysis does not cover a potentially nonlinear part
        
        for i, sym1 in enumerate(shape_sys.x_):
            for j, sym2 in enumerate(shape_sys.x_):
                if not sympy.simplify(shape_sys.A_[j, i]) == sympy.parsing.sympy_parser.parse_expr("0"):
                    E.append((sym2, sym1))
                    #E.append((str(sym2).replace("__d", "'"), str(sym1).replace("__d", "'")))
                else:
                    if not sympy.simplify(sympy.diff(shape_sys.C_[j], sym1)) == sympy.parsing.sympy_parser.parse_expr("0"):
                        E.append((sym2, sym1))
                        #E.append((str(sym2).replace("__d", "'"), str(sym1).replace("__d", "'")))

        # initial pass: is a node linear and constant coefficient by itself?
        node_is_lin = {}
        for shape in shapes:
            if shape.is_lin_const_coeff(shapes):
                _node_is_lin = True
            else:
                _node_is_lin = False
            all_shape_symbols = [ sympy.Symbol(str(shape.symbol) + "__d" * i) for i in range(shape.order) ]
            for sym in all_shape_symbols:
                node_is_lin[sym] = _node_is_lin

        # propagate: if a node depends on a node that is not linear and constant coefficient, it cannot be linear and constant coefficient

        queue = [ sym for sym, is_lin_cc in node_is_lin.items() if not is_lin_cc ]
        while len(queue) > 0:

            n = queue.pop(0)

            if not node_is_lin[n]:
                # mark dependent neighbours as also not lin_cc
                dependent_neighbours = [ n1 for (n1, n2) in E if n2 == n ]    # nodes that depend on n
                for n_neigh in dependent_neighbours:
                    print("\t\tMarking dependent node " + str(n_neigh))
                    if node_is_lin[n_neigh]:
                        node_is_lin[n_neigh] = False
                        queue.append(n_neigh)
 


        #node_is_visited = { n : False for n in shape_sys.x_ }
        ## pick a random unvisited node to begin with
        #n = shape_sys.x_[0]
        ##initial_nodes = [ sym for sym, is_lin_cc in node_is_lin.items() if is_lin_cc ]
        #while True:
            #print("Checking node " + str(n))
            #dependent_neighbours = [ n2 for (n1, n2) in E if n2 == n ]    # nodes that depend on n
            #if not node_is_lin[n]:
                #print("\tNode is not lin_cc")
                ## mark dependent neighbours as also not lin_cc
                #for n_neigh in dependent_neighbours:
                    #print("\t\tMarking dependent node " + str(n_neigh))
                    #node_is_lin[n_neigh] = False
            
            ## mark `n` as visited
            #node_is_visited[n] = True
            
            ## pick a new unvisited node
            #all_visited = True
            #for n, visited in node_is_visited.items():
                #if not visited:
                    #all_visited = False
                    #break
            
            #if all_visited:
                #break
            
        
        import pdb;pdb.set_trace()


        from graphviz import Digraph
        
        E = [ (str(sym1).replace("__d", "'"), str(sym2).replace("__d", "'")) for sym1, sym2 in E ]
        
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
