#
# dependency_graph_plotter.py
#
# This file is part of the NEST ODE toolbox.
#
# Copyright (C) 2017 The NEST Initiative
#
# The NEST ODE toolbox is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 2 of
# the License, or (at your option) any later version.
#
# The NEST ODE toolbox is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
#

import logging

from .config import Config


class DependencyGraphPlotter:
    r"""
    Use graphviz to plot a dependency graph between state variables.
    """

    @classmethod
    def plot_graph(cls, shapes, dependency_edges, node_is_lin, fn=None):
        r"""
        Plot graph and write to file.

        :param shapes: List of Shape instances.
        :param dependency_edges: List of edges returned from dependency analysis.
        :param node_is_lin: List of Booleans returned from dependency analysis.
        :param fn: Filename to write PNG image as.
        """

        from graphviz import Digraph

        E = [(str(sym1).replace(Config().differential_order_symbol, "'"),
              str(sym2).replace(Config().differential_order_symbol, "'")) for sym1, sym2 in dependency_edges]

        dot = Digraph(comment="Dependency graph", engine="dot", format="pdf")
        dot.attr(compound="true")
        nodes = []
        for shape in shapes:
            if node_is_lin[shape.symbol]:
                style = "filled"
                colour = "#caffca"
            else:
                style = "rounded"
                colour = "black"
            if shape.order > 1:
                with dot.subgraph(name="cluster_" + str(shape.symbol)) as sg:
                    nodes.append("cluster_" + str(shape.symbol))
                    for i in range(shape.order):
                        sg.node(str(shape.symbol) + i * "'", style=style, color=colour)
            else:
                dot.node(str(shape.symbol), style=style, color=colour)
                nodes.append(str(shape.symbol))

        for e in E:
            prefer_connections_to_clusters = False
            if prefer_connections_to_clusters:
                e = list(e)
                i = 0
                if "cluster_" + str(e[i]) in nodes:
                    e[i] = "cluster_" + str(e[i])

            dot.edge(str(e[0]), str(e[1]))

        if not fn is None:
            logging.info("Saving dependency graph plot to " + fn)
            dot.render(fn)
