#!/usr/bin/env python

"""
    plot.py

    This file contains shared functions for plotting graphs
    and networks.
"""

from numpy import *
import numpy.random

import matplotlib.pyplot as plt
import networkx as nx

def save_plot(name=""):
    def wrap(f):
        def wrapped(*args, **kwargs):
            fname = ""
            fext = ""
            dpi=200
            if 'fname' in list(kwargs.keys()):
                fname = kwargs['fname']
                del kwargs['fname']
            if 'fext' in list(kwargs.keys()):
                fext = kwargs['fext']
                del kwargs['fext']
            if 'dpi' in list(kwargs.keys()):
                dpi = kwargs['dpi']
                del kwargs['dpi']

            retval = f(*args, **kwargs)
            if fname != "":
                plt.savefig(fname + "_" + name + fext, dpi=dpi)

            return retval

        return wrapped

    return wrap

@save_plot(name="hierarchical_tree")
def draw_tree(tree, pos=None, arrows=False, return_edges=False):
    """
        Draws the hierarchical tree generated by hierarchical_decomposition.
        If pos==None, positions are calculated by using graphviz's dot.
        Returns the calculated positions for the tree nodes.
    """
    plt.title("Hierarchical decomposition tree")

    # No idea *why* this line has to be here, but it fixes a bug (?) in
    # pygraphviz
    ag = nx.to_agraph(tree)
    
    if pos == None:
        pos = nx.graphviz_layout(tree, prog='dot')

    edges = nx.draw_networkx_edges(tree, 
            pos, with_labels=False, node_size=0, arrows=arrows)
    
    if return_edges:
        return pos, edges
    else:
        return pos

def draw_leaf_raw(leaf, title="Leaf", edge_list=None, mark_edges=None,
        color=None, fixed_width=False, node_pos=None):
    """
        Draws the leaf network using the data in the edges.
	If mark_edges is not None, it contains a list of tuples of
    the form (index, color).
    """
    plt.title(title)

    if node_pos == None:
        pos = {}
        for k in list(leaf.node.keys()):
            pos[k] = (leaf.node[k]['x'], leaf.node[k]['y'])
    else:
        pos = node_pos
    
    if edge_list == None:
        es = leaf.edges_iter()
    else:
        es = edge_list
    
    if fixed_width:
        widths = 4.
    else:
        widths = 2*array([leaf[e[0]][e[1]]['conductivity'] \
                for e in es])
        widths = 10./amax(widths)*widths
    
    if mark_edges and not edge_list:
        col_list = leaf.number_of_edges()*['k']
        for i, c in mark_edges:
            col_list[i] = c

        edges = nx.draw_networkx_edges(leaf, pos=pos, width=widths, \
                edge_color=col_list)
    elif color and edge_list:
        edges = nx.draw_networkx_edges(leaf, pos=pos, width=widths, \
                edge_color=color, edgelist=edge_list)
    else:
        edges = nx.draw_networkx_edges(leaf, edgelist=edge_list, pos=pos, \
                width=widths)

    #nx.draw(leaf, pos=pos)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().axis('equal')

    return edges

@save_plot(name="leaf_network")
def draw_leaf(*args, **kwargs):
    draw_leaf_raw(*args, **kwargs)

@save_plot(name="dual")
def draw_dual(dual):
    """
        Draws the dual network to the pruned cycle
        form.
    """
    pos = {}
    for k in list(dual.node.keys()):
        pos[k] = (dual.node[k]['x'], dual.node[k]['y'])

    widths = array([1./d['conductivity']
        for u, v, d in dual.edges_iter(data=True)])

    #widths = 5./mean(widths)*widths

    widths[isinf(widths)] = 0.

    #nx.draw_networkx_nodes(dual, pos=pos)
    nx.draw(dual, pos=pos, with_labels=False, node_size=10,
            width=widths)

@save_plot(name="filtration")
def draw_filtration(filtration, steps=9, biased=True):
    """ Draws several steps of the filtration.
    If biased is true, the steps chosen are biased towards the end
    of the decomposition (in a power way)
    """
    plt.title("Filtration steps")
    n = len(filtration)
    
    for i in range(steps):
        if biased:
            s = (i/float(steps - 1.))**0.15
        else:
            s = i/float(steps - 1.)

        j = int(floor(s*(n-1)))
        fil = filtration[j]

        plt.subplot(ceil(float(steps)/3), 3, i+1)
        #draw_leaf(fil, title="Step {}".format(filtration.step_nums[j]))
        draw_leaf_raw(fil, title="")

