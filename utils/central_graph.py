import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import yaml

# import utils
import sys
import matplotlib.pyplot as plt
import networkx as nx


class CentralGraph:
    def __init__(self, env_params) -> None:
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.graph = grid_world_graph((self.Nx, self.Ny))
        self.base_graph = grid_world_graph((self.Nx, self.Ny))

    def UpdateSafeInGraph(self):
        return 1


def expansion_operator(graph, true_constraint, init_node, thresh, Lc):
    # print("init_node", init_node)
    # Total safet set
    total_safe_nodes = torch.arange(0, true_constraint.shape[0])[
        true_constraint > thresh
    ]
    total_safe_nodes = torch.cat([total_safe_nodes, init_node.reshape(-1)])
    total_safe_nodes = torch.unique(total_safe_nodes)
    total_safe_graph = graph.subgraph(total_safe_nodes.numpy())
    edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
        total_safe_graph, init_node.item()
    )

    connected_nodes = [init_node.item()] + [v for u, v in edges]
    reachable_safe_graph = graph.subgraph(np.asarray(connected_nodes))

    return reachable_safe_graph

    # S_po_mat = []
    # S_po_prev = copy(init_set)
    # S_po = copy(S_po_prev)
    # S_po_mat.append(S_po_prev)
    # termin_condn = False
    # bound_left = True_Constraint[get_idx(V, [S_po_prev.Xleft])[0]].detach()
    # bound_right = True_Constraint[get_idx(V, [S_po_prev.Xright])[0]].detach()
    # set_bound = SafeSet(bound_left, bound_right, V, 0.12)
    # while not termin_condn:
    #     bound_left = True_Constraint[get_idx(V,
    #                                          [S_po_prev.Xleft])[0]].detach()
    #     bound_right = True_Constraint[get_idx(V,
    #                                           [S_po_prev.Xright])[0]].detach()
    #     set_bound.Update(bound_left, bound_right)
    #     # print((set_bound.Xleft-constraint)/Lc, (set_bound.Xright-constraint)/Lc)

    #     S_po_left = S_po_prev.Xleft
    #     for steps in range(100):
    #         if set_bound.Xleft - Lc*(0.12)*steps < thresh:
    #             if steps == 0:
    #                 break
    #             S_po_left = max(S_po_prev.Xleft - (0.12)
    #                             * (steps-1), V.min())
    #             break
    #     # if V[min(V.shape[0]-1, get_idx(V, [S_po_prev.Xleft - (set_bound.Xleft-thresh)/Lc])[0])][0] < S_po_prev.Xleft:
    #     #     S_po_left = V[get_idx(V, [
    #     #         S_po_prev.Xleft - (set_bound.Xleft-thresh)/Lc])[0]][0]

    #     S_po_right = S_po_prev.Xright
    #     for steps in range(100):
    #         if set_bound.Xright - Lc*(0.12)*steps < thresh:
    #             if steps == 0:
    #                 break
    #             S_po_right = min(S_po_prev.Xright + (0.12)
    #                              * (steps-1), V.max())
    #             break
    #     # if V[get_idx(V, [S_po_prev.Xright + (set_bound.Xright-thresh)/Lc])[0]][0] > S_po_prev.Xright:
    #     #     S_po_right = V[get_idx(V, [
    #     #         S_po_prev.Xright + (set_bound.Xright-thresh)/Lc])[0]][0]
    #     S_po.Update(S_po_left.reshape(-1), S_po_right.reshape(-1))
    #     termin_condn = ((S_po_prev.Xleft == S_po.Xleft)
    #                     and (S_po_prev.Xright == S_po.Xright))
    #     # print((S_po_prev.Xleft == S_po.Xleft), (S_po_prev.Xright == S_po.Xright))
    #     # print(termin_condn, S_po, S_po_prev)
    #     S_po_prev = copy(S_po)
    #     S_po_mat.append(S_po_prev)

    # return S_po


def grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(
        zip(grid_nodes[:, :-1].reshape(-1), grid_nodes[:, 1:].reshape(-1)), action=1
    )

    # action 2: go down
    graph.add_edges_from(
        zip(grid_nodes[:-1, :].reshape(-1), grid_nodes[1:, :].reshape(-1)), action=2
    )

    # action 3: go left
    graph.add_edges_from(
        zip(grid_nodes[:, 1:].reshape(-1), grid_nodes[:, :-1].reshape(-1)), action=3
    )

    # action 4: go up
    graph.add_edges_from(
        zip(grid_nodes[1:, :].reshape(-1), grid_nodes[:-1, :].reshape(-1)), action=4
    )

    return graph


def diag_grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(
        zip(grid_nodes[:, :-1].reshape(-1), grid_nodes[:, 1:].reshape(-1)), action=1
    )

    # action 2: go down
    graph.add_edges_from(
        zip(grid_nodes[:-1, :].reshape(-1), grid_nodes[1:, :].reshape(-1)), action=2
    )

    # action 3: go left
    graph.add_edges_from(
        zip(grid_nodes[:, 1:].reshape(-1), grid_nodes[:, :-1].reshape(-1)), action=3
    )

    # action 4: go up
    graph.add_edges_from(
        zip(grid_nodes[1:, :].reshape(-1), grid_nodes[:-1, :].reshape(-1)), action=4
    )

    graph.add_edges_from(
        zip(grid_nodes[:-1, :-1].reshape(-1), grid_nodes[1:, 1:].reshape(-1)), action=5
    )
    graph.add_edges_from(
        zip(grid_nodes[:-1, 1:].reshape(-1), grid_nodes[1:, :-1].reshape(-1)), action=6
    )
    graph.add_edges_from(
        zip(grid_nodes[1:, :-1].reshape(-1), grid_nodes[:-1, 1:].reshape(-1)), action=7
    )
    graph.add_edges_from(
        zip(grid_nodes[1:, 1:].reshape(-1), grid_nodes[:-1, :-1].reshape(-1)), action=8
    )

    return graph
