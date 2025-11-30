# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import networkx as nx
import streamlit.components.v1 as components

from lm_lens.models.transparent_lm import ModelInfo
from lm_lens.server.graph_selection import GraphSelection, UiGraphNode

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(parent_dir, "frontend/build")
    
    # 检查构建目录是否存在
    if not os.path.exists(build_path):
        # 检查是否在 Streamlit Cloud 环境
        is_streamlit_cloud = os.getenv("STREAMLIT_CLOUD") is not None or os.path.exists("/mount/src")
        
        error_msg = (
            f"前端组件构建目录不存在: {build_path}\n\n"
        )
        
        if is_streamlit_cloud:
            error_msg += (
                "在 Streamlit Cloud 上，请确保：\n"
                "1. 将构建产物提交到 Git（推荐）：\n"
                "   cd lm_lens/components/frontend\n"
                "   npm install --legacy-peer-deps\n"
                "   npm run build\n"
                "   git add lm_lens/components/frontend/build/\n"
                "   git commit -m 'Add frontend build'\n"
                "   git push\n"
                "2. 或者确保 .streamlit/packages.sh 能够正确执行构建\n"
            )
        else:
            error_msg += (
                "请在本地运行构建命令：\n"
                "   cd lm_lens/components/frontend\n"
                "   npm install --legacy-peer-deps\n"
                "   npm run build\n"
            )
        
        raise FileNotFoundError(error_msg)
    
    config = {
        "path": build_path,
    }
else:
    config = {
        "url": "http://localhost:3001",
    }

_component_func = components.declare_component("contribution_graph", **config)


def is_node_valid(node: UiGraphNode, n_layers: int, n_tokens: int):
    return node.layer < n_layers and node.token < n_tokens


def is_selection_valid(s: GraphSelection, n_layers: int, n_tokens: int):
    if not s:
        return True
    if s.node:
        if not is_node_valid(s.node, n_layers, n_tokens):
            return False
    if s.edge:
        for node in [s.edge.source, s.edge.target]:
            if not is_node_valid(node, n_layers, n_tokens):
                return False
    return True


def contribution_graph(
    model_info: ModelInfo,
    tokens: List[str],
    graphs: List[nx.Graph],
    key: str,
) -> Optional[GraphSelection]:
    """Create a new instance of contribution graph.

    Returns selected graph node or None if nothing was selected.
    """
    assert len(tokens) == len(graphs)

    # Convert graphs to node-link format, safely handling empty graphs
    edges_per_token = []
    for g in graphs:
        graph_data = nx.node_link_data(g)
        # Safely get links, defaulting to empty list if not present
        links = graph_data.get("links", [])
        edges_per_token.append(links)

    result = _component_func(
        component="graph",
        model_info=model_info.__dict__,
        tokens=tokens,
        edges_per_token=edges_per_token,
        default=None,
        key=key,
    )

    selection = GraphSelection.from_json(result)

    n_tokens = len(tokens)
    n_layers = model_info.n_layers
    # We need this extra protection because even though the component has to check for
    # the validity of the selection, sometimes it allows invalid output. It's some
    # unexpected effect that has something to do with React and how the output value is
    # set for the component.
    if not is_selection_valid(selection, n_layers, n_tokens):
        selection = None

    return selection


def selector(
    items: List[str],
    indices: List[int],
    temperatures: Optional[List[float]],
    preselected_index: Optional[int],
    key: str,
) -> Optional[int]:
    """Create a new instance of selector.

    Returns selected item index.
    """
    n = len(items)
    assert n == len(indices)
    items = [{"index": i, "text": s} for s, i in zip(items, indices)]

    if temperatures is not None:
        assert n == len(temperatures)
        for i, t in enumerate(temperatures):
            items[i]["temperature"] = t

    result = _component_func(
        component="selector",
        items=items,
        preselected_index=preselected_index,
        default=None,
        key=key,
    )

    return None if result is None else int(result)
