import cv2
import numpy as np
import torch_geometric
import os.path as osp
import os
import pandas as pd
import seaborn as sns
import networkx as nx

from torch_geometric.datasets import MNISTSuperpixels
from typing import Tuple
from matplotlib import pyplot as plt
from PIL import ImageColor


def superpixels_to_2d_image(rec: torch_geometric.data.Data, scale: int = 30, edge_width: int = 1) -> np.ndarray:
    pos = (rec.pos.clone() * scale).int()

    image = np.zeros((scale * 26, scale * 26, 1), dtype=np.uint8)
    for (color, (x, y)) in zip(rec.x, pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale

        color = int(float(color + 0.15) * 255)
        color = min(color, 255)

        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    for node_ix_0, node_ix_1 in rec.edge_index.T:
        x0, y0 = list(map(int, pos[node_ix_0]))
        x1, y1 = list(map(int, pos[node_ix_1]))

        x0 -= scale // 2
        y0 -= scale // 2
        x1 -= scale // 2
        y1 -= scale // 2

        cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)
    return image


def visualize_first(
    dataset: MNISTSuperpixels,
    image_name: str,
    examples_per_class: int = 10,
    classes: Tuple[int, ...] = tuple(range(10)),
    figsize: Tuple[int, int] = (50, 50),
    edge_width: int = 1,
) -> None:
    class_to_examples = {class_ix: [] for class_ix in classes}

    for record in dataset:
        enough = True
        for examples in class_to_examples.values():
            if len(examples) < examples_per_class:
                enough = False
        if enough:
            break

        class_ix = int(record.y)

        if class_ix not in class_to_examples:
            continue

        if len(class_to_examples[class_ix]) == examples_per_class:
            continue

        if len(class_to_examples[class_ix]) > examples_per_class:
            raise RuntimeError

        class_to_examples[class_ix].append(superpixels_to_2d_image(record, edge_width=edge_width))

    plt.figure(figsize=figsize)
    subplot_ix = 1
    for class_ix in classes:
        for example in class_to_examples[class_ix]:
            plt.subplot(len(classes), examples_per_class, subplot_ix)
            subplot_ix += 1
            plt.imshow(example, cmap=plt.cm.binary)
    plt.savefig(image_name)

def visualize(image, data):
    plt.figure(figsize=(14, 8))
    
    # plot the mnist image
    plt.subplot(1, 2, 1)
    plt.title("MNIST")
    np_image = np.array(image)
    plt.imshow(np_image)
    
    # plot the super-pixel graph
    plt.subplot(1, 2, 2)
    x, edge_index = data.x, data.edge_index
    
    # construct networkx graph
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')    
    
    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([data.pos[i][0], 27 - data.pos[i][1]]) for i in range(data.num_nodes)}
    
    # get the current node index of G
    idx = list(G.nodes())

    # set the node sizes using node features
    size = x[idx] * 500 + 200
    
    # set the node colors using node features
    color = []
    for i in idx:
        grey = x[i]
        if grey == 0:
            color.append('skyblue')
        else:
            color.append('red')
    
    nx.draw(G, with_labels=True, node_size=size, node_color=color, pos=pos)
    plt.title("MNIST Superpixel")

"""
if __name__ == "__main__":
    train_dataset = build_mnist_superpixels_dataset(train=True)
    test_dataset = build_mnist_superpixels_dataset(train=False)

    visualize_first(
        train_dataset,
        image_name="all_classes.jpg",
    )

    visualize_first(
        train_dataset,
        image_name="one_class.jpg",
        classes=(8,),
        figsize=(10,10),
        examples_per_class=1,
    )
"""