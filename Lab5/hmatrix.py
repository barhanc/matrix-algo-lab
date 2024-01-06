import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import svd
from numpy.linalg import norm
from time import perf_counter
from sklearn.utils.extmath import randomized_svd


class Node:
    def __init__(self, rank=None, side=None, sMin=None, tMin=None, U=None, V=None, D=None):
        self.next: list[Node] = []

        self.sMin: int = sMin
        self.tMin: int = tMin
        self.side: int = side

        self.rank: int = rank
        self.U: np.ndarray[float] = U
        self.D: np.ndarray[float] = D
        self.V: np.ndarray[float] = V


def CompressMatrix(
    A: np.ndarray[float],
    U: np.ndarray[float],
    D: np.ndarray[float],
    V: np.ndarray[float],
    r: int,
    sMin: int,
    tMin: int,
):
    if np.allclose(A, np.zeros(A.shape)):
        return Node(rank=0, side=A.shape[0], sMin=sMin, tMin=tMin)

    return Node(
        rank=r,
        side=A.shape[0],
        sMin=sMin,
        tMin=tMin,
        U=U[:, :r],
        D=D[:r],
        V=V[:r, :],
    )


def CreateTree(A: np.ndarray[float], r: int, eps: float, sMin: int = 0, tMin: int = 0):
    n = A.shape[0]
    U, D, V = randomized_svd(A, r + 1)
    if len(D) <= r or D[r] < eps:
        v = CompressMatrix(A, U, D, V, len(D), sMin, tMin)
    else:
        v = Node(rank=None, side=n, sMin=sMin, tMin=tMin)
        v.next.append(CreateTree(A[: n // 2, : n // 2], r, eps, sMin, tMin))
        v.next.append(CreateTree(A[: n // 2, n // 2 :], r, eps, sMin, tMin + n // 2))
        v.next.append(CreateTree(A[n // 2 :, : n // 2], r, eps, sMin + n // 2, tMin))
        v.next.append(CreateTree(A[n // 2 :, n // 2 :], r, eps, sMin + n // 2, tMin + n // 2))
    return v


def DrawTree(root: Node):
    n = root.side
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_title(r"$n=$" + f"{n}")

    def DrawTreeRecursive(root: Node):
        nonlocal ax

        if root.rank is None:  # Not a leaf -> draw grid lines
            x, y, d = root.sMin, root.tMin, root.side // 2
            ax.plot((x, x + 2 * d), (y + d, y + d), color="k", lw=0.6)
            ax.plot((x + d, x + d), (y, y + 2 * d), color="k", lw=0.6)

        elif root.rank > 0:  # Leaf with SVD decomposition -> fill block
            x, y, d = (root.sMin, root.tMin + root.side, root.side)
            lw = root.side / (6 * root.rank)
            for i in range(root.rank):
                ax.fill([x, x + d, x + d, x], [y, y, y - (i + 1) * lw, y - (i + 1) * lw], color="k")
                ax.fill([x, x + (i + 1) * lw, x + (i + 1) * lw, x], [y, y, y - d, y - d], color="k")

        elif root.rank == 0:  # Leaf with all 0s -> pass
            pass

        for node in root.next:
            DrawTreeRecursive(node)

    DrawTreeRecursive(root)
    return fig


def Decompress(root: Node):
    n = root.side
    A = np.zeros((n, n))

    def DecompressRecursive(root: Node):
        nonlocal A
        if root.rank is None:  # Not a leaf -> pass
            pass
        elif root.rank == 0:  # Leaf with all 0s -> pass
            pass
        elif root.rank > 0:
            s, t, side = root.sMin, root.tMin, root.side
            U, D, V = root.U, np.diag(root.D), root.V
            A[s : s + side, t : t + side] = U @ D @ V

        for node in root.next:
            DecompressRecursive(node)

    DecompressRecursive(root)
    return A
