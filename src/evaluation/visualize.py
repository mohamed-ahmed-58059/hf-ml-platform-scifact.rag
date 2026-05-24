"""t-SNE visualization of embedding sets.

Runs k-means on the FIRST embedder's vectors to derive cluster labels, then projects
each embedder's vectors to 2D with t-SNE and plots them side-by-side using the same
colors. Lets you eyeball whether different embedders agree on global structure.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from src.model.embedding_set import EmbeddingSet


def visualize_embeddings(
    embedding_sets: list[EmbeddingSet],
    output_path: str,
    n_clusters: int = 8,
    sample_size: int | None = None,
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    """Save a side-by-side t-SNE scatter plot, one subplot per embedder.

    sample_size=None uses the full corpus. Same indices are sampled across all
    embedders so the coloring stays consistent.
    """
    if not embedding_sets:
        raise ValueError("embedding_sets is empty")

    n_total = len(embedding_sets[0].embedded_chunks)
    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size < n_total:
        idx = rng.choice(n_total, size=sample_size, replace=False)
    else:
        idx = np.arange(n_total)

    # Stack each embedder's vectors into a (N, dim) matrix, in the same row order.
    matrices = [
        np.stack([s.embedded_chunks[i].vector for i in idx])
        for s in embedding_sets
    ]

    # Cluster on the first embedder's vectors — those labels color every subplot.
    print(f"[visualize] KMeans (k={n_clusters}) on '{embedding_sets[0].model_id}'...")
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(matrices[0])

    fig, axes = plt.subplots(1, len(embedding_sets), figsize=(6 * len(embedding_sets), 6))
    if len(embedding_sets) == 1:
        axes = [axes]

    for ax, mat, s in zip(axes, matrices, embedding_sets):
        print(f"[visualize] t-SNE projecting '{s.model_id}' ({mat.shape[0]} vectors)...")
        proj = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            n_jobs=-1,
        ).fit_transform(mat)
        ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7)
        ax.set_title(s.model_id, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"t-SNE of embedding spaces (colored by KMeans on '{embedding_sets[0].model_id}')")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved {output_path}")
