"""Optional t-SNE visualization of the vector store . Needs the ``viz`` extra."""

from __future__ import annotations

COLORS = ["blue", "green", "red", "orange", "purple", "brown"]


def build_figure(vectors, documents: list[str], metadatas: list[dict], dims: int = 2):
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.manifold import TSNE

    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    arr = np.array(vectors)
    if len(arr) < 3:
        raise ValueError("Need at least 3 vectors for t-SNE")
    types = [m.get("type", "unknown") for m in metadatas]
    palette = {t: COLORS[i % len(COLORS)] for i, t in enumerate(sorted(set(types)))}
    perplexity = min(30, len(arr) - 1)
    red = TSNE(n_components=dims, random_state=42, perplexity=perplexity).fit_transform(arr)
    text = [f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(types, documents)]
    marker = dict(size=5, color=[palette[t] for t in types], opacity=0.8)
    if dims == 2:
        trace = go.Scatter(x=red[:, 0], y=red[:, 1], mode="markers", marker=marker, text=text, hoverinfo="text")
    else:
        trace = go.Scatter3d(
            x=red[:, 0], y=red[:, 1], z=red[:, 2], mode="markers", marker=marker, text=text, hoverinfo="text"
        )
    fig = go.Figure(data=[trace])
    fig.update_layout(title=f"{dims}D Chroma Vector Store Visualization", width=900, height=700)
    return fig


def visualize(settings, pipeline: str, dims: int = 2, output: str | None = None, show: bool = True):
    from .embeddings import make_embedder
    from .store import VectorStore

    store = VectorStore(settings.db_dir(pipeline), make_embedder(settings.embedding_spec(pipeline)))
    if store.count() == 0:
        raise RuntimeError("Vector store is empty. Run `rag ingest` first.")
    vectors, docs, metas = store.get_all()
    fig = build_figure(vectors, docs, metas, dims)
    if output:
        fig.write_html(output)
        print(f"Wrote {output}")
    if show and not output:
        fig.show()
    return fig
