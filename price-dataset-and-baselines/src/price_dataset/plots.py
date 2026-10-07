"""Optional Plotly charts (HTML output). Requires the ``plots`` extra."""

from __future__ import annotations

from price_dataset.evaluator import running_mean_ci


def scatter_chart(tester, path: str) -> str:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.DataFrame(
        {
            "truth": tester.truths,
            "guess": tester.guesses,
            "title": tester.titles,
            "color": tester.colors,
        }
    )
    df["hover"] = [f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}" for t, g, y in zip(df.title, df.guess, df.truth)]
    max_val = float(max(df.truth.max(), df.guess.max()))
    m = tester.metrics
    title = (
        f"{tester.title} results<br><b>Error:</b> ${m.average_error:,.2f} <b>MSE:</b> {m.mse:,.0f} "
        f"<b>r²:</b> {m.r2_percent:.1f}% <b>hit rate:</b> {m.hit_rate:.0%}"
    )
    fig = px.scatter(
        df,
        x="truth",
        y="guess",
        color="color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=title,
        labels={"truth": "Actual Price", "guess": "Predicted Price"},
        width=1000,
        height=800,
    )
    for tr in fig.data:
        mask = df["color"] == tr.name
        tr.customdata = df.loc[mask, ["hover"]].to_numpy()
        tr.hovertemplate = "%{customdata[0]}<extra></extra>"
        tr.marker.update(size=6)
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                   line=dict(width=2, dash="dash", color="deepskyblue"), hoverinfo="skip", showlegend=False)
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    fig.update_layout(showlegend=False)
    fig.write_html(path)
    return path


def error_trend_chart(tester, path: str) -> str:
    import plotly.graph_objects as go

    means, ci = running_mean_ci(tester.errors)
    x = list(range(1, len(means) + 1))
    upper = [m + c for m, c in zip(means, ci)]
    lower = [m - c for m, c in zip(means, ci)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=upper + lower[::-1], fill="toself",
                             fillcolor="rgba(128,128,128,0.2)", line=dict(color="rgba(255,255,255,0)"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=means, mode="lines", line=dict(width=3, color="firebrick"),
                             name="Cumulative Avg Error"))
    fig.update_layout(
        title=f"{tester.title} Error: ${means[-1]:,.2f} ± ${ci[-1]:,.2f}",
        xaxis_title="Number of Datapoints", yaxis_title="Average Absolute Error ($)",
        width=1000, height=360, template="plotly_white", showlegend=False,
    )
    fig.write_html(path)
    return path
