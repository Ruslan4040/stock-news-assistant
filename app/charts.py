"""Plotly figures for the UI: one palette for light and dark mode (display code only)."""

from datetime import date

import plotly.graph_objects as go

# Reference palette of the dataviz skill; the two series colors pass its validator in both modes.
PALETTE = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926"]},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LINE_WIDTH = 2
MARKER_SIZE = 9
LABEL_GAP = 0.06  # share of the y range below which two end labels would collide


def _layout(figure: go.Figure, colors: dict, title: str, money: str) -> go.Figure:
    """Apply the shared chrome: quiet grid, one axis, a crosshair that lists every series."""
    figure.update_layout(
        title={"text": title, "font": {"size": 15, "color": colors["text"]}},
        font={"family": FONT, "color": colors["secondary"]},
        paper_bgcolor=colors["surface"], plot_bgcolor=colors["surface"],
        hovermode="x unified", height=360,
        margin={"l": 70, "r": 90, "t": 48, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": colors["secondary"]}},
    )
    ticks = {"color": colors["muted"]}
    figure.update_xaxes(showgrid=False, linecolor=colors["axis"], tickfont=ticks, automargin=True,
                        showspikes=True, spikemode="across", spikethickness=1,
                        spikedash="solid", spikecolor=colors["axis"])
    figure.update_yaxes(gridcolor=colors["grid"], zeroline=False, tickprefix=money,
                        tickformat=",.0f", tickfont=ticks, automargin=True)
    return figure


def _line(figure: go.Figure, days: list, values: list, name: str, color: str, surface: str) -> None:
    """Add a 2px line with a ringed end marker."""
    hover = "%{y:$,.2f}<extra>" + name + "</extra>"
    figure.add_trace(go.Scatter(x=days, y=values, name=name, mode="lines", hovertemplate=hover,
                                line={"color": color, "width": LINE_WIDTH}))
    ring = {"color": surface, "width": 2}
    figure.add_trace(go.Scatter(x=[days[-1]], y=[values[-1]], mode="markers", showlegend=False,
                                hoverinfo="skip",
                                marker={"color": color, "size": MARKER_SIZE, "line": ring}))


def _end_label(figure: go.Figure, day: object, value: float, text: str, colors: dict) -> None:
    """Label the end of a line in text ink, to the right of the plot area."""
    figure.add_annotation(x=day, y=value, text=text, showarrow=False, xanchor="left", xshift=10,
                          font={"color": colors["text"], "size": 12})


def price_chart(days: list[date], closes: list[float], title: str, dark: bool) -> go.Figure:
    """Adjusted close up to the decision day; the last point is the decision close."""
    colors = PALETTE["dark" if dark else "light"]
    figure = go.Figure()
    _line(figure, days, closes, "Adjusted close", colors["series"][0], colors["surface"])
    _end_label(figure, days[-1], closes[-1], f"${closes[-1]:,.2f}", colors)
    figure.update_layout(showlegend=False)
    return _layout(figure, colors, title, "$")


def equity_chart(
    days: list[str], strategy: list[float], baseline: list[float], dark: bool
) -> go.Figure:
    """Simulator equity against buy & hold, on one dollar axis."""
    colors = PALETTE["dark" if dark else "light"]
    figure = go.Figure()
    _line(figure, days, strategy, "Simulator", colors["series"][0], colors["surface"])
    _line(figure, days, baseline, "Buy & hold", colors["series"][1], colors["surface"])
    span = max(strategy + baseline) - min(strategy + baseline)
    if span and abs(strategy[-1] - baseline[-1]) / span > LABEL_GAP:
        _end_label(figure, days[-1], strategy[-1], f"${strategy[-1]:,.0f}", colors)
        _end_label(figure, days[-1], baseline[-1], f"${baseline[-1]:,.0f}", colors)
    return _layout(figure, colors, "Equity: simulator vs buy & hold", "$")
