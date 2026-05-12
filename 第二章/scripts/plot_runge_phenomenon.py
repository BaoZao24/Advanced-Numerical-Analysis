from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def runge(x):
    return 1 / (1 + 25 * x**2)


def barycentric_interpolate(nodes, values, x):
    """Evaluate the interpolation polynomial through nodes using barycentric form."""
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)

    weights = np.ones_like(nodes)
    for j in range(len(nodes)):
        diff = nodes[j] - np.delete(nodes, j)
        weights[j] = 1 / np.prod(diff)

    y = np.empty_like(x)
    for idx, x_value in enumerate(x):
        hit = np.isclose(x_value, nodes, rtol=0, atol=1e-14)
        if np.any(hit):
            y[idx] = values[hit][0]
        else:
            terms = weights / (x_value - nodes)
            y[idx] = np.sum(terms * values) / np.sum(terms)
    return y


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "assets"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "2.12-runge-phenomenon.png"

    x = np.linspace(-1, 1, 1200)
    y_true = runge(x)

    nodes_low = np.linspace(-1, 1, 6)
    values_low = runge(nodes_low)
    y_low = barycentric_interpolate(nodes_low, values_low, x)

    nodes_high = np.linspace(-1, 1, 16)
    values_high = runge(nodes_high)
    y_high = barycentric_interpolate(nodes_high, values_high, x)

    y_piecewise = np.interp(x, nodes_high, values_high)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180, constrained_layout=True)

    ax = axes[0]
    ax.plot(x, y_true, color="#111111", lw=2.6, label="Original Runge function")
    ax.plot(x, y_low, color="#2563eb", lw=1.9, label="Global interpolation, 6 equally spaced nodes")
    ax.plot(x, y_high, color="#dc2626", lw=1.9, label="Global interpolation, 16 equally spaced nodes")
    ax.scatter(nodes_high, values_high, s=24, color="#dc2626", zorder=5, label="16 interpolation nodes")
    ax.axvspan(-1, -0.72, color="#fee2e2", alpha=0.8)
    ax.axvspan(0.72, 1, color="#fee2e2", alpha=0.8)
    ax.annotate(
        "More nodes,\nbut bigger edge wiggles",
        xy=(0.88, y_high[np.searchsorted(x, 0.88)]),
        xytext=(0.36, 1.55),
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.4),
        color="#991b1b",
        fontsize=10,
    )
    ax.set_title("Runge phenomenon: one high-degree curve tries to pass every point")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 2.1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", fontsize=8)

    ax = axes[1]
    ax.plot(x, y_true, color="#111111", lw=2.6, label="Original Runge function")
    ax.plot(x, y_piecewise, color="#16a34a", lw=2.0, label="Piecewise low-degree idea")
    ax.scatter(nodes_high, values_high, s=24, color="#16a34a", zorder=5, label="Local nodes")
    for node in nodes_high:
        ax.axvline(node, color="#bbf7d0", lw=0.7, alpha=0.8)
    ax.annotate(
        "Each small interval\nonly handles nearby points",
        xy=(0.54, runge(0.54)),
        xytext=(-0.15, 1.45),
        arrowprops=dict(arrowstyle="->", color="#166534", lw=1.4),
        color="#166534",
        fontsize=10,
    )
    ax.set_title("Why piecewise interpolation is calmer")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 1.75)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", fontsize=8)

    fig.suptitle("Runge Function  f(x)=1/(1+25x^2)", fontsize=14, fontweight="bold")
    fig.savefig(out_file, bbox_inches="tight")
    print(out_file)


if __name__ == "__main__":
    main()
