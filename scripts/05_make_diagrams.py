"""Draw the three block diagrams used in the report: system schematic, workflow, PINC architecture."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plotting import save


def box(ax, x, y, w, h, text, fc="0.92", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text=None, dy=0.12, ls="-", ha="center"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.1, ls=ls))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha=ha, va="bottom", fontsize=7.5)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * h / w)
    ax.axis("off")
    return fig, ax


def schematic():
    fig, ax = canvas(6.5, 3.6)
    box(ax, 0.3, 2.3, 2.0, 1.2, "PEM stack\n(26 cells)", "0.85")
    box(ax, 3.4, 3.4, 1.8, 0.9, "Separator", "0.95")
    box(ax, 3.4, 1.6, 1.8, 0.9, "Hot pump\n$u_1$", "0.95")
    box(ax, 6.2, 2.1, 1.8, 1.5, "Heat\nexchanger", "0.85")
    box(ax, 6.2, 0.3, 1.8, 0.8, "Cold pump\n$u_2$", "0.95")
    box(ax, 8.4, 0.3, 1.4, 0.8, "Chiller", "0.95")
    arrow(ax, (2.3, 3.2), (3.4, 3.8), "$T_{out,ele}$", dy=0.05)
    arrow(ax, (5.2, 3.8), (6.2, 3.3))
    arrow(ax, (6.2, 2.5), (5.2, 2.1))
    arrow(ax, (3.4, 2.0), (2.3, 2.6), "$T_{in,ele}$", dy=-0.55)
    arrow(ax, (9.1, 1.1), (7.6, 2.1), "$T_{in,c}$", dy=0.0, ha="left")
    arrow(ax, (7.1, 2.1), (7.1, 1.1))
    arrow(ax, (8.4, 0.7), (8.0, 0.7))
    ax.text(7.2, 1.55, "$T_{out,c}$", fontsize=7.5, ha="left")
    arrow(ax, (1.3, 4.9), (1.3, 3.5), "current $I$", dy=0.1)
    save(fig, "fig_system_schematic.png")


def workflow():
    fig, ax = canvas(7.0, 2.4)
    names = ["First-principles\nmodel (RK4)", "Simulated\nexperiments\n(steady, dynamic)", "Preprocessing\nand splitting",
             "Training and\nvalidation\n(grids, seeds)", "Test evaluation\n(metrics)", "Selected\nsurrogates\n(Stages 2, 3)"]
    w, gap = 1.35, 0.3
    for i, n in enumerate(names):
        x = 0.1 + i * (w + gap)
        box(ax, x, 1.0, w, 1.4, n, "0.85" if i in (0, 5) else "0.95", fs=7)
        if i < len(names) - 1:
            arrow(ax, (x + w, 1.7), (x + w + gap, 1.7))
    save(fig, "fig_workflow.png")


def pinc():
    fig, ax = canvas(6.5, 3.4)
    box(ax, 0.2, 1.8, 2.2, 1.6, "inputs\n$x_k,\\ u_k,\\ I_k,\\ \\tau$", "0.95")
    box(ax, 3.2, 1.8, 2.4, 1.6, "network $\\mathcal{N}_\\theta$\n4 x 64 tanh", "0.85")
    box(ax, 6.4, 1.8, 3.2, 1.6, "$\\hat{x}(\\tau)=x_k+\\tau\\,\\sigma_\\Delta\\odot\\mathcal{N}_\\theta$", "0.95")
    arrow(ax, (2.4, 2.6), (3.2, 2.6))
    arrow(ax, (5.6, 2.6), (6.4, 2.6))
    arrow(ax, (8.0, 1.8), (8.0, 0.5), "$\\tau=1$:  $\\hat{x}_{k+1}$", dy=0.15, ha="left")
    ax.plot([8.0, 1.3], [0.5, 0.5], "k-", lw=1.1)
    ax.annotate("", xy=(1.3, 1.8), xytext=(1.3, 0.5), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(4.6, 0.55, "feedback: next initial state", fontsize=7.5, ha="center", va="bottom")
    box(ax, 3.2, 4.0, 5.0, 1.0, "physics loss: $\\frac{1}{\\Delta t}\\partial_\\tau\\hat{x}-f(\\hat{x},u_k,I_k)$ (autodiff)", "0.95", fs=7.5)
    arrow(ax, (7.2, 3.4), (7.2, 4.0), ls="--")
    save(fig, "fig_pinc_architecture.png")


if __name__ == "__main__":
    schematic()
    workflow()
    pinc()
