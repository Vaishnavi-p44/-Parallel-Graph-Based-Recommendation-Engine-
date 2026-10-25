"""
Visualization module using Matplotlib and NetworkX.
Generates publication-quality benchmark performance plots (Speedup, Execution Time, Efficiency)
and serializes them to base64 images for seamless embedding in web views and reports.
"""

import io
import base64
from typing import Dict, Any, List
import matplotlib
# Use Agg backend for headless server rendering without GUI windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_benchmark_chart_base64(benchmark_data: Dict[str, Any]) -> str:
    """
    Renders a multi-panel Matplotlib dashboard chart from benchmark results:
    1. Execution Time vs Worker Processes
    2. Speedup Curve vs Ideal Linear Speedup
    3. Parallel Efficiency (%)
    Returns:
        Base64 encoded PNG string.
    """
    records = benchmark_data.get("benchmark_results", [])
    if not records:
        return ""

    workers = [r["workers"] for r in records]
    times = [r["time_sec"] for r in records]
    speedups = [r["speedup"] for r in records]
    efficiencies = [r["efficiency_pct"] for r in records]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)
    fig.patch.set_facecolor("#0f172a") # Dark Slate Theme

    # Colors
    accent_blue = "#38bdf8"
    accent_emerald = "#34d399"
    accent_purple = "#a855f7"
    text_color = "#f8fafc"
    grid_color = "#334155"

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors=text_color, labelsize=9)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, linestyle="--", alpha=0.3, color=grid_color)

    # Panel 1: Execution Time
    ax1 = axes[0]
    ax1.plot(workers, times, marker="o", linewidth=2.5, markersize=8, color=accent_blue, label="Execution Time")
    ax1.axhline(times[0], color="#f43f5e", linestyle=":", linewidth=1.8, label=f"Seq Baseline ({times[0]:.3f}s)")
    ax1.set_title("Execution Time (Lower is Better)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Worker Processes (P)", fontsize=10)
    ax1.set_ylabel("Time (Seconds)", fontsize=10)
    ax1.set_xticks(workers)
    ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=8)

    # Panel 2: Speedup vs Ideal
    ax2 = axes[1]
    ax2.plot(workers, speedups, marker="s", linewidth=2.5, markersize=8, color=accent_emerald, label="Observed Speedup")
    ideal_speedup = [w for w in workers]
    ax2.plot(workers, ideal_speedup, linestyle="--", linewidth=1.8, color="#94a3b8", label="Ideal Linear Speedup (S = P)")
    ax2.set_title("Speedup S(P) = T_seq / T_par", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Worker Processes (P)", fontsize=10)
    ax2.set_ylabel("Speedup Factor", fontsize=10)
    ax2.set_xticks(workers)
    ax2.legend(loc="upper left", facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=8)

    # Panel 3: Parallel Efficiency (%)
    ax3 = axes[2]
    bars = ax3.bar([str(w) for w in workers], efficiencies, color=accent_purple, width=0.55, edgecolor="#c084fc", linewidth=1.2)
    ax3.axhline(100.0, color="#f59e0b", linestyle="--", linewidth=1.5, label="100% Ideal Efficiency")
    ax3.set_title("Parallel Efficiency E(P) = S / P", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Worker Processes (P)", fontsize=10)
    ax3.set_ylabel("Efficiency (%)", fontsize=10)
    ax3.set_ylim(0, max(max(efficiencies) * 1.25, 115))
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=8)

    # Add value annotations on bars
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom",
                     color=text_color, fontsize=8, fontweight="bold")

    fig.suptitle(
        f"Distributed Graph Engine Scaling Benchmark ({benchmark_data.get('algorithm', '').replace('_', ' ').title()}) "
        f"— Batch Size: {benchmark_data.get('user_batch_size')} Users",
        fontsize=13, fontweight="bold", color=text_color, y=1.02
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return b64_str
