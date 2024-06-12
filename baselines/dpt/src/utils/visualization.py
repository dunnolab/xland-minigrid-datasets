import numpy as np

import logging
import matplotlib

matplotlib.use("Agg")  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def per_episode_in_context(eval_res, name, ylim=None, max_return=None, max_return_eps=None):
    rets = np.vstack([h for h in eval_res.values()])
    means = rets.mean(0)
    stds = rets.std(0)
    x = np.arange(1, rets.shape[1] + 1)

    fig, ax = plt.subplots(dpi=100)
    ax.grid(visible=True)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.plot(x, means)
    ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_ylabel("Return")
    ax.set_xlabel("Episodes In-Context")
    ax.set_title(f"{name}")

    if max_return is not None:
        ax.axhline(
            max_return,
            ls="--",
            color="goldenrod",
            lw=2,
            label=f"optimal_return: {max_return:.2f}",
        )
    if max_return_eps is not None:
        ax.axhline(
            max_return_eps,
            ls="--",
            color="indigo",
            lw=2,
            label=f"max_perf_return: {max_return_eps:.2f}",
        )
    if max_return_eps is not None or max_return is not None:
        plt.legend()

    fig.savefig(f"rets_vs_eps_{name}.png")
    plt.close()

    return f"rets_vs_eps_{name}.png"