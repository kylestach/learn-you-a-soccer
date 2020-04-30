import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plots_dir = "plots"
results_dir = "results/comparison"
eval_freq = 5e4
x_units = 1e6


def set_plot_defaults() -> None:
    sns.set()
    sns.set_context("paper")
    plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')


def plot_comparison(linear_f: np.ndarray, constant_f: np.ndarray, linear_c: np.ndarray,
                    title: str, name: str):
    """
    @param linear_f: (num_linears, num_evals)
    @param constant_f: (num_constants, num_evals)
    @param linear_c: (num_linears, num_evals)
    @param name: Name to use for plot
    @param title: Title of the plot
    @return:
    """
    mean_linear_f = np.mean(linear_f, axis=0)
    mean_constant_f = np.mean(constant_f, axis=0)
    quantiles_linear_f = np.quantile(linear_f, [0.25, 0.75], axis=0)
    quantiles_constant_f = np.quantile(constant_f, [0.25, 0.75], axis=0)
    mean_linear_c = np.mean(linear_c, axis=0)
    quantiles_linear_c = np.quantile(linear_c, [0.25, 0.75], axis=0)

    num_evals = mean_linear_f.shape[0]
    xs = np.arange(num_evals) * eval_freq / x_units

    alpha = 0.2

    mean_linear_line_f, = plt.plot(xs, mean_linear_f)
    mean_linear_line_f.set_label("Linear curriculum (out-of-curriculum)")
    plt.fill_between(xs, quantiles_linear_f[0], quantiles_linear_f[1], alpha=alpha)

    mean_linear_line_c, = plt.plot(xs, mean_linear_c)
    mean_linear_line_c.set_label("Linear curriculum (in-curriculum)")
    plt.fill_between(xs, quantiles_linear_c[0], quantiles_linear_c[1], alpha=alpha)

    mean_constant_line, = plt.plot(xs, mean_constant_f)
    mean_constant_line.set_label("No curriculum")
    plt.fill_between(xs, quantiles_constant_f[0], quantiles_constant_f[1], alpha=alpha)

    plt.gca().set_title(f'robocup-{title}-v0')
    plt.xlabel('Interactions in millions')
    plt.ylabel('Reward')


def create_plots_dir():
    Path(plots_dir).mkdir(parents=True, exist_ok=True)


def plot_graphs(env_name: str):
    linear_data_final = []
    linear_data_curriculum = []

    constant_data_final = []
    constant_data_curriculum = []
    for filename in os.listdir(results_dir):
        if env_name not in filename:
            continue
        # (n,)
        data = np.load(f"{results_dir}/{filename}")

        if "curriculum" in filename and "linear" in filename:
            linear_data_curriculum.append(data)
        elif "final" in filename and "linear" in filename:
            linear_data_final.append(data)
        elif "curriculum" in filename and "constant" in filename:
            constant_data_curriculum.append(data)
        elif "final" in filename and "constant" in filename:
            constant_data_final.append(data)
        else:
            print(f"Unknown type: {filename}")

    if len(linear_data_final) == 0:
        print(f"Couldn't find any data for env {env_name}")
        return

    linear_data_final = np.stack(linear_data_final)
    linear_data_curriculum = np.stack(linear_data_curriculum)
    constant_data_final = np.stack(constant_data_final)
    constant_data_curriculum = np.stack(constant_data_curriculum)

    if "score" in env_name:
        mult1 = 120
        mult2 = 80
    else:
        mult1 = 1
        mult2 = 1

    linear_data_final *= mult1
    constant_data_final *= mult1
    linear_data_curriculum *= mult2
    constant_data_curriculum *= mult2

    # Plot finals
    plot_comparison(linear_data_final, constant_data_final, linear_data_curriculum, env_name, f"{env_name}.png")


def main():
    set_plot_defaults()
    create_plots_dir()

    plt.figure(figsize=(8, 12), dpi=500)
    plt.subplot(311)
    print(f"Plotting collect...")
    plot_graphs("collect")
    plt.subplot(312)
    print(f"Plotting score...")
    plot_graphs("score")
    plt.subplot(313)
    print(f"Plotting pass...")
    plot_graphs("pass")

    plt.tight_layout()
    lgd = plt.figlegend(("Linear (out-of-curriculum)", "Linear (in-curriculum)", "No curriculum"),
                        loc='upper center', borderaxespad=0.,
                        bbox_to_anchor=(0.5, 0.015),
                        ncol=3
                        )
    plt.savefig(f"{plots_dir}/plots.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    main()
