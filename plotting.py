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
    plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')


def plot_comparison(linear: np.ndarray, constant: np.ndarray, title: str, name: str):
    """
    @param linear: (num_linears, num_evals)
    @param constant: (num_constants, num_evals)
    @param name: Name to use for plot
    @return:
    """
    mean_linear = np.mean(linear, axis=0)
    mean_constant = np.mean(constant, axis=0)
    quantiles_linear = np.quantile(linear, [0.25, 0.75], axis=0)
    quantiles_constant = np.quantile(constant, [0.25, 0.75], axis=0)

    num_evals = mean_linear.shape[0]
    xs = np.arange(num_evals) * eval_freq / x_units

    alpha = 0.2

    plt.figure(figsize=(8, 4))
    mean_linear_line, = plt.plot(xs, mean_linear)
    mean_linear_line.set_label("Linear curriculum")
    plt.fill_between(xs, quantiles_linear[0], quantiles_linear[1], alpha=alpha)

    mean_constant_line, = plt.plot(xs, mean_constant)
    mean_constant_line.set_label("No curriculum")
    plt.fill_between(xs, quantiles_constant[0], quantiles_constant[1], alpha=alpha)

    plt.title(f'robocup-{title}-v0')
    plt.xlabel('Interactions in millions')
    plt.ylabel('Reward')
    plt.legend(loc="upper left")
    plt.savefig(f"{plots_dir}/{name}")


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
    plot_comparison(linear_data_final, constant_data_final, env_name, f"{env_name}_final_comparison.png")

    # Plot curriculum
    plot_comparison(linear_data_curriculum, constant_data_curriculum, env_name, f"{env_name}_curriculum_comparison.png")


def main():
    set_plot_defaults()
    create_plots_dir()

    for env_name in ["collect", "score", "pass"]:
        plot_graphs(env_name)


if __name__ == '__main__':
    main()
