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


def plot_comparison(linear: np.ndarray, constant: np.ndarray, name: str):
    """
    @param linear: (num_linears, num_evals)
    @param constant: (num_constants, num_evals)
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

    plt.title('robocup-score-v1')
    plt.xlabel('Interactions in millions')
    plt.ylabel('Reward')
    plt.legend(loc="upper left")
    plt.savefig(f"{plots_dir}/{name}")


def create_plots_dir():
    Path(plots_dir).mkdir(parents=True, exist_ok=True)


def main():
    set_plot_defaults()
    create_plots_dir()

    linear_data_final = []
    linear_data_curriculum = []

    constant_data_final = []
    constant_data_curriculum = []
    for filename in os.listdir(results_dir):
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

    linear_data_final = np.stack(linear_data_final)
    linear_data_curriculum = np.stack(linear_data_curriculum)
    constant_data_final = np.stack(constant_data_final)
    constant_data_curriculum = np.stack(constant_data_curriculum)

    # Plot finals
    plot_comparison(linear_data_final, constant_data_final, "score_final_comparison.png")
    #
    # # Plot curriculum
    # plot_comparison(linear_data_curriculum, constant_data_curriculum)


if __name__ == '__main__':
    main()
