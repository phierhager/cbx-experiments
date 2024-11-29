import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn.objects as so
import typing
from scipy.stats import linregress
from scipy.optimize import curve_fit

def plot_bar_plot_comparing_dynamics_to_rel_value(df: pd.DataFrame) -> None:
    """This function plots a bar plot comparing the dynamics to the relative function value."""
    dfm = df.melt(
        id_vars=["Function Name", "Dynamic Name"],
        value_vars="Relative Function Value",
        var_name="variable",
        value_name="value",
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=dfm, x="Function Name", y="value", hue="Dynamic Name")
    plt.title(
        f"Dynamics on different Functions (M={df.iloc[0].M}, N={df.iloc[0].N}, d={df.iloc[0].d})"
    )
    plt.xlabel("Function Name")
    plt.ylabel("Relative Function Value (1 is best)")
    plt.show()


def plot_N(df: pd.DataFrame, plot_dir: str = "../results/plots/") -> None:
    df = df.explode("Success")
    dfm = df.melt(
        id_vars=["Dynamic Name", "Function Name", "N"],
        value_vars=["Success"],
        var_name="variable",
        value_name="success",
    )
    sns.set_theme(style="white")
    sns.lineplot(
        data=dfm, x="N", y="success", hue="Dynamic Name", marker="o", errorbar="ci"
    )
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    sigma = df.iloc[0].loc["sigma"]
    plt.title(
        "Success Probability for d={}, max_it={}, alpha={}, dt={}, sigma={}, and {}".format(
            d, max_it, alpha, dt, sigma, f
        )
    )
    plt.xlabel("N")
    plt.ylabel("Success Probability")
    plt.ylim(bottom=0)
    plt.savefig(f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__plot_n.png")


def plot_sigma(df: pd.DataFrame, plot_dir: str = "../results/plots/") -> None:
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "Function Name", "sigma"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )
    sns.set_theme(style="white")
    sns.lineplot(data=dfm, x="sigma", y="success_prob", hue="Dynamic Name", marker="o")
    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    N = df.iloc[0].loc["N"]
    plt.title(
        "Success Probability for d={}, max_it={}, alpha={}, dt={}, N={}, and {}".format(
            d, max_it, alpha, dt, N, f
        )
    )
    plt.xlabel("sigma")
    plt.ylabel("Success Probability")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__plot_sigma.png")


def plot_2d_sigma_lamda(df: pd.DataFrame) -> None:
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "lamda", "Function Name"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )
    sns.set_theme(style="white")

    sigma_values = sorted(dfm["sigma"].unique())
    lamda_values = sorted(dfm["lamda"].unique())
    dynamic_names = sorted(dfm["Dynamic Name"].unique())
    function_names = sorted(dfm["Function Name"].unique())
    success_prob_matrix = np.zeros(
        (len(dynamic_names), len(function_names), len(lamda_values), len(sigma_values))
    )

    # Fill the success probability matrix
    for index, row in dfm.iterrows():
        sigma_index = sigma_values.index(row["sigma"])
        lamda_index = lamda_values.index(row["lamda"])
        dynamic_index = dynamic_names.index(row["Dynamic Name"])
        function_index = function_names.index(row["Function Name"])
        success_prob_matrix[dynamic_index, function_index, lamda_index, sigma_index] = (
            row["success_prob"]
        )

    # Determine the number of dynamic names
    num_dynamic_names = len(dynamic_names)
    num_function_names = len(function_names)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Success Probabilities for different Dynamics")

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=num_dynamic_names, ncols=1)
    if num_dynamic_names == 1:
        subfigs = [subfigs]

    # fill subfigs
    for row_index, subfig in enumerate(subfigs):
        dynamic_name = dynamic_names[row_index]
        dynamic_index = dynamic_names.index(dynamic_name)
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_function_names, sharey=True)
        if num_function_names == 1:
            axs = [axs]
        for col_index, ax in enumerate(axs):
            ax.plot()
            ax.set_xlabel("Sigma (\u03c3)")
            ax.set_ylabel("Lambda (\u03bb)")

            function_name = function_names[col_index]
            ax.set_title(f"Function {function_name}")
            function_index = function_names.index(function_name)

            # Get the success probability matrix for the current dynamic name
            success_prob_matrix_dynamic = success_prob_matrix[
                dynamic_index, function_index
            ]

            # Plot the color map in the appropriate subplot
            im = ax.imshow(
                success_prob_matrix_dynamic,
                cmap="viridis",
                origin="lower",
                extent=[0, len(sigma_values), 0, len(lamda_values)],
            )
        subfig.colorbar(im, shrink=0.6, ax=axs)
        subfig.suptitle(f"Dynamic {dynamic_name}")
    plt.show()


def plot_bar_plot_comparing_dynamics_to_time(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="Dynamic Name", y="Best Function Value", hue="Experiment Name"
    )
    plt.title("Comparison of Experiment Results")
    plt.xlabel("Dynamic Name")
    plt.ylabel("Best Function Value")
    plt.show()


def fit_gaussian_exp(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    # Define the fitting function
    def fit_func(x, A, mu, sigma1, B, sigma2):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma1**2)) + B * np.exp(-x / sigma2)

    # Initial guesses for the parameters
    initial_guesses = [1, 5, 1, 0.5, 10]

    # Perform the curve fitting
    popt, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guesses)

    # Get the fitted values
    fitted_y: np.ndarray = fit_func(x_data, *popt)

    return fitted_y


def plot_sigma_N_success(df: pd.DataFrame) -> None:
    """Plot the success probability against sigma for different N values.

    Args:
        df (pd.DataFrame): The DataFrame containing the experiment results.
    """
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )

    # Plot the original data
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    so.Plot(dfm, x="sigma", y="success_prob", color="Dynamic Name").add(
        so.Dots(), pointsize="N", marker="Function Name"
    ).show()


def plot_sigma_N_success_heat(df: pd.DataFrame, dynamic_name: str = "cbo") -> None:
    """Plot the success probability against sigma for different N values."""
    df["Success Probability"] = df["Success"].apply(np.mean)
    df = df[df["Dynamic Name"] == dynamic_name]
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="N", columns="sigma", values="success_prob")

    # Plot the original data
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    sns.heatmap(dfm)  # , annot=True, cmap="viridis", fmt=".2f")
    plt.title(
        f"Success Probability for {dynamic_name} and Ackley at d=2, max_it=100, alpha=100, dt=0.1"
    )
    plt.show()


def plot_sigma_lamda_success_heat_dynamics(
    df: pd.DataFrame, plot_dir: str = "../results/plots/"
) -> None:
    """Plot the success probability against sigma for different dynamics."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]

    # Plot the data side by side
    dynamic_names = set(df["Dynamic Name"])
    sns.set_theme(style="whitegrid")
    _, axes = plt.subplots(1, len(dynamic_names), figsize=(20, 7), sharey=True)

    for i, dynamic_name in enumerate(dynamic_names):
        df_dynamic = df[df["Dynamic Name"] == dynamic_name]
        dfm = df_dynamic.melt(
            id_vars=["Dynamic Name", "sigma", "Function Name", "lamda"],
            value_vars=["Success Probability"],
            var_name="variable",
            value_name="success_prob",
        ).pivot(index="lamda", columns="sigma", values="success_prob")
        ax = (
            axes[i] if isinstance(axes, np.ndarray) else axes
        )  # TODO: Really np ndarrayß
        sns.heatmap(dfm, ax=ax, cbar=True)

        ax.plot(
            x := np.linspace(0, 20, 100),
            (a := 2.1) * x,
            color="green",
            linewidth=2,
            label=f"Exponential Function, factor {a}",
        )
        ax.plot
        ax.set_title(
            f"Success Probability for {dynamic_name} and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}"
        )
        ax.invert_yaxis()
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__sigma_lamda_success_heat_dynamics.png"
    )


def plot_sigma_N_success_heat_dynamics(
    df: pd.DataFrame, plot_dir: str = "../results/plots/"
) -> None:
    """Plot the success probability against sigma for different dynamics."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # Plot the data side by side
    sns.set_theme(style="whitegrid")

    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    lamda = df.iloc[0].loc["lamda"]

    dynamic_names = set(df["Dynamic Name"])
    _, axes = plt.subplots(1, len(dynamic_names), figsize=(20, 7), sharey=True)
    if len(dynamic_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, dynamic_name in enumerate(dynamic_names):
        df_dynamic = df[df["Dynamic Name"] == dynamic_name]
        dfm = df_dynamic.melt(
            id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
            value_vars=["Success Probability"],
            var_name="variable",
            value_name="success_prob",
        ).pivot(index="N", columns="sigma", values="success_prob")
        sns.heatmap(dfm, ax=axes[i], cbar=True)
        axes[i].set_title(
            f"Success Probability for {dynamic_name} and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}, lamda={lamda}"
        )
        axes[i].invert_yaxis()
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__sigma_N_success_heat_dynamics.png"
    )


MODE = typing.Literal["sigma_lamda", "sigma_N"]
ALPHA_D = typing.Literal["alpha", "d"]


def plot_sigma_success_heat_subplots(
    df: pd.DataFrame,
    mode: MODE = "sigma_lamda",
    alpha_d: ALPHA_D = "alpha",
    plot_dir: str = "../results/plots/",
) -> None:
    """Plot the success probability against sigma for different dynamics."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)
    if mode == "sigma_lamda":
        melt_vars = ["Dynamic Name", "sigma", "Function Name", "lamda"]
        pivot_vars = ["lamda", "sigma"]
    elif mode == "sigma_N":
        melt_vars = ["Dynamic Name", "sigma", "Function Name", "N"]
        pivot_vars = ["N", "sigma"]
    alpha_values = set(df["alpha"])
    d_values = set(df["d"])
    dynamic_names = set(df["Dynamic Name"])
    dfs = {}
    for dynamic in dynamic_names:
        if alpha_d == "alpha":
            alpha_df = {}
            for alpha in alpha_values:
                df_alpha = df[(df["Dynamic Name"] == dynamic) & (df["alpha"] == alpha)]
                dfm_alpha = df_alpha.melt(
                    id_vars=melt_vars,
                    value_vars=["Success Probability"],
                    var_name="variable",
                    value_name="success_prob",
                ).pivot(
                    index=pivot_vars[0], columns=pivot_vars[1], values="success_prob"
                )
                alpha_df[f"{alpha}"] = dfm_alpha
            dfs[dynamic] = alpha_df
        elif alpha_d == "d":
            d_df = {}
            for d in d_values:
                df_d = df[(df["Dynamic Name"] == dynamic) & (df["d"] == d)]
                dfm_d = df_d.melt(
                    id_vars=melt_vars,
                    value_vars=["Success Probability"],
                    var_name="variable",
                    value_name="success_prob",
                ).pivot(
                    index=pivot_vars[0], columns=pivot_vars[1], values="success_prob"
                )
                d_df[f"{d}"] = dfm_d
            dfs[dynamic] = d_df

    # Plot the data side by side
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        2, len(dynamic_names), figsize=(20, 30), sharey=True, sharex=True
    )
    if len(dynamic_names) == 1:
        axes = np.expand_dims(axes, axis=1)

    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    title_str = f"{f}, max_it={max_it}, dt={dt}"
    if mode == "sigma_N":
        lamda = df.iloc[0].loc["lamda"]
        title_str += f", lamda={lamda}"
    else:
        n = df.iloc[0].loc["N"]
        title_str += f", N={n}"

    for i, dynamic in enumerate(dynamic_names):
        if alpha_d == "alpha":
            alpha_values = [1, 10000]
            for j, alpha in enumerate(alpha_values):
                sns.heatmap(dfs[dynamic][f"{alpha}"], ax=axes[j][i], cbar=True)
                d = df.iloc[0].loc["d"]
                axes[j][i].set_title(
                    f"Success Probability for {dynamic}, alpha={alpha}, and d={d}, "
                    + title_str
                )
                axes[j][i].invert_yaxis()
        elif alpha_d == "d":
            for j, d in enumerate(d_values):
                sns.heatmap(dfs[dynamic][f"{d}"], ax=axes[j][i], cbar=True)
                alpha = df.iloc[0].loc["alpha"]
                axes[j][i].set_title(
                    f"Success Probability for {dynamic}, alpha={alpha}, and d={d}, "
                    + title_str
                )
                axes[j][i].invert_yaxis()

    plt.subplots_adjust(hspace=0.5)
    for ax in axes.flat:
        ax.tick_params(axis="y", labelrotation=45)
        ax.tick_params(axis="x", labelsize=8, labelrotation=45)

    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__plot_sigma_lamda_success_heat_subplots.png"
    )


def plot_sigma_max_to_lambda(
    df: pd.DataFrame, plot_dir: str = "../results/plots/"
) -> None:
    # Compute mean success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # Find the sigma that maximizes success probability for each Dynamic Name and lamda
    df_max_sigma = df.loc[
        df.groupby(["Dynamic Name", "lamda"])["Success Probability"].idxmax()
    ]

    # Prepare the data for plotting
    df_max_sigma = df_max_sigma[["N", "sigma", "Dynamic Name", "lamda"]]

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    unique_dynamic_names = df_max_sigma["Dynamic Name"].unique()
    legend_labels = []

    for dynamic_name in unique_dynamic_names:
        df_subset = df_max_sigma[df_max_sigma["Dynamic Name"] == dynamic_name]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            df_subset["lamda"], df_subset["sigma"]
        )

        # Plot the data points and regression line
        sns.regplot(
            data=df_subset,
            x="lamda",
            y="sigma",
            label=f"{dynamic_name} (slope={slope:.2f})",
            marker="o",
            scatter_kws={"s": 100},  # Size of the markers
            line_kws={"linewidth": 2},  # Width of the regression line
        )

        # Prepare custom legend label with the slope value
        legend_labels.append(f"{dynamic_name} (slope={slope:.2f})")

    plt.xlabel("lambda")
    plt.ylabel("Sigma (Max Success Probability)")
    plt.title("Sigma Position of Maximum Success Probability vs lambda")

    # Custom legend
    plt.legend(title="Dynamic Name")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{df.iloc[0].loc['Experiment Name']}__sigma_max_to_N.png")
    plt.show()
