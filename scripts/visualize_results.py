import argparse
import sys
import csv
from cbx_experiments.experiment_result import ExperimentResult
import pandas as pd
from cbx_experiments.visualization import (
    plot_sigma_max_to_N,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize results of an experiment.")
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the CSV file containing the experiment results.",
    )
    parser.add_argument(
        "extra_files",
        type=str,
        nargs="*",
        help="Extra CSV files to concatenate with the main CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    experiment_result = ExperimentResult.from_csv(sys.argv[1])
    df = experiment_result.to_dataframe()

    print(df.columns)

    # concatenate the dataframes
    if len(sys.argv) >= 3:
        for i in range(2, len(sys.argv)):
            experiment_result = ExperimentResult.from_csv(sys.argv[i])
            df_extra = experiment_result.to_dataframe()
            df = pd.concat((df, df_extra), ignore_index=True)

    # plot_sigma_success_heat_subplots(df, mode="sigma_n", alpha_d="alpha")
    # plot_N(df)
    # plot_sigma(df)

    # plot_sigma_N_success_heat_dynamics(df)
    plot_sigma_max_to_N(df)

    # plot_sigma_lamda_success_heat_dynamics(df)
    # plot_sigma_N_success_heat_dynamics(df)

    # plot_sigma_lamda_success_heat_dynamics_with_overlay(df)

    # df["Success Probability"] = df["Success"].apply(np.mean)
    # dfm = df.melt(
    #     id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
    #     value_vars=["Success Probability"],
    #     var_name="variable",
    #     value_name="success_prob",
    # )

    # print(dfm.head())

    # print(df.columns)
    # print(df)
    # plot_2d_sigma_lamda(df)

    # dfm = df.melt(
    #     id_vars="Dynamic Name",
    #     value_vars=["Time", "Best Function Value"],
    #     var_name="class",
    #     value_name="value",
    # )

    # sns.catplot(
    #     x="class",
    #     y="value",
    #     hue="Dynamic Name",
    #     data=dfm,
    #     kind="bar",
    #     height=5,
    #     aspect=1,
    # )
    # plt.show()

    # plot_bar_plot_comparing_dynamics_to_best_value(df)

    # # Time Series Analysis: Time vs. Experiment Name
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=df, x="Time", y="Experiment Name", hue="Dynamic Name", marker="o")
    # plt.title("Time vs. Experiment Name")
    # plt.xlabel("Time")
    # plt.ylabel("Experiment Name")
    # plt.show()

    # # Distribution Analysis: Best Function Value
    # plt.figure(figsize=(10, 6))
    # sns.histplot(data=df, x="Best Function Value", bins=20, kde=True)
    # plt.title("Distribution of Best Function Values")
    # plt.xlabel("Best Function Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # # Correlation Analysis: Heatmap of Correlation Matrix
    # numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    # corr_matrix = df[numeric_cols].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title("Correlation Matrix")
    # plt.show()
