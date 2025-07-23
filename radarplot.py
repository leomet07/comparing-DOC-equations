# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import inspect_shapefile
import apply_equations


def df_with_normalization_across_ten_eqs_by_lake(df, lakeid, categories):
    # filter df to this lake
    only_lake_df = df[df["OBJECTID"] == float(lakeid)]
    only_lake_df_index = only_lake_df.index

    for category in categories:
        columns_of_interest = list(
            map(
                lambda equation_index: f"equation_i{equation_index}_{category}",
                range(10),
            )
        )
        only_values_of_interest = only_lake_df[columns_of_interest]

        scaled_only_values_of_interest = only_values_of_interest.apply(
            lambda row: (row - row.min()) / (row.max() - row.min()), axis=1
        )
        scaled_only_values_of_interest = scaled_only_values_of_interest.values.tolist()[
            0
        ]

        for i in range(len(scaled_only_values_of_interest)):
            df.loc[only_lake_df_index, columns_of_interest[i]] = (
                scaled_only_values_of_interest[i]
            )

    return df


def plot_lake(ax, df, lakeid, equation_index, columns_to_plot, fill_color):
    lake_row = df[df["OBJECTID"] == float(lakeid)]
    lake_row = lake_row[columns_to_plot]

    values = lake_row.values.tolist()[0]

    # add first value back so polygon loops around
    values += values[0:1]

    ax.plot(angles, values, linewidth=2, label=f"Equation Index {equation_index}")
    ax.fill(angles, values, alpha=0.1)


# for a specific lake
alg_name = apply_equations.out_folder.split("_")[
    -1
].upper()  # our convention is last underscore contains alg name

number_of_subplots = 4
num_rows = 2
num_cols = number_of_subplots // num_rows
fig, axs = plt.subplots(
    nrows=num_rows, ncols=num_cols, subplot_kw={"projection": "polar"}
)

lakeids_to_graph = [298315, 298126, 298351, 298091]
subplot_index = 0
for lakeid in lakeids_to_graph:
    ax = axs[subplot_index // num_cols, subplot_index % num_cols]

    # -------------------------- Radar plot ---------------------------------
    # number of variable
    categories = ["r2", "rmse", "mae"]
    N = len(categories)
    categories_to_normalize = ["r2", "rmse", "mae"]  # try not using r2 here

    df = df_with_normalization_across_ten_eqs_by_lake(
        apply_equations.results_df, lakeid, categories_to_normalize
    )

    lakename = df[df["OBJECTID"] == lakeid]["NAME"].iloc[0]
    number_of_truth_values = df[df["OBJECTID"] == lakeid]["number_truth_values"].iloc[0]
    print(f"{lakename} has {number_of_truth_values} values.")

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1], minor=False)
    ax.set_xticklabels(categories, minor=False)

    # Draw ylabels
    ax.set_rlabel_position(2 * pi / N)
    ax.set_yticks([0, 0.5, 1], labels=["0.1", "0.5", "1.0"], color="grey", size=7)
    ax.set_ylim(0, 1)

    # plot every equation
    for equation_index in range(10):
        columns_to_plot = list(
            map(lambda category: f"equation_i{equation_index}_{category}", categories)
        )
        plot_lake(ax, df, lakeid, equation_index, columns_to_plot, "r")

    ax.set_title(f"{lakename}")

    if subplot_index == (number_of_subplots - 1):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

    subplot_index += 1
plt.suptitle(f"{alg_name} performance")
plt.show()
