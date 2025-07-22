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
        ).values.tolist()[0]

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

    ax.plot(angles, values, label=f"Equation Index {equation_index}")
    ax.fill(angles, values, alpha=0.1)


# for a specific lake
alg_name = apply_equations.out_folder.split("_")[
    -1
].upper()  # our convention is last underscore contains alg name
lake_of_interest = inspect_shapefile.lake_infos_of_interest[1]
lakeid = lake_of_interest["OBJECTID"]
lakename = lake_of_interest["NAME"]


# -------------------------- Radar plot ---------------------------------
# number of variable
categories = ["r2", "rmse", "mae"]
N = len(categories)

df = df_with_normalization_across_ten_eqs_by_lake(
    apply_equations.results_df, lakeid, categories
)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0, 0.5, 1], ["0.1", "0.5", "1.0"], color="grey", size=7)
plt.ylim(0, 1)


# plot every equation
for equation_index in range(10):
    columns_to_plot = list(
        map(lambda category: f"equation_i{equation_index}_{category}", categories)
    )
    plot_lake(ax, df, lakeid, equation_index, columns_to_plot, "r")


ax.legend()
plt.title(f"{alg_name} performance for {lakename}")
plt.show()
