import tqdm
import rasterio
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import inspect_shapefile
from shapely.geometry import Point
from sklearn.metrics import r2_score, root_mean_squared_error
import rasterio.features
import warnings
from sklearn.linear_model import LinearRegression
import equations


def get_ratio_from_tif(tif_path, equation_functions):
    with rasterio.open(tif_path) as src:
        profile = src.profile  # Get the profile of the existing raster
        transform = src.transform
        tags = src.tags()
        scale = tags["scale"]
        x_res = src.res[0]  # same as src.res[1]
        closest_insitu_date = tags["closest_insitu_date"]
        objectid = tags["objectid"]

        bands = src.read()

        list_of_ratio_tuples = []

        with np.errstate(divide="ignore", invalid="ignore"):
            for equation_function in equation_functions:
                list_of_ratio_tuples.append(equation_function(bands))

        return (
            list_of_ratio_tuples,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


number_of_equations = len(equations.equation_functions)

out_folder = "all_lake_images_five_front_and_back_water_mask"

display = False

list_of_results_df_rows = []
results_reg_eq = []

all_X_s_ever_by_equation = list(map(lambda x: [], equations.equation_functions))
all_true_ln_docs_ever = []

for subfolder in os.listdir(out_folder):
    if os.path.isfile(os.path.join(out_folder, subfolder)):
        continue  # this is the log file
    if subfolder == "rondaxe,_lake_tifs" or subfolder == "otter_lake_tifs":
        continue  # temporary, rondaxe does not have enough pixels around centroid
    true_doc_values = []
    input_means_by_equation = list(
        map(lambda x: [], equations.equation_functions)
    )  # index 0 corresponds with first equation, 1 with second, etc...

    tif_folder_path = os.path.join(out_folder, subfolder)

    for filename in os.listdir(tif_folder_path):
        tif_filepath = os.path.join(tif_folder_path, filename)

        (
            list_of_ratio_tuples,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        ) = get_ratio_from_tif(tif_filepath, equations.equation_functions)

        # a440 is absorptivity of filtered water at 440nm wavelength, a measure of CDOM, proportional to DOC

        # matched doc
        all_doc = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["DOC_MG_L"]
        try:
            doc = all_doc.item()
        except ValueError:  # array either has 2+ or 0 items
            if len(all_doc) > 0:  # means 2+ measurements for that date, take mean
                doc = all_doc.mean()
            else:
                raise Exception("No DOC values found for that date.")

        # get lat and long
        centroid_lat = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lat-Cent"].iloc[
            0
        ]  # take first entry, lake centroid lat will be the same for any matched insitu
        centroid_long = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lon-Cent"].iloc[
            0
        ]  # take first entry, lake centroid long will be the same for any matched insitu

        radius_in_meters = 60
        circle = Point(centroid_long, centroid_lat).buffer(
            x_res * (radius_in_meters / float(scale))
        )  # however many x_res sized pixels needed for buffer of radius at downloaded scale

        outside_circle_mask = rasterio.features.geometry_mask(
            [circle], list_of_ratio_tuples[0][0].shape, transform
        )

        for ratio_tuple in list_of_ratio_tuples:
            # multi vaiable
            for subratio in ratio_tuple:
                subratio[outside_circle_mask] = (
                    np.nan
                )  # arrays store pointer to ratio array, this is okay bc just a mutation

        # ------------------------------------------------------------
        # MAKE SURE THAT FOR THIS TIFF, CENTROID MEAN ACTUALLY EXISTS (consisting of at least 3 pixels)
        is_any_mean_ratio_nan = False
        list_of_input_tuples = []

        for ratio_tuple_index in range(len(list_of_ratio_tuples)):
            ratio_tuple = list_of_ratio_tuples[ratio_tuple_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # multi vaiable
                mean_ratio_tuple = []
                for subratio in ratio_tuple:
                    #  number of values to be averaging
                    number_of_valid_pixels_within_centroid = np.sum(~np.isnan(subratio))

                    if number_of_valid_pixels_within_centroid < 3:
                        is_any_mean_ratio_nan = (
                            True  # less than 3 pixels is not good enough to get a mean
                        )
                        break
                    mean_ratio_tuple.append(np.nanmean(subratio))

            if not np.all(
                np.isfinite(mean_ratio_tuple)
            ):  # nanmean can return inf or nan if array is all nans or has infinities in it (infs bugs is why this check is still needed)
                is_any_mean_ratio_nan = (
                    True  # hopefully if one is nan, all the rest are nan too
                )
                break

            list_of_input_tuples.append(mean_ratio_tuple)

        if not is_any_mean_ratio_nan:
            # only append finite values to r2 comparison
            true_doc_values.append(doc)
            for i in range(number_of_equations):
                input_means_by_equation[i].append(
                    list_of_input_tuples[i]
                )  # predicted value
        # ------------------------------------------------------------

    true_doc_values = np.array(true_doc_values)
    true_ln_doc_values = np.log(true_doc_values)  # base e
    all_true_ln_docs_ever.extend(true_ln_doc_values)

    # this_results_r2 = []
    results_df_row = {}
    this_results_reg_eq = []
    for i in range(number_of_equations):
        X = input_means_by_equation[i]
        all_X_s_ever_by_equation[i].extend(X)

        # see slope of line of best fit
        # print(f"Subfolder {subfolder} | Equation i{i} | X: ", X)
        reg = LinearRegression().fit(
            X, true_ln_doc_values
        )  # .reshape(-1, 1) because there is only one feature

        predicted_doc = reg.predict(X)
        regression_r2_score = r2_score(true_ln_doc_values, predicted_doc)
        regression_rmse = root_mean_squared_error(true_ln_doc_values, predicted_doc)

        results_df_row[f"equation_i{i}_r2"] = regression_r2_score
        results_df_row[f"equation_i{i}_rmse"] = regression_rmse
        this_results_reg_eq.append(reg)

    proper_lake_name = inspect_shapefile.shp_df[
        inspect_shapefile.shp_df["OBJECTID"] == float(objectid)
    ]["NAME"].iloc[0]

    results_df_row["number_truth_values"] = len(true_ln_doc_values)
    results_df_row["OBJECTID"] = float(
        objectid
    )  # latest object id should be same as all other object ids for this subfolder
    results_df_row["NAME"] = (
        proper_lake_name  # latest object id should be same as all other object ids for this subfolder
    )

    list_of_results_df_rows.append(results_df_row)
    results_reg_eq.append(this_results_reg_eq)

results_df = pd.DataFrame(list_of_results_df_rows)

# first r2s, then rmse
column_order = list(
    map(lambda i: f"equation_i{i}_r2", range(number_of_equations))
) + list(map(lambda i: f"equation_i{i}_rmse", range(number_of_equations)))

results_df = results_df[column_order + ["number_truth_values", "OBJECTID", "NAME"]]

results_df.to_csv("results.csv")

print("Results: \n", results_df)

# Mass apply a set equation to all

lake_moose_row = results_df[results_df["OBJECTID"] == float(298315)]  # big moose
lake_moose_row_index = lake_moose_row.index[0]
equation_index_of_interest = 1

reg_to_use = results_reg_eq[lake_moose_row_index][equation_index_of_interest]

r2_score = reg_to_use.score(
    all_X_s_ever_by_equation[equation_index_of_interest], all_true_ln_docs_ever
)
print(
    f"Global r2_score of equation with index ({equation_index_of_interest}): ", r2_score
)
