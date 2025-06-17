import tqdm
import rasterio
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import inspect_shapefile
from shapely.geometry import Point
from sklearn.metrics import r2_score
import rasterio.features
import warnings
from sklearn.linear_model import LinearRegression


def get_3_5_ratio(bands):
    return bands[2] / bands[4]  # zero-indexed


def get_ln_2_5_ratio(bands):
    return np.log(bands[1] / bands[4])  # zero-indexed


def get_2_5_ratio(bands):
    return bands[1] / bands[4]  # zero-indexed


def get_3_4_ratio(bands):
    return bands[2] / bands[3]  # zero-indexed


def get_1_4_ratio_and_2_4_ratio(bands):
    return (bands[0] / bands[3]), (bands[1] / bands[3])  # zero-indexed


def get_1_4_ratio_and_3_4_ratio(bands):
    return (bands[0] / bands[3]), (bands[2] / bands[3])  # zero-indexed


def get_3_4_ratio_and_4_5_ratio(bands):
    return (bands[2] / bands[3]), (bands[3] / bands[4])  # zero-indexed


equation_functions = [
    get_3_5_ratio,
    get_ln_2_5_ratio,
    get_2_5_ratio,
    get_3_4_ratio,
    get_1_4_ratio_and_2_4_ratio,
    get_1_4_ratio_and_3_4_ratio,
    get_3_4_ratio_and_4_5_ratio,
]


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

        ratios = []

        with np.errstate(divide="ignore", invalid="ignore"):
            for equation_function in equation_functions:
                ratios.append(equation_function(bands))

        return (
            ratios,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


out_folder = "all_lake_images"

display = False

results = []

for subfolder in os.listdir(out_folder):
    true_doc_values = []
    predicted_ratio_ln_a440_value_by_equation = list(
        map(lambda x: [], equation_functions)
    )  # index 0 corresponds with first equation, 1 with second, etc...

    tif_folder_path = os.path.join(out_folder, subfolder)

    for filename in os.listdir(tif_folder_path):
        tif_filepath = os.path.join(tif_folder_path, filename)

        (
            ratios,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        ) = get_ratio_from_tif(tif_filepath, equation_functions)

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
            [circle], ratios[0].shape, transform
        )

        for ratio in ratios:
            if type(ratio) is tuple:  # multi vaiable
                for subratio in ratio:
                    subratio[outside_circle_mask] = (
                        np.nan
                    )  # arrays store pointer to ratio array, this is okay bc just a mutation
            else:
                ratio[outside_circle_mask] = (
                    np.nan
                )  # arrays store pointer to ratio array, this is okay bc just a mutation

        # copy over geo data from tif to output, then get circle of output and take average

        if display:
            title = f"Prediction for Lake-OID-{objectid} on {closest_insitu_date}"
            fig = plt.figure(title, figsize=(10, 8))
            plt.imshow(ratio_ln_a440, cmap="viridis", interpolation="none")
            plt.title(title)
            plt.colorbar()
            plt.axis("off")
            plt.show()

        is_any_mean_ratio_nan = False
        mean_ratio_ln_a440_list = []
        for ratio_index in range(len(ratios)):
            ratio = ratios[ratio_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if type(ratio) is tuple:  # multi vaiable
                    mean_ratio_ln_a440 = []
                    for subratio in ratio:
                        mean_ratio_ln_a440.append(np.nanmean(subratio))
                else:
                    mean_ratio_ln_a440 = np.nanmean(ratio)

            if not np.all(
                np.isfinite(mean_ratio_ln_a440)
            ):  # nanmean can return inf or nan if array is all nans
                is_any_mean_ratio_nan = (
                    True  # hopefully if one is nan, all the rest are nan too
                )
                break

            mean_ratio_ln_a440_list.append(mean_ratio_ln_a440)

        if not is_any_mean_ratio_nan:
            # only append finite values to r2 comparison
            true_doc_values.append(doc)
            for i in range(len(ratios)):
                predicted_ratio_ln_a440_value_by_equation[i].append(
                    mean_ratio_ln_a440_list[i]
                )  # predicted value

    true_doc_values = np.array(true_doc_values)
    true_ln_doc_values = np.log(true_doc_values)  # base e

    this_result = []
    for i in range(len(equation_functions)):
        predicted_ratio_ln_a440_values = predicted_ratio_ln_a440_value_by_equation[i]

        if type(ratios[i]) is tuple:
            X = predicted_ratio_ln_a440_values
        else:
            X = np.array(predicted_ratio_ln_a440_values).reshape(
                -1, 1
            )  # make it in [[1], [2], [3]] shape

        # see slope of line of best fit
        reg = LinearRegression().fit(
            X, true_ln_doc_values
        )  # .reshape(-1, 1) because there is only one feature

        regression_r2_score = reg.score(
            X, true_ln_doc_values
        )  # with proper slope applied

        this_result.append(regression_r2_score)
        # print(
        #     f"EQUATION INDEX ({i})| {regression_r2_score:.3f} is the r2 of scaled ln-a440 to ln-DOC for {subfolder}"
        # )

        if display:
            # plot all values
            plt.scatter(
                predicted_ratio_ln_a440_values, true_ln_doc_values
            )  # band ratio to a440
            plt.plot(
                predicted_ratio_ln_a440_values, reg.predict(X)
            )  # plot line of best fit of a440 to true doc
            plt.xlabel("predicted a440")
            plt.ylabel("DOC")
            plt.show()

    proper_lake_name = inspect_shapefile.shp_df[
        inspect_shapefile.shp_df["OBJECTID"] == float(objectid)
    ]["NAME"].iloc[0]

    this_result.append(len(true_ln_doc_values))
    this_result.append(
        objectid
    )  # latest object id should be same as all other object ids for this subfolder
    this_result.append(
        proper_lake_name
    )  # latest object id should be same as all other object ids for this subfolder

    results.append(this_result)

results_df = pd.DataFrame.from_records(results)
results_df.columns = [
    "3_5_ratio",
    "ln_2_5_ratio",
    "2_5_ratio",
    "3_4_ratio",
    "1_4_ratio_and_2_4_ratio",
    "1_4_ratio_and_3_4_ratio",
    "3_4_ratio_and_4_5_ratio",
    "number_truth_values",
    "objectid",
    "name",
]

results_df.to_csv("results.csv")

print("Results: \n", results_df)
