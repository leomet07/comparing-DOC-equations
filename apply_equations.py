import tqdm
import rasterio
import numpy as np
import os
from matplotlib import pyplot as plt
import inspect_shapefile
from shapely.geometry import Point
from sklearn.metrics import r2_score
import rasterio.features
import warnings


def apply_equation_to_tif(tif_path):
    with rasterio.open(tif_path) as src:
        profile = src.profile  # Get the profile of the existing raster
        transform = src.transform
        tags = src.tags()
        scale = tags["scale"]
        x_res = src.res[0]  # same as src.res[1]
        closest_insitu_date = tags["closest_insitu_date"]
        objectid = tags["objectid"]

        band3 = src.read(3)
        band4 = src.read(4)
        band5 = src.read(5)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio3to5 = (
                band3 / band5
            )  # masked out parts may be nan, dividing nan by nan is invalid

            b0 = 23.5
            b1 = -36
            b2 = 0.004

            y = b0 + b1 * (ratio3to5) + b2 * (band4)

        return (
            y,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


out_folder = "all_lake_images"

display = False

for subfolder in os.listdir(out_folder):
    doc_values = []
    predicted_a440_values = []

    tif_folder_path = os.path.join(out_folder, subfolder)

    for filename in os.listdir(tif_folder_path):
        tif_filepath = os.path.join(tif_folder_path, filename)

        (
            output_ln_a440,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        ) = apply_equation_to_tif(tif_filepath)

        a440 = np.exp(
            output_ln_a440
        )  # a440 is absorptivity of filtered water at 440nm wavelength, a measure of CDOM, proportional to DOC

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
            [circle], a440.shape, transform
        )

        a440[outside_circle_mask] = np.nan
        # copy over geo data from tif to output, then get circle of output and take average

        if display:
            title = f"Prediction for Lake-OID-{objectid} on {closest_insitu_date}"
            fig = plt.figure(title, figsize=(10, 8))
            plt.imshow(a440, cmap="viridis", interpolation="none")
            plt.title(title)
            plt.colorbar()
            plt.axis("off")
            plt.show()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_a440 = np.nanmean(a440)

        if not np.isfinite(
            mean_a440
        ):  # nanmean can return inf or nan if array is all nans
            continue

        # only append finite values to r2 comparison
        doc_values.append(doc)  # true value
        predicted_a440_values.append(mean_a440)  # predicted value

    r2 = r2_score(doc_values, predicted_a440_values)
    print(f"R2 score for {subfolder}: ", r2)
