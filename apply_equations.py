import tqdm
import rasterio
import numpy as np
import os
from matplotlib import pyplot as plt
import inspect_shapefile
from shapely.geometry import Point
from sklearn.metrics import r2_score
import rasterio.features
import rasterio.mask


tif_folder = "queer_lake_tifs"

display = True

doc_values = []
predicted_a440_values = []

for filename in os.listdir(tif_folder):
    tif_filepath = os.path.join(tif_folder, filename)

    with rasterio.open(tif_filepath) as src:
        profile = src.profile  # Get the profile of the existing raster
        transform = src.transform
        tags = src.tags()
        scale = tags["scale"]
        x_res = src.res[0]  # same as src.res[1]
        closest_insitu_date = tags["closest_insitu_date"]
        objectid = tags["objectid"]

        # get lat and long
        centroid_lat = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lat-Cent"].item()
        centroid_long = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lon-Cent"].item()

        # create circle mask
        radius_in_meters = 60

        circle = Point(centroid_long, centroid_lat).buffer(
            x_res * (radius_in_meters / float(scale))
        )  # however many x_res sized pixels needed for buffer of radius at downloaded scale

        out_image, transformed = rasterio.mask.mask(
            src, [circle], invert=False, crop=False
        )  # read pixels within just the circle mask

    out_image[~np.isfinite(out_image)] = np.nan
    band3 = out_image[2]  # zero indexed
    band4 = out_image[3]
    band5 = out_image[4]
    mean_band_3 = np.nanmean(band3)
    mean_band_4 = np.nanmean(band4)
    mean_band_5 = np.nanmean(band5)

    ratio3meanto5mean = mean_band_3 / mean_band_5

    b0 = 23.5
    b1 = -36
    b2 = 0.004

    output_ln_a440 = (
        b0 + b1 * (ratio3meanto5mean) + b2 * (mean_band_4)
    )  # this is now a single value
    print(
        f"output_ln_a440: {output_ln_a440}, ratio3meanto5mean: {ratio3meanto5mean}, mean_band_3: {mean_band_3}, mean_band_4: {mean_band_4}, mean_band_5: {mean_band_5}"
    )
    a440 = np.exp(
        output_ln_a440
    )  # this is now a single float # a440 is absorptivity of filtered water at 440nm wavelength, a measure of CDOM, proportional to DOC

    # matched doc
    all_doc = inspect_shapefile.truth_data[
        (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
        & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
    ]["DOC_MG_L"]
    doc = all_doc.item()

    if display:
        title = f"Prediction for Lake-OID-{objectid} on {closest_insitu_date}"
        fig = plt.figure(title, figsize=(10, 8))
        plt.imshow(band4, cmap="viridis", interpolation="none")
        plt.title(title)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    print(doc, a440)
    if not np.isfinite(a440):  # nanmean can return inf or nan
        print("a440 is not a finite value, continuing")
        continue
    doc_values.append(doc)  # true value
    predicted_a440_values.append(a440)  # predicted value

r2 = r2_score(doc_values, predicted_a440_values)
print("R2 score: ", r2)
