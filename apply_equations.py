import tqdm
import rasterio
import numpy as np
import os
from matplotlib import pyplot as plt
import inspect_shapefile


def apply_equation_to_tif(tif_path):
    with rasterio.open(tif_path) as src:
        tags = src.tags()
        date = tags["date"]
        objectid = tags["objectid"]

        band3 = src.read(3)
        band4 = src.read(4)
        band5 = src.read(5)

        ratio3to5 = band3 / band5

        b0 = 23.5
        b1 = -36
        b2 = 0.004

        y = b0 + b1 * (ratio3to5) + b2 * (band4)
        return y, date, objectid  # this is raster-of-ln(a440), date, objectid


tif_folder = "woods_lake_tifs"

display = False

predicted_a440_values = []
doc_values = []

for filename in os.listdir(tif_folder):
    tif_filepath = os.path.join(tif_folder, filename)

    output_ln_a440, date, objectid = apply_equation_to_tif(tif_filepath)
    a440 = np.exp(
        output_ln_a440
    )  # a440 is absorptivity of filtered water at 440nm wavelength, a measure of CDOM

    # matched doc
    doc = inspect_shapefile.truth_data[
        (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
        & (inspect_shapefile.truth_data["DATE_SMP"] == date)
    ]["DOC_MG_L"].item()
    print(doc)

    if display:
        title = f"Prediction for Lake-OID-{objectid} on {date}"
        fig = plt.figure(title, figsize=(10, 8))
        plt.imshow(a440, cmap="viridis", interpolation="none")
        plt.title(title)
        plt.colorbar()
        plt.axis("off")
        plt.show()
