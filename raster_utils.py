import numpy as np
import rasterio
from shapely.geometry import Point
import rasterio.mask


def get_circular_section_from_file(
    file_path: str, lat: float, lng: float, radius_in_meters: float
):
    with rasterio.open(file_path) as src:
        x_res = src.res[0]  # same as src.res[1]
        scale = src.tags()["scale"]

        circle = Point(lng, lat).buffer(
            x_res * (radius_in_meters / float(scale))
        )  # however many x_res sized pixels needed for buffer of radius at downloaded scale

        out_image, transformed = rasterio.mask.mask(
            src, [circle], invert=False, crop=True
        )  # read pixels within just the circle mask

        return out_image


def run_analytics_on_raster(raster_array):
    flatten_raster_array = raster_array.flatten()
    flatten_raster_array = flatten_raster_array[
        np.isfinite(flatten_raster_array)
    ]  # Remove infinities, nans

    max_val = np.nanmax(flatten_raster_array)
    min_val = np.nanmin(flatten_raster_array)

    mean_val = np.nanmean(flatten_raster_array)  # mean EXCLUDING nans
    stdev = np.nanstd(flatten_raster_array)  # std EXCLUDING nans

    # top_ten_highest_indices = np.argpartition(flatten_raster_array, -10)[-10:]
    # top_ten = flatten_raster_array[top_ten_highest_indices]
    # top_ten.sort()
    # print("Top ten: ", top_ten)

    return max_val, min_val, mean_val, stdev
