# Code originally from https://github.com/leomet07/PredictChlorophyllALibrary

import ee
import os
import pandas as pd
import io
import requests
import rasterio
from rasterio.transform import from_bounds
import multiprocessing
import sys
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

## GLOBAL CONSTANTS FOR THIS PROJECT
CLOUD_FILTER = 50


def visualize(tif_path: str):
    # Open the GeoTIFF file
    with rasterio.open(tif_path) as src:
        # Read the number of bands and the dimensions
        num_bands = src.count
        height = src.height
        width = src.width
        tags = src.tags()
        title = f"Date: {tags["date"]}, OBJECTID: {tags["objectid"]}, Scale: {tags["scale"]}\n"

        print(f"Number of bands: {num_bands}")
        # print(f"Dimensions: {width} x {height}")

        # Read the entire image into a numpy array (bands, height, width)
        img = src.read()
        # Display each band separately
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_bands:
                ax.imshow(img[i, :, :], cmap="gray")  # Display each band separately
                ax.set_title(f"Band {i+1}")
                ax.axis("off")
                # print(img[i, :, :])
        plt.tight_layout()
        plt.suptitle(title, fontsize=24)  # Super title for all the subplots!
        plt.show()


def see_if_all_image_bands_valid(band_values):  # min values passed in here usually
    # for band in band_values:
    #     print(
    #         "band: ", band, band_values[band]
    #     )  # catches if band is empty for some reason
    for band in band_values:
        if band_values[band] != None:
            return True
    # if it made it all the way here, all values in this dict are None
    return False


"""
Set up GEE account and get the name of your GEE project.
"""


def open_gee_project(project: str):
    try:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(
            project=project, opt_url="https://earthengine-highvolume.googleapis.com"
        )


"""
Import assets from gee depending on which lake you want an image of
"""


def import_assets(objectid: int, projectName: str) -> ee.FeatureCollection:
    LakeShp = ee.FeatureCollection(
        f"projects/{projectName}/assets/195-ALTM-ALAP-lakes-withCentroid"
    )
    LakeShp = ee.FeatureCollection(
        LakeShp.filter(
            ee.Filter.eq("OBJECTID", objectid)
        )  # use SITE_ID or name to get objectid
    )
    return LakeShp


"""# MAIN Atmospheric Correction

Page, B.P., Olmanson, L.G. and Mishra, D.R., 2019. A harmonized image processing workflow using Sentinel-2/MSI and Landsat-8/OLI for mapping water clarity in optically variable lake systems. Remote Sensing of Environment, 231, p.111284.

https://github.com/Nateme16/geo-aquawatch-water-quality/blob/main/Atmospheric%20corrections/main_L8L9.ipynb
"""

# MAIN Atmospheric Correction


def atm_corr(img):
    img_b6_raw = img.select("B6")
    img_b7_raw = img.select("B7")
    img_b8_raw = img.select("B8")

    target_image_number = 1
    ozone = ee.ImageCollection("TOMS/MERGED")
    pi = ee.Image(3.141592)
    JRC = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    mask = JRC.select("occurrence").gt(0)

    footprint = img.geometry()

    # DEM
    dem = ee.Image("USGS/SRTMGL1_003").clip(footprint)
    DEM_OLI = ee.Image(1)

    # ozone
    # DU_OLI = ee.Image(ozone.filterBounds(footprint).filter(ee.Filter.calendarRange(startMonth, endMonth, 'month')).filter(ee.Filter.calendarRange(startYear, endYear, 'year')).mean())

    DU_OLI = ee.Image(300)
    # ozone @ sea level

    # Julian Day
    imgDate_OLI = ee.Date(img.get("system:time_start"))
    FOY_OLI = ee.Date.fromYMD(imgDate_OLI.get("year"), 1, 1)
    JD_OLI = imgDate_OLI.difference(FOY_OLI, "day").int().add(1)

    # Earth-Sun distance
    d_OLI = ee.Image.constant(img.get("EARTH_SUN_DISTANCE"))

    # Sun elevation
    SunEl_OLI = ee.Image.constant(img.get("SUN_ELEVATION"))

    # Sun azimuth
    SunAz_OLI = img.select("SAA").multiply(ee.Image(0.01))

    # Satellite zenith
    SatZe_OLI = img.select("VZA").multiply(ee.Image(0.01))
    cosdSatZe_OLI = (SatZe_OLI).multiply(pi.divide(ee.Image(180))).cos()
    sindSatZe_OLI = (SatZe_OLI).multiply(pi.divide(ee.Image(180))).sin()

    # Satellite azimuth
    SatAz_OLI = img.select("VAA").multiply(ee.Image(0.01))

    # Sun zenith
    SunZe_OLI = img.select("SZA").multiply(ee.Image(0.01))
    cosdSunZe_OLI = SunZe_OLI.multiply(
        pi.divide(ee.Image.constant(180))
    ).cos()  # in degrees
    sindSunZe_OLI = SunZe_OLI.multiply(pi.divide(ee.Image(180))).sin()  # in degrees

    # Relative azimuth
    RelAz_OLI = ee.Image(SunAz_OLI)
    cosdRelAz_OLI = RelAz_OLI.multiply(pi.divide(ee.Image(180))).cos()

    # Pressure calculation
    P_OLI = (
        ee.Image(101325)
        .multiply(
            ee.Image(1).subtract(ee.Image(0.0000225577).multiply(DEM_OLI)).pow(5.25588)
        )
        .multiply(0.01)
    )
    Po_OLI = ee.Image(1013.25)

    # Radiometric Calibration
    # define bands to be converted to radiance
    bands_OLI = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]

    # radiance_mult_bands
    rad_mult_OLI = ee.Image(
        ee.Array(
            [
                ee.Image(img.get("RADIANCE_MULT_BAND_1")),
                ee.Image(img.get("RADIANCE_MULT_BAND_2")),
                ee.Image(img.get("RADIANCE_MULT_BAND_3")),
                ee.Image(img.get("RADIANCE_MULT_BAND_4")),
                ee.Image(img.get("RADIANCE_MULT_BAND_5")),
                ee.Image(img.get("RADIANCE_MULT_BAND_6")),
                ee.Image(img.get("RADIANCE_MULT_BAND_7")),
                ee.Image(img.get("RADIANCE_MULT_BAND_8")),
            ]
        )
    ).toArray(1)

    # radiance add band
    rad_add_OLI = ee.Image(
        ee.Array(
            [
                ee.Image(img.get("RADIANCE_ADD_BAND_1")),
                ee.Image(img.get("RADIANCE_ADD_BAND_2")),
                ee.Image(img.get("RADIANCE_ADD_BAND_3")),
                ee.Image(img.get("RADIANCE_ADD_BAND_4")),
                ee.Image(img.get("RADIANCE_ADD_BAND_5")),
                ee.Image(img.get("RADIANCE_ADD_BAND_6")),
                ee.Image(img.get("RADIANCE_ADD_BAND_7")),
                ee.Image(img.get("RADIANCE_ADD_BAND_8")),
            ]
        )
    ).toArray(1)

    # create an empty image to save new radiance bands to
    # print('Bands in image:', img.bandNames())

    imgArr_OLI = img.select(bands_OLI).toArray().toArray(1)
    Ltoa_OLI = imgArr_OLI.multiply(rad_mult_OLI).add(rad_add_OLI)

    # print(Ltoa_OLI)

    # esun (extra-terrestrial solar irradiance) Units = mW cm-2 um-1
    ESUN_OLI = (
        ee.Image.constant(197.24790954589844)
        .addBands(ee.Image.constant(201.98426818847656))
        .addBands(ee.Image.constant(186.12677001953125))
        .addBands(ee.Image.constant(156.95257568359375))
        .addBands(ee.Image.constant(96.04714965820312))
        .addBands(ee.Image.constant(23.8833221450863))
        .addBands(ee.Image.constant(8.04995873449635))
        .addBands(ee.Image.constant(173.7))  # TODO: verify 173.7 value from USGS
        .toArray()
        .toArray(1)
    )  # Example esun value for b8
    ESUN_OLI = ESUN_OLI.multiply(ee.Image(1))

    ESUNImg_OLI = ESUN_OLI.arrayProject([0]).arrayFlatten([bands_OLI])

    # Ozone Correction
    # Ozone coefficients https://www.arm.gov/publications/tech_reports/doe-sc-arm-tr-129.pdf?id=811 (Appendix A) by band center (lambda)
    koz_OLI = (
        ee.Image.constant(0.0039)
        .addBands(ee.Image.constant(0.0218))
        .addBands(ee.Image.constant(0.1078))
        .addBands(ee.Image.constant(0.0608))
        .addBands(ee.Image.constant(0.0019))
        .addBands(ee.Image.constant(0))
        .addBands(ee.Image.constant(0))
        .addBands(ee.Image.constant(0))
        .toArray()
        .toArray(1)
    )

    # Calculate ozone optical thickness
    Toz_OLI = koz_OLI.multiply(DU_OLI).divide(ee.Image.constant(1000))

    # Calculate TOA radiance in the absense of ozone
    Lt_OLI = Ltoa_OLI.multiply(
        ((Toz_OLI))
        .multiply(
            (ee.Image.constant(1).divide(cosdSunZe_OLI)).add(
                ee.Image.constant(1).divide(cosdSatZe_OLI)
            )
        )
        .exp()
    )

    # Rayleigh optical thickness
    bandCenter_OLI = (
        ee.Image(443)
        .divide(1000)
        .addBands(ee.Image(483).divide(1000))
        .addBands(ee.Image(561).divide(1000))
        .addBands(ee.Image(655).divide(1000))
        .addBands(ee.Image(865).divide(1000))
        .addBands(ee.Image(1609).divide(1000))
        .addBands(ee.Number(2201).divide(1000))
        .addBands(ee.Image(590).divide(1000))
        .toArray()
        .toArray(1)
    )

    # create an empty image to save new Tr values to
    Tr_OLI = (
        (P_OLI.divide(Po_OLI))
        .multiply(ee.Image(0.008569).multiply(bandCenter_OLI.pow(-4)))
        .multiply(
            (
                ee.Image(1)
                .add(ee.Image(0.0113).multiply(bandCenter_OLI.pow(-2)))
                .add(ee.Image(0.00013).multiply(bandCenter_OLI.pow(-4)))
            )
        )
    )

    # Fresnel Reflection
    # Specular reflection (s- and p- polarization states)
    theta_V_OLI = ee.Image(0.0000000001)
    sin_theta_j_OLI = sindSunZe_OLI.divide(ee.Image(1.333))

    theta_j_OLI = sin_theta_j_OLI.asin().multiply(ee.Image(180).divide(pi))

    theta_SZ_OLI = SunZe_OLI

    R_theta_SZ_s_OLI = (
        (
            (theta_SZ_OLI.multiply(pi.divide(ee.Image(180)))).subtract(
                theta_j_OLI.multiply(pi.divide(ee.Image(180)))
            )
        )
        .sin()
        .pow(2)
    ).divide(
        (
            (
                (theta_SZ_OLI.multiply(pi.divide(ee.Image(180)))).add(
                    theta_j_OLI.multiply(pi.divide(ee.Image(180)))
                )
            )
            .sin()
            .pow(2)
        )
    )

    R_theta_V_s_OLI = ee.Image(0.0000000001)

    R_theta_SZ_p_OLI = (
        (
            (theta_SZ_OLI.multiply(pi.divide(180))).subtract(
                theta_j_OLI.multiply(pi.divide(180))
            )
        )
        .tan()
        .pow(2)
    ).divide(
        (
            (
                (theta_SZ_OLI.multiply(pi.divide(180))).add(
                    theta_j_OLI.multiply(pi.divide(180))
                )
            )
            .tan()
            .pow(2)
        )
    )

    R_theta_V_p_OLI = ee.Image(0.0000000001)

    R_theta_SZ_OLI = ee.Image(0.5).multiply(R_theta_SZ_s_OLI.add(R_theta_SZ_p_OLI))

    R_theta_V_OLI = ee.Image(0.5).multiply(R_theta_V_s_OLI.add(R_theta_V_p_OLI))

    # Rayleigh scattering phase function
    # Sun-sensor geometry

    theta_neg_OLI = (
        (cosdSunZe_OLI.multiply(ee.Image(-1))).multiply(cosdSatZe_OLI)
    ).subtract((sindSunZe_OLI).multiply(sindSatZe_OLI).multiply(cosdRelAz_OLI))

    theta_neg_inv_OLI = theta_neg_OLI.acos().multiply(ee.Image(180).divide(pi))

    theta_pos_OLI = (cosdSunZe_OLI.multiply(cosdSatZe_OLI)).subtract(
        sindSunZe_OLI.multiply(sindSatZe_OLI).multiply(cosdRelAz_OLI)
    )

    theta_pos_inv_OLI = theta_pos_OLI.acos().multiply(ee.Image(180).divide(pi))

    cosd_tni_OLI = theta_neg_inv_OLI.multiply(pi.divide(180)).cos()  # in degrees

    cosd_tpi_OLI = theta_pos_inv_OLI.multiply(pi.divide(180)).cos()  # in degrees

    Pr_neg_OLI = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tni_OLI.pow(2))))

    Pr_pos_OLI = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tpi_OLI.pow(2))))

    # Rayleigh scattering phase function
    Pr_OLI = Pr_neg_OLI.add((R_theta_SZ_OLI.add(R_theta_V_OLI)).multiply(Pr_pos_OLI))

    # Calulate Lr,
    denom_OLI = ee.Image(4).multiply(pi).multiply(cosdSatZe_OLI)
    Lr_OLI = (ESUN_OLI.multiply(Tr_OLI)).multiply(Pr_OLI.divide(denom_OLI))

    # Rayleigh corrected radiance
    Lrc_OLI = (Lt_OLI.divide(ee.Image(10))).subtract(Lr_OLI)
    LrcImg_OLI = Lrc_OLI.arrayProject([0]).arrayFlatten([bands_OLI])

    # Rayleigh corrected reflectance
    prc_OLI = (
        Lrc_OLI.multiply(pi)
        .multiply(d_OLI.pow(2))
        .divide(ESUN_OLI.multiply(cosdSunZe_OLI))
    )
    prcImg_OLI = prc_OLI.arrayProject([0]).arrayFlatten([bands_OLI])
    rhorc = prc_OLI.arrayProject([0]).arrayFlatten([bands_OLI])

    # Aerosol Correction
    # Bands in nm
    bands_nm_OLI = (
        ee.Image(443)
        .addBands(ee.Image(483))
        .addBands(ee.Image(561))
        .addBands(ee.Image(655))
        .addBands(ee.Image(865))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .toArray()
        .toArray(1)
    )

    # # Lam in SWIR bands
    Lam_6_OLI = LrcImg_OLI.select("B6")
    Lam_7_OLI = LrcImg_OLI.select("B7")

    # Calculate aerosol type
    eps_OLI = (
        (((Lam_7_OLI).divide(ESUNImg_OLI.select("B7"))).log()).subtract(
            ((Lam_6_OLI).divide(ESUNImg_OLI.select("B6"))).log()
        )
    ).divide(
        ee.Image(2201).subtract(ee.Image(1609))
    )  # .multiply(water_mask)

    # Calculate multiple scattering of aerosols for each band
    Lam_OLI = (
        (Lam_7_OLI)
        .multiply(((ESUN_OLI).divide(ESUNImg_OLI.select("B7"))))
        .multiply(
            (eps_OLI.multiply(ee.Image(-1)))
            .multiply((bands_nm_OLI.divide(ee.Image(2201))))
            .exp()
        )
    )

    # diffuse transmittance
    trans_OLI = (
        Tr_OLI.multiply(ee.Image(-1))
        .divide(ee.Image(2))
        .multiply(ee.Image(1).divide(cosdSatZe_OLI))
        .exp()
    )

    # Compute water-leaving radiance
    Lw_OLI = Lrc_OLI.subtract(Lam_OLI).divide(trans_OLI)

    # water-leaving reflectance
    pw_OLI = (
        Lw_OLI.multiply(pi)
        .multiply(d_OLI.pow(2))
        .divide(ESUN_OLI.multiply(cosdSunZe_OLI))
    )
    pwImg_OLI = pw_OLI.arrayProject([0]).arrayFlatten([bands_OLI])

    # Rrs
    Rrs = (
        pw_OLI.divide(pi)
        .arrayProject([0])
        .arrayFlatten([bands_OLI])
        .slice(0, 5)  # only need b1-b5 like before, ignore b6-b8 cuz undefined
    ).multiply(mask)
    Rrs = Rrs.updateMask(Rrs.gt(0))

    new_img_with_raws = (
        Rrs.addBands(img_b6_raw, overwrite=True)
        .addBands(img_b7_raw, overwrite=True)
        .addBands(img_b8_raw, overwrite=True)
    )

    return new_img_with_raws.set("system:time_start", img.get("system:time_start"))


"""## Creating mask functions

These functions mask clouds based on the QA_PIXEL band (maskL8sr), select pixels that are >= 75% water (jrcMask), and a 30m buffer around roads to mask bridges (roadMask)
"""


# Creating mask functions
# function to mask out clouds
def maskL8sr(image):
    # Bits 2, 3, and 4, are cirrus, cloud  and cloudshadow, respectively.
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 4
    cirrusBitMask = 1 << 2
    waterBitMask = 1 << 7  # 1 means water

    # Get the pixel QA band.
    qa = image.select("QA_PIXEL")
    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask)
        .eq(0)
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))  # want not high cloud confidence
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        .And(qa.bitwiseAnd(waterBitMask).neq(0))  # keep water
    )
    return image.updateMask(mask)


# jrc water occurrence mask
def jrcMask(image):
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # select only water occurence
    occurrence = jrc.select("occurrence")
    # selectonly water occurences of greater than 75%
    water_mask = occurrence.mask(occurrence.gt(50))
    return image.updateMask(water_mask)


def func_uem(feature):
    num = ee.Number.parse(ee.String(feature.get("linearid")))
    return feature.set("linearid", num)


def bufferPoly30(feature):
    return feature.buffer(30)


# Creating 30m road buffer mask
def roadMask(image):
    roads = ee.FeatureCollection("TIGER/2016/Roads")
    # 30m road buffer

    Buffer = roads.map(bufferPoly30)
    # Convert 'areasqkm' property from string to number.

    roadBuffer = Buffer.map(func_uem)
    roadRaster = roadBuffer.reduceToImage(["linearid"], ee.Reducer.first())
    # create an image with a constant value of one to apply roadmask to
    blank = ee.Image.constant(1)
    inverseMask = blank.updateMask(roadRaster)
    # get reverse mask to have everything but roads kept
    mask1 = inverseMask.mask().Not()
    return image.updateMask(mask1)


def import_collections(filter_range, LakeShp) -> ee.Image:
    """## Buffer function for points"""
    # # filter landsat 8 and 9 scenes by path / row
    FC_OLI = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1")  # level 1 (T1_L2 would be level 2)
        .filterMetadata("CLOUD_COVER", "less_than", CLOUD_FILTER)
        .filter(filter_range)
        .filterBounds(LakeShp)
        .map(maskL8sr)
        .map(jrcMask)
        .map(roadMask)
        .sort("system:time_start")
    )

    FC_OLI2 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1")  # level 1 (T1_L2 would be level 2)
        .filterMetadata("CLOUD_COVER", "less_than", CLOUD_FILTER)
        .filter(filter_range)
        .filterBounds(LakeShp)
        .map(maskL8sr)
        .map(jrcMask)
        .map(roadMask)
        .sort("system:time_start")
    )

    FC_combined = FC_OLI.merge(FC_OLI2).sort("system:time_start")

    # filter S2A by the filtered buffer and apply atm corr
    FC_combined = FC_combined.map(atm_corr).sort("system:time_start")

    FC_combined = FC_combined.select(["B1", "B2", "B3", "B4", "B5"])

    return FC_combined


def get_image_and_date_from_image_collection(coll, index, shp):
    image = ee.Image(coll.toList(coll.size()).get(index))
    image_index = image.get("system:index").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    image = image.clip(shp)
    image = image.toFloat()

    return image, image_index, date


def get_raster(start_date, end_date, LakeShp, scale) -> ee.Image:
    date_range = ee.Filter.date(start_date, end_date)
    filter_range = ee.Filter.Or(date_range)

    merged_landsat_image_collection = import_collections(filter_range, LakeShp)
    merged_landsat_image_collection_len = (
        merged_landsat_image_collection.size().getInfo()
    )

    if merged_landsat_image_collection_len == 0:
        raise Exception("NO IMAGES FOUND")

    for i in range(0, merged_landsat_image_collection_len):
        image, image_index, date = get_image_and_date_from_image_collection(
            merged_landsat_image_collection, i, LakeShp
        )

        min_value = image.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=LakeShp.geometry(),  # or your specific geometry
            scale=scale,
            maxPixels=1e9,
            crs="EPSG:4326",
        ).getInfo()

        if see_if_all_image_bands_valid(min_value):
            return image, image_index, date
    # if it made it here, all have blank images (due to NASA JPL aggressive cloud alterer/filter)
    raise Exception("IMAGE IS ALL BLANK :(((")


def export_raster_main_landsat(
    out_dir: str,
    out_filename: str,
    project: str,
    lakeid: int,
    start_date: str,
    end_date: str,
    insitu_date: str,
    scale: int,
    shouldVisualize: bool = False,
    index_logfile_path=None,
):
    LakeShp = import_assets(lakeid, project)  # get shape of lake
    image, image_index, date = get_raster(
        start_date=start_date, end_date=end_date, LakeShp=LakeShp, scale=scale
    )

    url = image.getDownloadURL(
        {
            "format": "GEO_TIFF",
            "scale": scale,  #  increasing this makes predictions more blocky but reduces request size (smaller means more resolution tho!)
            "region": LakeShp.geometry(),
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # export!
    out_filepath = os.path.join(out_dir, out_filename)
    # download image, and then view metadata with rasterio
    # print("Downloading raster...")

    response = requests.get(url)
    with open(out_filepath, "wb") as f:
        f.write(response.content)

    new_metadata = {
        "date": date,
        "closest_insitu_date": insitu_date,  # this was the date from the insitu
        "objectid": lakeid,
        "scale": scale,
        "satellite": "landsat",
        "image_index": image_index,
        "algorithm": "MAIN",
    }
    with rasterio.open(out_filepath, "r+") as dst:
        dst.update_tags(**new_metadata)

    if shouldVisualize:
        print(f"Image saved to {out_filepath}")
        print("Saved image metadata: ", new_metadata)
        visualize(out_filepath)

    if index_logfile_path:
        with open(index_logfile_path, "a") as index_logfile:
            index_logfile.write(f"{image_index}\n")

    return out_filepath


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "python landsat.py <out_dir> <project> <lakeid> <start_date> <end_date> <scale> <out_filename>"
        )
        sys.exit(1)

    out_dir = sys.argv[1]
    project = sys.argv[2]
    lakeid = int(sys.argv[3])
    start_date = sys.argv[4]  # STR, in format YYYY-MM-DD
    end_date = sys.argv[5]  # STR, in format YYYY-MM-DD
    scale = int(sys.argv[6])
    out_filename = sys.argv[7]

    open_gee_project(project=project)

    export_raster_main_landsat(
        out_dir=out_dir,
        out_filename=out_filename,
        project=project,
        lakeid=lakeid,
        start_date=start_date,
        end_date=end_date,
        insitu_date=start_date,  # this doesnt matter when only fetching one, just for testing
        scale=scale,
        shouldVisualize=True,
    )
