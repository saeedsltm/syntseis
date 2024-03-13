import pykonal
import os
import sys
from yaml import SafeLoader, load
from pandas import read_fwf, read_csv
from obspy import read_events
from obspy import UTCDateTime as utc
# from core.Extra import loadxyzm
from obspy.geodetics.base import kilometers2degrees as k2d
from obspy.core.event import Catalog
from numpy import nan


def readConfiguration():
    """
    Read configuration file

    Returns
    -------
    config : dict
        configuration parameters.

    """
    if not os.path.exists("config.yaml"):
        msg = "+++ Could not find configuration file! Aborting ..."
        print(msg)
        sys.exit()
    with open("config.yaml") as f:
        config = load(f, Loader=SafeLoader)
    msg = "+++ Configuration file was loaded successfully ..."
    print(msg)
    return config


def loadVelocityModel():
    vpPath = os.path.join("model", "vp.mod")
    vsPath = os.path.join("model", "vs.mod")
    vp = pykonal.fields.read_hdf(vpPath)
    vs = pykonal.fields.read_hdf(vsPath)
    return vp, vs


def getPick(picks, requestedPickID):
    for pick in picks:
        if pick.resource_id == requestedPickID:
            return pick


def roundTo(x, base=5):
    return base * round(x/base)


# def loadhypo71Out():
#     names = ["yy", "mo", "dd", "A", "hh", "mm", "B", "sssss", "C",
#              "yd", "D", "ymmmm", "E", "xdd", "F", "xmmmm", "G",
#              "depth_", "HHHH", "mag", "L", "ns", "M", "gap", "N", "dmin", "O",
#              "rms_", "erh__", "erz__", "P", "qm"]
#     widths = [len(name) for name in names]
#     bulletin_df = read_fwf("results/hyp71.out", names=names,
#                            widths=widths, header=0)
#     return bulletin_df


def hypo71Nordic(inputCatalogName):
    print(f"+++ Reading & Updating catalog for {inputCatalogName} ...")
    catalog = read_events(f"catalog_{inputCatalogName}.out")
    hypo71_df = read_csv(f"xyzm_{inputCatalogName}.dat", delim_whitespace=True)
    hypo71_df_out = hypo71_df.copy()
    hypo71_df.ERH = k2d(hypo71_df.ERH)
    hypo71_df.ERZ = hypo71_df.ERZ*1e3
    hypo71_df.replace(nan, None, inplace=True)
    outCatalog = Catalog()
    for r, row in hypo71_df.iterrows():
        event = catalog[r]
        eOrt = utc(row.ORT)
        preferred_origin = event.preferred_origin()
        preferred_magnitude = event.preferred_magnitude()
        eLat = row.Lat
        erLat = row.ERH
        eLon = row.Lon
        erLon = row.ERH
        eDep = row.Dep
        erDep = row.ERZ
        eGap = row.GAP
        preferred_origin.time = eOrt
        preferred_origin.latitude = eLat
        preferred_origin.longitude = eLon
        preferred_origin.depth = eDep*1e3
        preferred_origin.latitude_errors.uncertainty = erLat
        preferred_origin.longitude_errors.uncertainty = erLon
        preferred_origin.depth_errors.uncertainty = erDep
        preferred_origin.quality.azimuthal_gap = eGap
        outCatalog.append(event)
        hypo71_df_out.loc[r, "Mag"] = preferred_magnitude.mag
    outCatalog.write(f"hypo71_{inputCatalogName}.out",
                     format="nordic", high_accuracy=False)
    columns = ["ORT", "Lon", "Lat", "Dep", "Mag",
               "Nus", "NuP", "NuS", "ADS", "MDS", "GAP", "RMS", "ERH", "ERZ"]
    with open(f"xyzm_{inputCatalogName}.dat", "w") as f:
        hypo71_df_out.to_string(
            f, columns=columns, index=False, float_format="%7.3f")
