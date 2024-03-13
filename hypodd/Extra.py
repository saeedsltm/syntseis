from pandas import read_csv, to_datetime, Series
from numpy import sqrt, diff, max
from obspy import read_events
from obspy import UTCDateTime as utc
from obspy.geodetics.base import kilometers2degrees as k2d
from obspy.core.event import Catalog
from obspy.geodetics.base import gps2dist_azimuth as gps
from tqdm import tqdm
from yaml import SafeLoader, load
import os
import sys


def readHypoddConfig():
    hypoddConfigPath = os.path.join("files", "hypodd.yml")
    if not os.path.exists(hypoddConfigPath):
        msg = "+++ Could not find hypoDD configuration file! Aborting ..."
        print(msg)
        sys.exit()
    with open(hypoddConfigPath) as f:
        config = load(f, Loader=SafeLoader)
    msg = "+++ Configuration file was loaded successfully."
    print(msg)
    return config


def loadHypoDDRelocFile():
    names = ["ID",  "LAT",  "LON",  "DEPTH",
             "X",  "Y",  "Z",
             "EX",  "EY",  "EZ",
             "YR",  "MO",  "DY",  "HR",  "MI",  "SC",
             "MAG",
             "NCCP",  "NCCS",
             "NCTP",  "NCTS",
             "RCC",  "RCT",
             "CID "]
    hypodd_df = read_csv("hypoDD.reloc", delim_whitespace=True, names=names)
    return hypodd_df


def writexyzm(outName):
    hypodd_df = loadHypoDDRelocFile()
    outputFile = f"xyzm_{outName}.dat"
    hypodd_df["year"] = hypodd_df.YR
    hypodd_df["month"] = hypodd_df.MO
    hypodd_df["day"] = hypodd_df.DY
    hypodd_df["hour"] = hypodd_df.HR
    hypodd_df["minute"] = hypodd_df.MI
    hypodd_df["second"] = hypodd_df.SC
    hypodd_df["ORT"] = to_datetime(hypodd_df[["year",
                                              "month",
                                              "day",
                                              "hour",
                                              "minute",
                                              "second"]])
    hypodd_df["ORT"] = hypodd_df["ORT"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    hypodd_df["Lon"] = hypodd_df.LON
    hypodd_df["Lat"] = hypodd_df.LAT
    hypodd_df["Dep"] = hypodd_df.DEPTH
    hypodd_df["Mag"] = hypodd_df.MAG
    hypodd_df["Nus"] = hypodd_df.NCTP
    hypodd_df["NuP"] = hypodd_df.NCTP
    hypodd_df["NuS"] = hypodd_df.NCTS
    hypodd_df["ADS"] = 0
    hypodd_df["MDS"] = 0
    hypodd_df["GAP"] = 0
    hypodd_df["RMS"] = hypodd_df.RCT
    hypodd_df["ERH"] = sqrt(hypodd_df.EX**2 + hypodd_df.EY**2)*1e-3
    hypodd_df["ERZ"] = hypodd_df.EZ*1e-3
    columns = ["ORT", "Lon", "Lat", "Dep", "Mag",
               "Nus", "NuP", "NuS", "ADS", "MDS", "GAP", "RMS", "ERH", "ERZ"]
    with open(outputFile, "w") as f:
        hypodd_df.to_string(f, columns=columns, index=False, float_format="%7.3f")


def getGap(evLat, evLon, arrivals, stationFile):
    station_df = read_csv(stationFile)
    station_df[["Azim"]] = station_df.apply(
        lambda x: Series(gps(evLat, evLon, x.lat, x.lon)[1]), axis=1)
    azimuths = station_df["Azim"].sort_values()
    gap = int(max(diff(azimuths)))
    return gap


def hypoDD2nordic(outName, stationFile):
    nordicCat = read_events(f"catalog_{outName}.out")
    hypoddCat = loadHypoDDRelocFile()
    xyzm_df = read_csv(f"xyzm_{outName}.dat", delim_whitespace=True)
    finalCat = Catalog()
    hypoddCat.SC.replace(60, 59.99, inplace=True)
    desc = f"+++ Converting hypoDD to NORDIC for {outName} ..."
    for i, hypoddEvent in tqdm(hypoddCat.iterrows(), desc=desc, unit=" event"):
        hypoddEventID = int(hypoddEvent.ID) - 1
        nordicEvent = nordicCat[hypoddEventID]
        preferred_origin = nordicEvent.preferred_origin()
        arrivals = preferred_origin.arrivals
        eOrt = utc(int(hypoddEvent.YR), int(hypoddEvent.MO), int(hypoddEvent.DY),
                   int(hypoddEvent.HR), int(hypoddEvent.MI), hypoddEvent.SC)
        eLat = hypoddEvent.LAT
        erLat = hypoddEvent.EY*1e-3
        eLon = hypoddEvent.LON
        erLon = hypoddEvent.EX*1e-3
        eDep = hypoddEvent.DEPTH
        erDep = hypoddEvent.EZ
        rms = hypoddEvent.RCT
        preferred_origin.time = eOrt
        preferred_origin.latitude = eLat
        preferred_origin.longitude = eLon
        preferred_origin.depth = eDep*1e3
        preferred_origin.latitude_errors.uncertainty = k2d(erLat)
        preferred_origin.longitude_errors.uncertainty = k2d(erLon)
        preferred_origin.depth_errors.uncertainty = erDep
        gap = getGap(eLat, eLon, arrivals, stationFile)
        preferred_origin.quality.azimuthal_gap = gap
        preferred_origin.quality.standard_error = rms
        finalCat.append(nordicEvent)
        xyzm_df.loc[i, "GAP"] = gap
    finalCat.write(f"hypodd_{outName}.out", format="nordic", high_accuracy=False)
    columns = ["ORT", "Lon", "Lat", "Dep", "Mag",
               "Nus", "NuP", "NuS", "ADS", "MDS", "GAP", "RMS", "ERH", "ERZ"]
    with open(f"xyzm_{outName}.dat", "w") as f:
        xyzm_df.to_string(f, columns=columns, index=False, float_format="%7.3f")
