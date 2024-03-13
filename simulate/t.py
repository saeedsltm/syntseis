import os
import sys
from pathlib import Path
from string import ascii_letters as as_le

from numpy import arange, meshgrid
from numpy.random import RandomState
from obspy.geodetics.base import kilometers2degrees as k2d
from pandas import DataFrame, Series, read_csv
from pyproj import Proj
from yaml import FullLoader, dump, load
from obspy.geodetics.base import gps2dist_azimuth as gps

station_db = read_csv("extra_sta.csv")
clat = 38.25
clon = 46.75
proj = Proj(f"+proj=sterea\
        +lon_0={clon}\
        +lat_0={clat}\
        +units=km")
station_db["elv"] *= 1e-3
station_db[["x", "y"]] = station_db.apply(
    lambda x: Series(
        proj(longitude=x.lon, latitude=x.lat)), axis=1)
station_db["z"] = station_db["elv"]
station_db[["r"]] = station_db.apply(
    lambda x: Series(
        gps(clat, clon, x.lat, x.lon)[0]*1e-3), axis=1)
station_db.to_csv("extra_sta_.csv", index=False)
