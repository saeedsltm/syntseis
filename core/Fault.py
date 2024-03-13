from numpy import (arange, array, cos, deg2rad, dot, mean, meshgrid, ones_like,
                   sin, sqrt)
from numpy.random import RandomState
from obspy.geodetics.base import degrees2kilometers as d2k
from pandas import DataFrame


def distributeEqOnFault(config, bound, nEvents, gaussianDist, xVar, yVar,
                        minEventSpacing, depth, strike, dip, rndID):
    rng = RandomState(rndID)
    hypocenter_db = {
        "Longitude": [],
        "Latitude": [],
        "Depth": []}
    lonMin, lonMax, latMin, latMax = bound
    if gaussianDist:
        Mean = [mean([lonMin, lonMax]), mean([latMin, latMax])]
        Cov = [[xVar, 0], [0, yVar]]
        xx, yy = rng.multivariate_normal(Mean, Cov, nEvents).T
        zz = ones_like(xx)
    else:
        Lons = arange(lonMin, lonMax, minEventSpacing)
        Lats = arange(latMin, latMax, minEventSpacing)
        Deps = ones_like(Lons)
        xx, yy, zz = meshgrid(Lons, Lats, Deps)
    strike = deg2rad(strike)
    dip = deg2rad(dip)
    R_y = array([[cos(dip), 0, sin(dip)],
                 [0, 1, 0],
                 [-sin(dip), 0, cos(dip)]])
    R_z = array([[cos(strike), -sin(strike), 0],
                 [sin(strike), cos(strike), 0],
                 [0, 0, 1]])
    for point in range(xx.size):
        xx.ravel()[point], yy.ravel()[point], zz.ravel()[point] = dot(
            array(
                [xx.ravel()[point],
                 yy.ravel()[point],
                 zz.ravel()[point]]).T, R_y)
        xx.ravel()[point], yy.ravel()[point], zz.ravel()[point] = dot(
            array([xx.ravel()[point],
                   yy.ravel()[point],
                   zz.ravel()[point]]).T, R_z)
    zz = d2k(zz)
    xx -= mean(xx) - mean([lonMin, lonMax])
    yy -= mean(yy) - mean([latMin, latMax])
    zz -= mean(zz) - depth
    xx += rng.normal(0, xx.std()*5e-2, xx.shape)
    yy += rng.normal(0, yy.std()*5e-2, yy.shape)
    zz += rng.normal(0, zz.std()*5e-2, zz.shape)
    idx = rng.choice(arange(xx.size), size=nEvents, replace=False)
    hypocenter_db["Longitude"].extend(xx.ravel()[idx].tolist())  # type: ignore
    hypocenter_db["Latitude"].extend(yy.ravel()[idx].tolist())
    hypocenter_db["Depth"].extend(zz.ravel()[idx].tolist())
    return DataFrame(hypocenter_db)
