#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:20:37 2024

@author: Basile Dupont
"""

import math
import random
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

# Set model if using TauP
TauP_model = TauPyModel(model="ak135")  # use Kennett 2005 propagation model

# If not using TauP
# Earth parameters
R_Earth = 6371e3
Vp = 8000               # vitesse onde P en m/s
Vs = Vp / math.sqrt(3)  # vitesse onde S en m/s

# Source coordinates
def generate_coordinates(depth=None):
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    if depth is None:
        depth = random.uniform(0, 100e3)
    return latitude, longitude, depth

# Angular distance (surface)
def to_rad(deg):
    return deg * math.pi / 180
    
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(to_rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * math.asin(math.sqrt(a))

# Chord distance (depth)
def chord_distance(delta):
    return 2 * R_Earth * math.sin(delta / 2)

# Direct distance
def to_cartesian(lat, lon, depth):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    r = R_Earth - depth
    x = r * math.cos(lat_rad) * math.cos(lon_rad)
    y = r * math.cos(lat_rad) * math.sin(lon_rad)
    z = r * math.sin(lat_rad)
    return x, y, z

def direct_distance(lat1, lon1, dep1, lat2, lon2, dep2):
    x1, y1, z1 = to_cartesian(lat1, lon1, dep1)
    x2, y2, z2 = to_cartesian(lat2, lon2, dep2)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Generate valid station (30-90°)
def generate_station(lat_epi, lon_epi):
    while True:
        # Génère une station aléatoire
        lat_station = random.uniform(-90, 90)
        lon_station = random.uniform(-180, 180)
        delta = haversine(lat_epi, lon_epi, lat_station, lon_station)
        delta_degrees = math.degrees(delta)

        if 30 <= delta_degrees <= 90:
            return lat_station, lon_station

# Compute travel times
def travel_times(lat1, lon1, dep1, lat2, lon2):
    # P arrival time
    P_distance = direct_distance(lat1, lon1, dep1, lat2, lon2, 0)
    tP = P_distance / Vp

    # pP and sP distance
    delta = haversine(lat1, lon1, lat2, lon2)
    depth_distance = chord_distance(delta)

    # pP arrival time
    t_hypo_epi = dep1 / Vp  # Hypocenter to epicenter
    t_epi_station = depth_distance / Vp  # Epicenter to station
    tpP = t_hypo_epi + t_epi_station

    # sP arrival time
    t_hypo_epi_S = dep1 / Vs  # Hypocenter to epicenter
    t_epi_station_P = depth_distance / Vp  # Epicenter to station
    tsP = t_hypo_epi_S + t_epi_station_P

    return tP, tpP, tsP

# Generate samples
def generate_arrival_samples(num_stations=50, depth=None, use_TauP=False):
    source = generate_coordinates(depth=depth)
    deltas = []
    stations = []
    
    for _ in range(num_stations):
        # Generate station
        station = generate_station(source[0], source[1])
        
        if use_TauP:
            distance_deg = locations2degrees(lat1=source[0],
                                             long1=source[1],
                                             lat2=station[0],
                                             long2=station[1]
                                            )
    
            arrivals = TauP_model.get_travel_times(source_depth_in_km=source[2]/1e3,
                                              distance_in_degree=distance_deg,
                                              phase_list=["P", "pP", "sP"]
                                             )

            if source[2] == 0:  # if source is at depth 0, no pP or sP so delta is 0
                delta_pP, delta_sP = 0, 0
            else:
                delta_pP = arrivals[1].time - arrivals[0].time
                delta_sP = arrivals[2].time - arrivals[0].time
            
        else:
            tP, tpP, tsP = travel_times(*source, station[0], station[1])
            delta_pP, delta_sP = tpP - tP, tsP - tP
            
        deltas.append((delta_pP, delta_sP))
        stations.append(station)
    
    return deltas, source, stations

