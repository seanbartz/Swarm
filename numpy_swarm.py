#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 22:29:34 2022

@author: seanbartz
"""
from matplotlib import pyplot as plt

import numpy as np
import scipy.spatial
fig, ax = plt.subplots()


# 3-D locations of particles
#positions = np.array([[1,0,0],[2,1,0],[2,2,0]], dtype=float)
N=6
positions=np.random.rand(N,3)
velocities =np.random.rand(N,3)-.5
# velocities = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float)
masses=np.random.rand(N).T
J=1

endtime=6
numtime=100*endtime

time = np.linspace(0, endtime,numtime)
dt = np.ediff1d(time).mean()


for i, t in enumerate(time):

    "symmetric matrix, where entry distanceMatrix[i,j] is the (scalar) distance between objects i and j"
    distanceMatrix=scipy.spatial.distance.cdist(positions, positions)
    
    
    "distanceVectorMatrix[i] is an Nx3 matrix where row j is the distance vector between i and j"
    distanceVectorMatrix=(positions.T[...,np.newaxis]-positions.T[:,np.newaxis]).T
    
    "divide by magnitudes to get unit vecttors"
    rhatMatrix=(distanceVectorMatrix.T / distanceMatrix).T
    rhatMatrix[np.isnan(rhatMatrix)] = 0 # replace NaN values with 0 for entries that were 0/0
    
    massMatrix=np.outer(masses,masses)
    
    "calculate force magnitudes. Symmetric matrix where force[i,j] is the magnitue of the force between objects i and j"
    force = J * massMatrix * distanceMatrix
    
    " calculate acceleration in all dimensions"
    acc = (rhatMatrix.T * force / masses).T.sum(axis=1)
    
    "Euler-Cromer method"
    velocities += acc *dt
    positions += velocities*dt
    
    # plotting
    if not i:
        color = 'k'
        zorder = 3
        ms = 3

    elif i == len(time)-1:
        color = 'b'
        zroder = 3
        ms = 3
    else:
        color = 'r'
        zorder = 1
        ms = 1
    ax.plot(positions[:,0], positions[:,1], '.', color=color, ms=ms, zorder=zorder)

ax.set_aspect('equal')