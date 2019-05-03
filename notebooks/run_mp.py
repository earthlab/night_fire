import pandas as pd
import numpy as np
import os, sys
from matplotlib import pyplot as plt
from glob import glob

from helpers import extract_day_night_counts
from functools import partial
import multiprocessing as mp
import time

if __name__ == '__main__':
    # fire_pts_folders = !ls -d ../modis_fire_points/*
    fire_pts_folders = glob('../modis_fire_points/*')
    shp_files = [glob(folder + '/*.shp')[0] for folder in fire_pts_folders]

    test=[]

    t0 = time.time()
    num_cpu = mp.cpu_count() - 2
    num_cpu = 3
    pool = mp.Pool(processes=num_cpu)
    a = pool.map(partial(extract_day_night_counts, conf=50), shp_files)

    test.append(a)

    pool.close()
    pool.join()
    t1 = time.time()

    print_time = (t1-t0)/60
    print('time: {} minutes'.format(print_time))

    import pickle
    with open('mp_result.pickle', 'wb') as f:
        pickle.dump(a, f)