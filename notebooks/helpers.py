import geopandas as gpd
from earthpy.clip import clip_shp
import os
from simpledbf import Dbf5
import numpy as np
from shapely.geometry import Point
from fiona.crs import from_epsg
import rasterio as rio
from rasterio import features
import pickle
import calendar
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob

from earthpy.spatial import stack
import xarray as xr

import pickle

# aggregating
from skimage.transform import rescale, resize, downscale_local_mean

# mapping
import cartopy
import cartopy.crs as ccrs

import warnings
warnings.filterwarnings('ignore')

def extract_day_night_counts(filepath, conf=50):
    
    
    #print('on file {} of {}'.format(i+1, len(shp_files)))
    pts_df = gpd.read_file(filepath)
    pts_df.query('CONFIDENCE > {}'.format(conf), inplace=True)
    
    
    daynight = list(pts_df.groupby(by='DAYNIGHT'))
    df_day = daynight[0][1]
    df_night = daynight[1][1]
    
    #print('hi', pts_df.ACQ_DATE[0])
    year = list(pts_df.ACQ_DATE)[0].split('-')[0]
    
    years = 'y' + year
    day_counts = df_day.shape[0]
    night_counts = df_night.shape[0]
    
    return years, day_counts, night_counts

def clip_points(points_filepath, boundary_filepath, out_folder=None, conf=50):
    
    pts_df = gpd.read_file(points_filepath)
    pts_df.query('CONFIDENCE > {}'.format(conf), inplace=True)
    
    bound_df = gpd.read_file(boundary_filepath)
    
    year = list(pts_df.ACQ_DATE)[0].split('-')[0]
    out_file_basename = os.path.basename(boundary_filepath)
    out_file_basename = out_file_basename.split('.')[0]
    out_filename = '{}_{}_pts_conf{}.shp'.format(out_file_basename, year, conf)
    
    if out_folder is None:
        out_fi = os.path.join(os.path.dirname(boundary_filepath), out_filename)
    else:
        out_fi = os.path.join(out_folder, out_filename)
    
    # clip the files
    sub_df = clip_shp(pts_df, bound_df)
    
    # save it
    sub_df.to_file(out_fi)
    
    return

def extract_frp_day_night_annual(filepath, conf=50, ftype='dbf', gtzero=True):
    
    if 'dbf' in ftype:
        dbf = Dbf5(filepath)
        pts_df = dbf.to_dataframe()
        
    else:
        pts_df = gpd.read_file(filepath)
    
    pts_df.query('CONFIDENCE > {}'.format(conf), inplace=True)
    if gtzero:
        pts_df.query('FRP > 0', inplace=True)
    
    daynight = list(pts_df.groupby(by='DAYNIGHT'))
    df_day = daynight[0][1]
    df_night = daynight[1][1]
    
    year=None
    if 'dbf' in ftype:
        year = int(list(pts_df.ACQ_DATE)[0].year)
    else:
        year = int(list(pts_df.ACQ_DATE)[0].split('-')[0])
        
    
    day_frp_mean = df_day.FRP.mean()
    day_frp_median = df_day.FRP.median()
    day_frp_std = df_day.FRP.std()
    day_frp_min = df_day.FRP.min()
    day_frp_max = df_day.FRP.max()
    day_frp_25 = np.percentile(df_day.FRP, 25)
    day_frp_75 = np.percentile(df_day.FRP, 75)
    day_frp_5 = np.percentile(df_day.FRP, 5)
    day_frp_95 = np.percentile(df_day.FRP, 95)
    
    night_frp_mean = df_night.FRP.mean()
    night_frp_median = df_night.FRP.median()
    night_frp_std = df_night.FRP.std()
    night_frp_min = df_night.FRP.min()
    night_frp_max = df_night.FRP.max()
    night_frp_25 = np.percentile(df_night.FRP, 25)
    night_frp_75 = np.percentile(df_night.FRP, 75)
    night_frp_5 = np.percentile(df_night.FRP, 5)
    night_frp_95 = np.percentile(df_night.FRP, 95)
    
    #return years, day_frp_mean, day_frp_std, day_frp_min, day_frp_max, night_frp_mean, night_frp_std, night_frp_min, night_frp_max
    return {'years': year, 
            'day_frp_mean': day_frp_mean,
            'day_frp_median': day_frp_median,
            'day_frp_std': day_frp_std, 
            'day_frp_min': day_frp_min, 
            'day_frp_max': day_frp_max,
            'day_frp_25': day_frp_25,
            'day_frp_75': day_frp_75,
            'day_frp_5': day_frp_5,
            'day_frp_95': day_frp_95,
            'night_frp_mean': night_frp_mean,
            'night_frp_median': night_frp_median,
            'night_frp_std': night_frp_std, 
            'night_frp_min': night_frp_min, 
            'night_frp_max': night_frp_max,
            'night_frp_25': night_frp_25,
            'night_frp_75': night_frp_75,
            'night_frp_5': night_frp_5,
            'night_frp_95': night_frp_95}

# boxplot code
def customized_box_plot(percentiles, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """

    box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs) 
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')

    for box_no, (q1_start, 
                 q2_start,
                 q3_start,
                 q4_start,
                 q4_end,
                 fliers_xy) in enumerate(percentiles):

        # Lower cap
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

        # Box
        box_plot['boxes'][box_no].set_ydata([q2_start, 
                                             q2_start, 
                                             q4_start,
                                             q4_start,
                                             q2_start])

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # Outliers
        if fliers_xy is not None and len(fliers_xy[0]) != 0:
            # If outliers exist
            box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
                                           ydata = fliers_xy[1])

            min_y = min(q1_start, min_y, fliers_xy[1].min())
            max_y = max(q4_end, max_y, fliers_xy[1].max())

        else:
            min_y = min(q1_start, min_y)
            max_y = max(q4_end, max_y)

        # The y axis is rescaled to fit the new box plot completely with 10% 
        # of the maximum value at both ends
        axes.set_ylim([min_y*1.1, max_y*1.1])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        ax.figure.canvas.draw()

    return box_plot

def clip_points_by_ecoregion(points_filepath, group, out_folder=None):
    
    eco_na1_name, eco_df = group
    pts_df = gpd.read_file(points_filepath)
    
    year = list(pts_df.ACQ_DATE)[0].split('-')[0]
    out_file_basename = os.path.basename(points_filepath)
    out_file_basename = out_file_basename.split('.')[0]
    eco_reg_name = eco_na1_name.replace(' ', '_')
    out_filename = '{}_{}_pts_{}.shp'.format(out_file_basename, year, eco_reg_name)
    
    if out_folder is None:
        out_fi = os.path.join(os.path.dirname(points_file_path), out_filename)
    else:
        out_fi = os.path.join(out_folder, out_filename)
    
    # clip the files
    sub_df = clip_shp(pts_df, eco_df)
    
    # save it
    sub_df.to_file(out_fi)
    
    return

def spatial_agg_point_df(gdf, agg_val=0.5, calc='mean', epsg=4326):
    ''' takes a geodata frame and aggregates to agg_val by calculation specified in calc'''
    
    # Make rounding function:
    def round_to_val(a, round_val):
        return np.round( np.array(a, dtype=float) / round_val) * round_val
    
    
    # Record the CRS epsg code of the incoming gdf
    if epsg is None:
        epsg_code = int(gdf.crs['init'].split(':')[1])
    else:
        epsg_code=epsg
    
    # Create the rounded coordinates
    gdf['lat_round'] = round_to_val(gdf['LATITUDE'].values, agg_val)
    gdf['lon_round'] = round_to_val(gdf['LONGITUDE'].values, agg_val)

    # Making dataframes and grouping stuff
    group_xy = gdf.groupby(['lon_round', 'lat_round'])

    # Calculating the value specified by calc
    if calc == 'mean':
        group_calc = group_xy.mean()
    elif calc == 'count':
        group_calc = group_xy.count()
    elif calc == 'std':
        group_calc = group_xy.std()
    elif calc == 'sum':
        group_calc = group_xy.sum()
    elif calc == 'max':
        group_calc = group_xy.max()
    else:
        raise ValueError('calc {} is not supported. Please use one of [mean, count, std, sum, max]. or... add it in!'.format(calc))
        
    # Introduce the geometry from the rounding
    group_calc['geometry'] = list(map(Point, list(group_calc.index)))
    
    # convert to geodataframe and assign crs
    group_calc = gpd.GeoDataFrame(group_calc)
    group_calc.crs = from_epsg(epsg_code)
    
    return group_calc


def create_global_agg_var_grid(shp_files, meta_file, agg=0.25, conf=1, type_code=0, out_folder='./'):
    ''' Here are the gridded fire data I think we want, with a question about total FRP that we can chat more about….
        # of nighttime active fire counts by month
        # of daytime active fire counts by month
        from a) and b) calculate the % of nighttime active fires/total active fires by month (I think you already have this one)
        mean nighttime FRP by month
        mean daytime FRP by month
        max nighttime FRP by month
        max daytime FRP by month
        total? nighttime FRP by month
        total? daytime FRP by month
'''
    var_list = ['AFC_num',
                'AFC_perc',
                'FRP_mean',
                'FRP_max',
                'FRP_total']
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    # load metadata
    with open(meta_file, 'rb') as fp:
        meta = pickle.load(fp)
        
    # update metadata dictionary as needed
    if agg != 0.25:
        from affine import Affine

        # restructure affine
        orig_size = meta['transform'].a
        agg_size = agg
        factor = agg_size/orig_size

        # calculate the new rows/cols
        new_height = int(np.floor_divide(meta['height'], factor))
        new_width = int(np.floor_divide(meta['width'], factor))

        # calculate the new center point for the upper left
        new_ul_x = meta['transform'].c - orig_size + agg_size/2
        new_ul_y =  meta['transform'].f + orig_size - agg_size/2

        # generate the new transform
        new_transform = Affine(agg_size, 0.0, new_ul_x, 
                               0.0, -agg_size, new_ul_y)

        # dictionary to update the metadata
        update_dict = {'height': new_height,
                       'width': new_width,
                      'transform': new_transform}

        meta.update(update_dict)
        
        print(agg, meta)
    
    else:
        print(agg, meta)
    
        
    # create some data for months info
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_no = list(range(1,13))
    
    # could probably farm out the shapefile
    for shp in shp_files:

        # read the file
        df = gpd.read_file(shp)

        # get month and year info
        df['month'] = [int(acq.split('-')[1]) for acq in df.ACQ_DATE]
        year = int(df.ACQ_DATE[0].split('-')[0])

        print('on year {}'.format(year))

        # iterate through the months
        for m in df['month'].unique():

            # get the month name for filenaming
            month_name = months[m-1]

            ### edit 9.17.2019 -- try-except due to type not existing in provisional data. continue...
            # subset the dataframe by month and group by daynight
            try:
            
                df_sub = df.query('month == {}'.format(m))
                df_sub = df_sub.query('CONFIDENCE > {}'.format(conf))
                df_sub = df_sub.query('TYPE == {}'.format(type_code))
                daynight = list(df_sub.groupby('DAYNIGHT'))
                df_day = daynight[0][1]
                df_night = daynight[1][1]
            
            except Exception as e:
                
                print(e)
                continue

            ##################
            ## AGGREGATIONS ##
            ##################
            
            # do the aggregation by count
            df_night_agg_count = spatial_agg_point_df(df_night, agg_val=agg, calc='count', epsg=4326)
            df_day_agg_count = spatial_agg_point_df(df_day, agg_val=agg, calc='count', epsg=4326)
            
            # do the aggregation by mean
            df_night_agg_mean = spatial_agg_point_df(df_night, agg_val=agg, calc='mean', epsg=4326)
            df_day_agg_mean = spatial_agg_point_df(df_day, agg_val=agg, calc='mean', epsg=4326)
            
            # do the aggregation by max
            df_night_agg_max = spatial_agg_point_df(df_night, agg_val=agg, calc='max', epsg=4326)
            df_day_agg_max = spatial_agg_point_df(df_day, agg_val=agg, calc='max', epsg=4326)
            
            # do the aggregation by sum
            df_night_agg_sum = spatial_agg_point_df(df_night, agg_val=agg, calc='sum', epsg=4326)
            df_day_agg_sum = spatial_agg_point_df(df_day, agg_val=agg, calc='sum', epsg=4326)

            
            # calculate the percentages per cell
            df_night_agg_count['count_perc'] = df_night_agg_count['FRP'] / (df_night_agg_count['FRP'] + df_day_agg_count['FRP'])
            df_night_agg_count['count_perc'] *= 100

            df_day_agg_count['count_perc'] = df_day_agg_count['FRP'] / (df_night_agg_count['FRP'] + df_day_agg_count['FRP'])
            df_day_agg_count['count_perc'] *= 100

            ###########################
            ## write out the rasters ##
            ###########################
            
            for var in var_list:
                
                print('writing out {} {} rasters'.format(month_name, var))
                
                # specify raster file names
                day_fname = 'modis_{}_{}_{}_{}.tif'.format('D', var, month_name, year)
                night_fname = 'modis_{}_{}_{}_{}.tif'.format('N', var, month_name, year)

                # dummy array for holding data (use meta['nodata'])
                out_arr = np.ones((meta['height'], meta['width'])).astype('float32') * float(meta['nodata'])
                
                #####################################
                ## assemble data frame for writing ##
                #####################################
                
                # percent AFC rasters
                if ('AFC_perc' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                    
                    day_arr = df_day_agg_count.dropna()
                    night_arr = df_night_agg_count.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.count_perc))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
                # num AFC rasters
                if ('AFC_num' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                        
                    day_arr = df_day_agg_count.dropna()
                    night_arr = df_night_agg_count.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.FRP))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
                # mean FRP rasters
                if ('FRP_mean' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                        
                    day_arr = df_day_agg_mean.dropna()
                    night_arr = df_night_agg_mean.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.FRP))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
                # max FRP rasters
                if ('FRP_max' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                        
                    day_arr = df_day_agg_max.dropna()
                    night_arr = df_night_agg_max.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.FRP))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
                # total FRP rasters
                if ('FRP_total' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                        
                    day_arr = df_day_agg_sum.dropna()
                    night_arr = df_night_agg_sum.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.FRP))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
                            
def create_global_agg_CONF_grid(shp_files, meta_file, agg=0.25, conf=1, type_code=0, out_folder='./'):
    ''' Here are the gridded fire data I think we want, with a question about total FRP that we can chat more about….
        # of nighttime active fire counts by month
        # of daytime active fire counts by month
        from a) and b) calculate the % of nighttime active fires/total active fires by month (I think you already have this one)
        mean nighttime FRP by month
        mean daytime FRP by month
        max nighttime FRP by month
        max daytime FRP by month
        total? nighttime FRP by month
        total? daytime FRP by month
'''
    var_list = ['CONF']
    
    # load metadata
    with open(meta_file, 'rb') as fp:
        meta = pickle.load(fp)
        
    if agg != 0.25:
        from affine import Affine

        # restructure affine
        orig_size = meta['transform'].a
        agg_size = agg
        factor = agg_size/orig_size
        factor

        # calculate the new rows/cols
        new_height = int(np.floor_divide(meta['height'], factor))
        new_width = int(np.floor_divide(meta['width'], factor))

        # calculate the new center point for the upper left
        new_ul_x = meta['transform'].c - orig_size + agg_size/2
        new_ul_y =  meta['transform'].f + orig_size - agg_size/2

        # generate the new transform
        new_transform = Affine(agg_size, 0.0, new_ul_x, 
                               0.0, -agg_size, new_ul_y)

        # dictionary to update the metadata
        update_dict = {'height': new_height,
                       'width': new_width,
                      'transform': new_transform}

        meta.update(update_dict)
    
    
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    
        
    # create some data for months info
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_no = list(range(1,13))
    
    # could probably farm out the shapefile
    for shp in shp_files:

        # read the file
        df = gpd.read_file(shp)

        # get month and year info
        df['month'] = [int(acq.split('-')[1]) for acq in df.ACQ_DATE]
        year = int(df.ACQ_DATE[0].split('-')[0])

        print('on year {}'.format(year))

        # iterate through the months
        for m in df['month'].unique():

            # get the month name for filenaming
            month_name = months[m-1]

            ### edit 9.17.2019 -- try-except due to type not existing in provisional data. continue...
            # subset the dataframe by month and group by daynight
            try:
            
                df_sub = df.query('month == {}'.format(m))
                df_sub = df_sub.query('CONFIDENCE > {}'.format(conf))
                df_sub = df_sub.query('TYPE == {}'.format(type_code))
                daynight = list(df_sub.groupby('DAYNIGHT'))
                df_day = daynight[0][1]
                df_night = daynight[1][1]
            
            except Exception as e:
                
                print(e)
                continue

            ##################
            ## AGGREGATIONS ##
            ##################
            
            # do the aggregation by count
            #df_night_agg_count = spatial_agg_point_df(df_night, agg_val=agg, calc='count', epsg=4326)
            #df_day_agg_count = spatial_agg_point_df(df_day, agg_val=agg, calc='count', epsg=4326)
            
            # do the aggregation by mean
            df_night_agg_mean = spatial_agg_point_df(df_night, agg_val=agg, calc='mean', epsg=4326)
            df_day_agg_mean = spatial_agg_point_df(df_day, agg_val=agg, calc='mean', epsg=4326)
            
            # do the aggregation by max
            #df_night_agg_max = spatial_agg_point_df(df_night, agg_val=agg, calc='max', epsg=4326)
            #df_day_agg_max = spatial_agg_point_df(df_day, agg_val=agg, calc='max', epsg=4326)
            
            # do the aggregation by sum
            #df_night_agg_sum = spatial_agg_point_df(df_night, agg_val=agg, calc='sum', epsg=4326)
            #df_day_agg_sum = spatial_agg_point_df(df_day, agg_val=agg, calc='sum', epsg=4326)

            
           
            ###########################
            ## write out the rasters ##
            ###########################
            
            for var in var_list:
                
                print('writing out {} {} rasters'.format(month_name, var))
                
                # specify raster file names
                day_fname = 'modis_{}_{}_{}_{}.tif'.format('D', var, month_name, year)
                night_fname = 'modis_{}_{}_{}_{}.tif'.format('N', var, month_name, year)

                # dummy array for holding data (use meta['nodata'])
                out_arr = np.ones((meta['height'], meta['width'])).astype('float32') * float(meta['nodata'])
                
                #####################################
                ## assemble data frame for writing ##
                #####################################
                
                                            
                # mean CONFIDENCE rasters
                if ('CONF' in day_fname):
                    out_folder_var = os.path.join(out_folder, var)
                    if not os.path.exists(out_folder_var):
                        os.makedirs(out_folder_var)
                        
                    day_arr = df_day_agg_mean.dropna()
                    night_arr = df_night_agg_mean.dropna()
                    for fname,df_2_write in zip((day_fname, night_fname), (day_arr, night_arr)):

                        out_fn = os.path.join(out_folder_var, fname)
                        with rio.open(out_fn, 'w', **meta) as out:

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom,value) for geom, value in zip(df_2_write.geometry, df_2_write.CONFIDENCE))

                            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                            out.write_band(1, burned)
                            
def lag_linregress_3D(x, y, lagx=0, lagy=0, n_obs=3):
    """
    https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three 
    dimensions (time,lat,lon). 
    
    Datasets can be provided in any order, but note that the regression slope 
    and intercept will be calculated for y with respect to x.
    
    
    Output: Covariance, correlation, regression slope and intercept, p-value, 
    and standard error on regression between the two datasets along their 
    aligned time dimension. 
    
    Lag values can be assigned to either of the data, with lagx shifting x, and
    lagy shifting y, with the specified lag amount. 
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)

    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:

        # If x lags y by 1, x must be shifted 1 step backwards. 
        # But as the 'zero-th' value is nonexistant, xr assigns it as invalid 
        # (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time') 

        # Next important step is to re-align the two datasets so that y adjusts
        # to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)

    #3. Compute data length, mean and standard deviation along time axis: X (1-D time) needs to be subset by non-NaN in Y
    n = y.notnull().sum(dim='time')
    n = xr.where(n<n_obs, np.nan, n) # pixels with less than n_obs set to NaN
    
    # retile 1-D array (assume X is time vector)
    x = np.expand_dims(np.expand_dims(x,-1), -1)
    x = np.tile(x, (1,y.shape[1], y.shape[2]))
    x = xr.where(y.notnull(), x, np.nan)
    
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)

    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)

    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)

    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  

    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats

    from scipy.stats import t
    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cov,cor,slope,intercept,pval,stderr,n

def plot_regress_var(day_data, night_data, folder, af_var='FRP_total', reg_month='August', reg_var='slope', n_obsv=7, agg_fact=1, absmin=None, absmax=None, cm='jet', save=False, save_dir=None, cartoplot=False, norm=False, min_year=None,base_year=2003):
    
    if (min_year is not None) and (min_year not in np.arange(2000,2019)):
        raise ValueError("min_year must be valid")
    
    var = af_var
    month= reg_month
    raster_folder = folder
    day_rasters = glob(raster_folder + '{}/*_D_*{}*.tif'.format(var, month))
#     night_rasters = glob(raster_folder + '{}/*_N_*{}*.tif'.format(var, month))
#     years = [int(os.path.basename(d).split('.')[0].split('_')[-1]) for d in day_rasters]

#     test_day = xr.DataArray(stack(day_rasters, nodata=0)[0], dims=['time', 'lat', 'lon'])
#     test_night = xr.DataArray(stack(night_rasters, nodata=0)[0], dims=['time', 'lat', 'lon'])
#     test_years = xr.DataArray(np.arange(2001,2001+test_night.shape[0]), dims=['time'])
    
    test_day = xr.DataArray(day_data, dims=['time', 'lat', 'lon'])
    test_night = xr.DataArray(night_data, dims=['time', 'lat', 'lon'])
    test_years = xr.DataArray(np.arange(base_year,base_year+test_night.shape[0]))
    
    if norm:
        # try to normalize the data
        day_max = test_day.max()
        day_min = test_day.min()
        night_max = test_night.max()
        night_min = test_night.min()
        tot_min = min(night_min, day_min)
        tot_max = max(night_max, day_max)
        test_day = (test_day - tot_min) / (tot_max - tot_min)
        test_night = (test_night - tot_min) / (tot_max - tot_min)
        test_years -= base_year # normalize
        
    if min_year is not None:
        
        # get index for min_year
        if norm:
            min_ind = min_year - base_year
            test_years = test_years[min_ind:]
            test_day = test_day[min_ind:,:,:]
            test_night = test_night[min_ind:,:,:]
        
        else:
            min_ind = np.where(test_years == min_year)[0][0]
            test_years = test_years[min_ind:]
            test_day = test_day[min_ind:,:,:]
            test_night = test_night[min_ind:,:,:]
    
    print(f'test_years: {test_years.max()} {test_years.min()}')
    d_cov, d_cor, d_slope, d_intercept, d_pval, d_stderr, d_n = lag_linregress_3D(test_years, test_day, n_obs=n_obsv)
    n_cov, n_cor, n_slope, n_intercept, n_pval, n_stderr, n_n = lag_linregress_3D(test_years, test_night, n_obs=n_obsv)
    
    print(f'day max: {d_slope.max().values}, night max: {n_slope.max().values}')
    print(f'day min: {d_slope.min().values}, night min: {n_slope.min().values}')
    #print(test_years, np.nanmax(d_slope), np.nanmin(d_slope), np.nanmax(n_slope), np.nanmin(n_slope))
    
    if (absmin is None) or (absmax is None):
        absmin = min(n_slope.min(), d_slope.min())
        absmax = min(n_slope.max(), d_slope.max())
    
    
    night_plot_im = np.ma.masked_equal(n_slope,0)
    day_plot_im = np.ma.masked_equal(d_slope,0)


    # aggregate?
    agg_fac = agg_fact # agg_fac*0.25... 4 == 1 deg
    if agg_fac > 1:
        night_resized = resize(night_plot_im, (int(night_plot_im.shape[0] / agg_fac), int(night_plot_im.shape[1] / agg_fac)), anti_aliasing=True)
    else:
        night_resized = night_plot_im
    night_resized = np.ma.masked_equal(night_resized,0)

    if agg_fac > 1:
        day_resized = resize(day_plot_im, (int(day_plot_im.shape[0] / agg_fac), int(day_plot_im.shape[1] / agg_fac)), anti_aliasing=True)
    else:
        day_resized = day_plot_im
    day_resized = np.ma.masked_equal(day_resized,0)

    if not cartoplot:
        fig, ax = plt.subplots(2,1, figsize=(30,20))
        ax[0].imshow(day_resized, cmap=cm, vmin=absmin, vmax=absmax)
        ax[0].set_title('daytime slope for {} 2001-2019, at least {} data points'.format(month, n_obsv))
        im=ax[1].imshow(night_resized, cmap=cm, vmin=absmin, vmax=absmax)
        ax[1].set_title('nighttime slope for {} 2001-2019, at least {} data points'.format(month, n_obsv))

        fig.subplots_adjust(right=0.96)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if save:
            save_file = os.path.join(save_dir, '{}_{}_slope_aggfact{}.png'.format(var, month, agg_fact))
            plt.savefig(save_file)

        plt.show()
        
    else:
        plot_carto_var(folder, day_resized, var, month, 'D', n_obsv, cm, absmin, absmax, agg_fact, save_dir=save_dir, base_year=base_year)
        plot_carto_var(folder, night_resized, var, month, 'N', n_obsv, cm, absmin, absmax, agg_fact, save_dir=save_dir, base_year=base_year)
        
        # plot some others to make sure the slopes are note exactly equal
        slope_title='{} daynight slope ratio (night / day)'.format(month)
        plot_carto_check(folder, night_resized/day_resized, cm, vmin=-2., vmax=2., agg_fact=agg_fact, save=False, title=slope_title, slopenan=True)
        
        dif_title = '{} daynight slope difference (night - day) (x100)'.format(month)
        plot_carto_check(folder, (night_resized - day_resized)*100, cm, vmin=-1., vmax=1., agg_fact=agg_fact, save=False, title=dif_title)
        
    
    return day_resized, night_resized, night_resized/day_resized, (night_resized - day_resized)
        
def plot_carto_var(raster_folder, data, var, month, daynight, n_obsv, cmap, vmin, vmax, agg_fact, save=True, save_dir=None, title=None, base_year=None):
    
    
    # get coords
    ## try for my data
    template = glob(raster_folder + '{}/*_D_*{}*.tif'.format('AFC_num', 'April'))[0]
    with rio.open(template) as src:
        meta = src.meta

    tform = meta['transform']
    #num_x = meta['width']
    #num_y = meta['height']
    
    num_x = data.shape[1]
    num_y = data.shape[0]

    # incorporate aggregation factor
    tlon = np.linspace(tform.c - tform.a*agg_fact, tform.c+num_x*tform.a*agg_fact, num_x)
    tlat = np.linspace(tform.f - tform.e*agg_fact, tform.f+num_y*tform.e*agg_fact, num_y)
    lon2d, lat2d = np.meshgrid(tlon, tlat)

    
    # make data into xarray with location
    xdata = xr.DataArray(data, coords=[tlat, tlon], dims=['lat', 'lon'])
    xdata = xr.where(xdata == 0, np.nan, xdata)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    xdata.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    
    # get title
    if daynight=='D':
        title = '{} daytime slope for {} {}-2019, at least {} data points'.format(var, month, base_year, n_obsv)
    elif daynight=='N':
        title = '{} nighttime slope for {} {}-2019, at least {} data points'.format(var, month, base_year, n_obsv)
        
    plt.title(title)
    
    if save:
        save_file = os.path.join(save_dir, '{}_{}_slope_aggfact{}_dn{}.png'.format(var, month, agg_fact, daynight))
        plt.savefig(save_file)
        
    plt.show()
    
def plot_carto_check(raster_folder, data, cmap, vmin, vmax, agg_fact, save=True, save_dir=None, title=None, slopenan=False):
    
    # get coords
    ## try for my data
    template = glob(raster_folder + '{}/*_D_*{}*.tif'.format('AFC_num', 'April'))[0]
    with rio.open(template) as src:
        meta = src.meta

    tform = meta['transform']
    #num_x = meta['width']
    #num_y = meta['height']
    
    num_x = data.shape[1]
    num_y = data.shape[0]

    # incorporate aggregation factor
    tlon = np.linspace(tform.c - tform.a*agg_fact, tform.c+num_x*tform.a*agg_fact, num_x)
    tlat = np.linspace(tform.f - tform.e*agg_fact, tform.f+num_y*tform.e*agg_fact, num_y)
    lon2d, lat2d = np.meshgrid(tlon, tlat)

    
    
    xdata = xr.DataArray(data, coords=[tlat, tlon], dims=['lat', 'lon'])
    xdata = xr.where(xdata == 0, np.nan, xdata)
    
    if slopenan:
        one_buffer = 0.01
        print('setting ratio==1.0 (+/- {}) to nan'.format(one_buffer))
        #xdata = xr.where(xdata==1.0, np.nan, xdata)
        xdata = xr.where((xdata<=1.0+one_buffer) & (xdata>=1.0-one_buffer), np.nan, xdata)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    xdata.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.title(title)
    
    if save:
        save_file = os.path.join(save_dir, '{}_{}_slope_aggfact{}_dn{}.png'.format(var, month, agg_fact, daynight))
        plt.savefig(save_file)
        
    plt.show()
    
def remap_months3(month_arr):
    
    # 1=DJF, 2=MAM, 3=JJA, 4=SON
    month_arr1 = month_arr.copy()
    
    month_arr1[month_arr==12] = 1
    month_arr1[month_arr==1] = 1
    month_arr1[month_arr==2] = 1
    
    month_arr1[month_arr==3] = 2
    month_arr1[month_arr==4] = 2
    month_arr1[month_arr==5] = 2
    
    month_arr1[month_arr==6] = 3
    month_arr1[month_arr==7] = 3
    month_arr1[month_arr==8] = 3
    
    month_arr1[month_arr==9] = 4
    month_arr1[month_arr==10] = 4
    month_arr1[month_arr==11] = 4
    
    return month_arr1

def plot_max_month(argm_data, raster_folder, title=None):
    
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['#1a68d6', # January (dark blue)
                                  '#5713bd', # February (purplish)
                                  '#8ee3fa', # March (light blue)
                                  '#e3df10', # April (yellow)
                                  '#fa6969', # May (light red)
                                   '#d8e3e1', # June (off white)
                                   '#a30000', # July (dark red)
                                   '#74c477', # August (light green)
                                   '#8a4300', # September (dark brown)
                                   '#fa7d07', # October (brighter orange)
                                   '#006c75', # November (darker tealish)
                                   '#00ebff']) # December (light blue)

    tick_labels = ['Jan',
                  'Feb',
                  'Mar',
                  'Apr',
                  'May',
                  'Jun',
                  'Jul',
                  'Aug',
                  'Sep',
                  'Oct',
                  'Nov',
                  'Dec']

    bounds=[0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)


    data=argm_data
    agg_fact=1
    template = glob(raster_folder + '{}/*_D_*{}*.tif'.format('AFC_num', 'April'))[0]
    with rio.open(template) as src:
        meta = src.meta

    tform = meta['transform']
    #num_x = meta['width']
    #num_y = meta['height']

    num_x = data.shape[1]
    num_y = data.shape[0]

    # incorporate aggregation factor
    tlon = np.linspace(tform.c - tform.a*agg_fact, tform.c+num_x*tform.a*agg_fact, num_x)
    tlat = np.linspace(tform.f - tform.e*agg_fact, tform.f+num_y*tform.e*agg_fact, num_y)
    lon2d, lat2d = np.meshgrid(tlon, tlat)


    # make data into xarray with location
    xdata = xr.DataArray(data, coords=[tlat, tlon], dims=['lat', 'lon'])
    xdata = xr.where(xdata == 0, np.nan, xdata)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    img = xdata.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.array(bounds)-0.5)
    cbar.ax.set_yticklabels(tick_labels)
    plt.title(title)
    plt.show()
    
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

def plot_max_month_groups(argm_data, raster_folder, title=None):
    
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['#1a68d6', # DJF (dark blue)
                                  '#fae311', # MAM (light orange)
                                  '#07db7f', # JJA (green)
                                  '#d95904']) # SON (red-orange)

    tick_labels = ['Winter (DJF)',
                  'Spring (MAM)',
                  'Summer (JJA)',
                  'Autumn (SON)']

    bounds=[0.5,1.5,2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)


    data=argm_data
    agg_fact=1
    template = glob(raster_folder + '{}/*_D_*{}*.tif'.format('AFC_num', 'April'))[0]
    with rio.open(template) as src:
        meta = src.meta

    tform = meta['transform']
    #num_x = meta['width']
    #num_y = meta['height']

    num_x = data.shape[1]
    num_y = data.shape[0]

    # incorporate aggregation factor
    tlon = np.linspace(tform.c - tform.a*agg_fact, tform.c+num_x*tform.a*agg_fact, num_x)
    tlat = np.linspace(tform.f - tform.e*agg_fact, tform.f+num_y*tform.e*agg_fact, num_y)
    lon2d, lat2d = np.meshgrid(tlon, tlat)


    # make data into xarray with location
    xdata = xr.DataArray(data, coords=[tlat, tlon], dims=['lat', 'lon'])
    xdata = xr.where(xdata == 0, np.nan, xdata)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    img = xdata.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.array(bounds)-0.5)
    cbar.ax.set_yticklabels(tick_labels)
    plt.title(title)
    plt.show()
    
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

def gen_plot_xarr(argm_data, raster_folder, title=None, cmap='jet'):
    
    data=argm_data
    agg_fact=1
    template = glob(raster_folder + '{}/*_D_*{}*.tif'.format('AFC_num', 'April'))[0]
    with rio.open(template) as src:
        meta = src.meta

    tform = meta['transform']
    #num_x = meta['width']
    #num_y = meta['height']

    num_x = data.shape[1]
    num_y = data.shape[0]

    # incorporate aggregation factor
    tlon = np.linspace(tform.c - tform.a*agg_fact, tform.c+num_x*tform.a*agg_fact, num_x)
    tlat = np.linspace(tform.f - tform.e*agg_fact, tform.f+num_y*tform.e*agg_fact, num_y)
    lon2d, lat2d = np.meshgrid(tlon, tlat)


    # make data into xarray with location
    xdata = xr.DataArray(data, coords=[tlat, tlon], dims=['lat', 'lon'])
    xdata = xr.where(xdata == 0, np.nan, xdata)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    img = xdata.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=True)
    plt.title(title)
    plt.show()
    
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image
    
    
def get_fire_year_files(raster_folder, var, year):
    
    yr_mos = ['March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    yr_mos_p1 = ['January', 'February']

    # extract rasters for a single fire year, defined as March to February of following year.
    day_files = glob(raster_folder + f'{var}/*_D_*_{year}*')
    day_files = [f for f in day_files if os.path.basename(f).split('_')[4] in yr_mos]
    day_files2 = glob(raster_folder + f'{var}/*_D_*_{year+1}*')
    day_files2 = [f for f in day_files2 if os.path.basename(f).split('_')[4] in yr_mos_p1]
    day_files_yr = day_files + day_files2

    # night_files
    night_files = glob(raster_folder + f'{var}/*_N_*_{year}*')
    night_files = [f for f in night_files if os.path.basename(f).split('_')[4] in yr_mos]
    night_files2 = glob(raster_folder + f'{var}/*_N_*_{year+1}*')
    night_files2 = [f for f in night_files2 if os.path.basename(f).split('_')[4] in yr_mos_p1]
    night_files_yr = night_files + night_files2
    
    month_vals = dict((v,k) for k,v in enumerate(calendar.month_name))
    month_names = [os.path.basename(f).split('_')[4] for f in day_files_yr]

    # get the way things are sorted
    cur_sort = [month_vals[m] for m in month_names] 
    cur_sort_inds = np.array(cur_sort)

    # should be sorted as 
    fy_sort = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
    fy_sort_inds = np.array(fy_sort)

    if len(fy_sort) > len(cur_sort):
        # this would happen if the current year data is provisional and doesn't have the fields!
        fy_sort = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        fy_sort_inds = np.array(fy_sort)
    
    temp_df = pd.DataFrame({'files': day_files_yr, 'cur_ind':cur_sort_inds, 'fy_ind': fy_sort_inds, 'dummy': 'blah'})

    # sort correctly for FY
    df1 = temp_df.set_index('cur_ind')
    df1 = df1.reindex(index=fy_sort_inds)
    df1.reset_index();

    temp_df = pd.DataFrame({'files': night_files_yr, 'cur_ind':cur_sort_inds, 'fy_ind': fy_sort_inds, 'dummy': 'blah'})

    # sort correctly for FY
    df2 = temp_df.set_index('cur_ind')
    df2 = df2.reindex(index=fy_sort_inds)
    df2.reset_index();
    
    return df1.files, df2.files

def nanargmax(files, nodataval=-32768):

    # stack one
    test_arr,_ = stack(files, nodata=nodataval)

    # mask the nodata
    ma = np.ma.masked_equal(test_arr, nodataval)
    #ma = np.ma.filled(ma, np.nan)
    all_na_mask = np.any(ma, axis=0)

    # get the argmax
    argm = np.argmax(test_arr, axis=0) + 1
    argm = np.ma.masked_less(argm, -np.inf)
    argm.mask = ~all_na_mask
    
    return argm

def nansum(files, nodataval=-32768):

    # stack one
    test_arr,_ = stack(files, nodata=nodataval)

    nsum = np.nansum(test_arr, axis=0)
    
    
    return nsum
    