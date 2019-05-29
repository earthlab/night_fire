import geopandas as gpd
from earthpy.clip import clip_shp
import os
from simpledbf import Dbf5
import numpy as np
from shapely.geometry import Point
from fiona.crs import from_epsg
import rasterio as rio
from rasterio import features

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


def create_global_agg_var_grid(shp_files, meta, agg=0.25, conf=0, out_folder='./'):
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

            # subset the dataframe by month and group by daynight
            df_sub = df.query('month == {}'.format(m))
            df_sub = df_sub.query('CONFIDENCE > {}'.format(conf))
            daynight = list(df.groupby('DAYNIGHT'))
            df_day = daynight[0][1]
            df_night = daynight[1][1]

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