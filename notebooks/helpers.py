import geopandas as gpd
from earthpy.clip import clip_shp
import os
from simpledbf import Dbf5
import numpy as np

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
    out_file_basename = os.path.basename(boundary_filepath)
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