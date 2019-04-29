import geopandas as gpd

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