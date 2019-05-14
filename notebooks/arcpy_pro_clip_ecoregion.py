import arcpy
import os
from glob import glob

def get_year(some_file):
    
    with arcpy.da.SearchCursor(some_file, ['ACQ_DATE']) as sc:
        for row in sc:
            #year = row[0].split('-')[0]
            year = row[0].year
            break
    return year

def crop_file(in_layer, pts_file, out_folder, desc=''):
    
    year = get_year(pts_file)
    out_file = os.path.basename(pts_file).split('.')[0]
    out_file = '{}_{}_{}.shp'.format(out_file, year, desc.replace(' ', '_'))
    out_file = os.path.join(out_folder, out_file)
    arcpy.Clip_analysis(pts_file, in_layer, out_file)
    
    return
    
    
    

output_folder = '../epa_L1_ecoregion_points'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

p = '../modis_fire_points'
shp_files = glob(p + '/*/*.shp')


# select western forests layer
na1 = 'NORTHWESTERN FORESTED MOUNTAINS'
eco_shp = '../ecoregions_L1_CONUS.shp'
arcpy.MakeFeatureLayer_management(eco_shp, 'lyr', where_clause=""" "NA_L1NAME" = '{}'""".format(na1))
#arcpy.MakeFeatureLayer_management(eco_shp, 'lyr', where_clause=""" "NA_L1CODE" = '6' """)

# crop each year
for i,fi in enumerate(shp_files):
    
    arcpy.AddMessage('on file {} of {}'.format(i+1, len(shp_files)))
    crop_file('lyr', fi, output_folder, na1)
    
    