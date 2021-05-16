# -*- coding: utf-8 -*-

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This module implements functions for plotting geographical coordinates on a map
using matplotlib. Tiles are currently hard coded to be fetched from api.mapbox.com
and retina tiles are used. This is motivated by use in print. Please consult
the Mapbox help if you like to distribute their content in any way. The author 
of this script package does not take responsibility for any copyright infringements
that the result of using this package. 
"""

__author__ = "Sebastian van de Hoef"

import matplotlib.pyplot as plt
import numpy as np
import http.client as httplib
from io import BytesIO

def _web_mercator(lat,lon):
  # web mercator
  
  lat = lat/180.*np.pi
  lon = lon/180.*np.pi
    
  x = 128./np.pi*(lon+np.pi)
  y = 128./np.pi*(np.pi-np.log(np.tan(np.pi/4.+lat/2.)))
  
  return [x,y]
 
def get_map_config(xref,yref,lat_min,lat_max,lon_min,lon_max):
  """
  Computes the configuration of the map for figure size xref and yref in pixels
  and the bounding box of the coordinates to be plotted.
  """
  
  x_min, y_min = _web_mercator(lat_max,lon_min) # upper left
  x_max, y_max = _web_mercator(lat_min,lon_max) # lower right
  
  x_w = x_max - x_min
  zoom_x = np.log2(xref/x_w)
  y_w = y_max - y_min
  zoom_y = np.log2(yref/y_w)
  max_zoom = min([zoom_x, zoom_y])
  zoom = max_zoom-.5
  
  c_lat = (lat_max + lat_min)/2.
  c_lon = (lon_max + lon_min)/2.
  
  # pixels over the cut-off
  c_x, c_y = _web_mercator(c_lat,c_lon)
  y_upper = c_y*2**zoom - yref/2.
  # adjust zoom if needed
  if y_upper < 0:
    zoom = np.log2(yref/2./c_y)
  
  return {'zoom':zoom, 
          'c_lat':c_lat, 'c_lon':c_lon,
          'xref':xref, 'yref':yref,
          'c_x':c_x, 'c_y':c_y}
  
def transform_data(lat,lon,map_config):
  """
  Converts geographic coordinates into map coordinates suitable for plotting.
  """
  
  zoom = map_config['zoom']
  
  x,y = _web_mercator(lat,lon)

  x = (x - map_config['c_x'])*2**zoom*2+map_config['xref']-1. # center, -1 is for zero based indexing
  y = (y - map_config['c_y'])*2**zoom*2+map_config['yref']-1. 
  
  return x,y
  
def get_figure(api_token,map_type,map_config,scale_config = False):
  """
  Creates a matplotlib figure object with map. Use the figure to plot geographic
  coordinates on top that have been transformed with transfrom_data.
  """
  
  url = 'api.mapbox.com'
  
  request = "/styles/v1/mapbox/{}/static/{:.6f},{:.6f},{:.2f},0,0/{}x{}@2x?access_token={}" \
  .format(map_type,map_config['c_lon'],map_config['c_lat'],map_config['zoom']-1,
          int(map_config['xref']),int(map_config['yref']),api_token)
  
  print(request)
  conn = httplib.HTTPSConnection(url)
  conn.request("GET", request)
  map_resp = conn.getresponse()
  map_data = map_resp.read()
  image = BytesIO(map_data)
  from PIL import Image
  image.seek(0)
  map_image = Image.open(image).convert('RGB')
  conn.close()
  map_im_np = np.array(map_image)
  
  fig = plt.figure()

  plt.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off')
  
  plt.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left='off',      # ticks along the bottom edge are off
      right='off',         # ticks along the top edge are off
      labelleft='off')
  
  plt.imshow(map_im_np)
  
  pixel_per_m = 512./(2*np.pi*6371000.*np.cos(map_config['c_lat']*np.pi/180.))*2**map_config['zoom']
  # at zoom level zero the whole world has 512 pixels (retina tiles) and then the mapbox api has one zoom-level offset
  
  if scale_config != False:
    # scale
    if 'offset_x' in scale_config:
      offset_x = scale_config['offset_x']
    else:
      offset_x = 20
    if 'offset_y' in scale_config:
      offset_y = scale_config['offset_y']
    else:
      offset_y = 20
    if 'text_offset_x' in scale_config:
      text_offset_x = scale_config['text_offset_x']
    else:
      text_offset_x = 20
    if 'text_offset_y' in scale_config:
      text_offset_y = scale_config['text_offset_y']
    else:
      text_offset_y = 25
    if 'real_length' in scale_config:
      real_length = scale_config['real_length']
    else:
      real_length = np.round(map_config['xref']*0.3/pixel_per_m/1e4)*1e4
    
    pixel_len = real_length*pixel_per_m
    
    
    plt.plot([offset_x, offset_x+pixel_len],2*[offset_y],'-|k',markersize=10.)
    plt.text(offset_x+text_offset_x, offset_y+text_offset_y,'{} km'.format(int(real_length/1e3)))
  
  plt.xlim((0,2*map_config['xref']))
  plt.ylim((0,2*map_config['yref']))
  ax = plt.gca()
  ax.invert_yaxis()
  
  return fig  