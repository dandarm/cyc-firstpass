import numpy as np
from mpl_toolkits.basemap import Basemap

from scipy.spatial import cKDTree

latcorners = [30, 48]
loncorners = [-7, 46]
lat_min, lat_max = latcorners
lon_min, lon_max = loncorners

def create_basemap_obj(ax=None, **kwargs):
  if ax is None:
    basemap_obj = Basemap(
        projection='geos',
        rsphere=(6378137.0, 6356752.3142),
        resolution='i',
        area_thresh=10000.,
        lon_0=9.5,
        satellite_height=3.5785831E7,
        llcrnrlon=lon_min,
        llcrnrlat=lat_min,
        urcrnrlon=lon_max,
        urcrnrlat=lat_max)
  else:
    basemap_obj = Basemap(
        projection='geos',
        ax=ax,
        rsphere=(6378137.0, 6356752.3142),
        resolution='i',
        area_thresh=10000.,
        lon_0=9.5,
        satellite_height=3.5785831E7,
        llcrnrlon=lon_min,
        llcrnrlat=lat_min,
        urcrnrlon=lon_max,
        urcrnrlat=lat_max)
  
  return basemap_obj

default_basem_obj = create_basemap_obj()
x_center, y_center = default_basem_obj(9.5, 0)

# griglia di corrispondenze da lon. lat a pixel_x, pixel_y

def get_lon_lat_grid_2_pixel(image_w, image_h):
    lon_grid, lat_grid, x, y = default_basem_obj.makegrid(image_w, image_h, returnxy=True)
    return lon_grid, lat_grid, x, y


# Funzione vettoriale
def get_cyclone_center_pixel_vector(lat_array, lon_array, image_w=1290, image_h=420):

    lon_grid, lat_grid, x, y = get_lon_lat_grid_2_pixel(image_w, image_h)

    # Appiattisci la griglia in array 1D di punti (lon, lat)
    grid_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Costruisci un KDTree per ricerca veloce
    tree = cKDTree(grid_points)

    # Prepara array dei punti da cercare
    query_points = np.column_stack((lon_array, lat_array))
    
    # Trova indice dei punti più vicini nella griglia
    dist, ind = tree.query(query_points)

    # Converti indici lineari in indici 2D
    i, j = np.unravel_index(ind, lon_grid.shape)

    # Ricorda: j → x, i → y (inverto y per altezza immagine)
    x_pix = j
    y_pix = image_h - i

    return x_pix, y_pix


def trova_indici_vicini(lon, lat, lon1, lat1):
    """
    Trova gli indici (riga, colonna) degli elementi nelle matrici 'lon' e 'lat'
    che sono più vicini al punto target (lon1, lat1).

    Parametri:
      - lon: np.ndarray di forma (height, width) che contiene le longitudini.
      - lat: np.ndarray di forma (height, width) che contiene le latitudini.
      - lon1: float, longitudine target.
      - lat1: float, latitudine target.

    Restituisce:
      Una tupla (i, j) dove:
         - i: indice della riga (asse verticale)
         - j: indice della colonna (asse orizzontale)
      tali che lon[i, j] e lat[i, j] sono le coordinate nella griglia più vicine a (lon1, lat1).

    Nota:
      Questa funzione calcola la distanza in termini di (lon-lon1)² + (lat-lat1)²;
      per trovare il pixel corrispondente, è sufficiente individuare il minimo di questa quantità.
    """
    # Calcola il quadrato della distanza per ogni elemento della griglia
    dist2 = (lon - lon1)**2 + (lat - lat1)**2

    # Trova l'indice (lineare) del minimo e convertilo in indici (riga, colonna)
    ind_min = np.argmin(dist2)
    i, j = np.unravel_index(ind_min, lon.shape)

    return j, i  # devono essere invertiti