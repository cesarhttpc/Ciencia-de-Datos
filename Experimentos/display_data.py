#Importar
import matplotlib.pyplot as plt
import numpy as np

def display_data(x, tile_width=-1, padding=0, axes=None):
    """
    Display data in a nice grid

    Parameters
    ----------
    x : ndarray
        Raw data.
    tile_width : int
        Width of each image.
    padding : int
        Padding around the image.
    axes : matplotlib.axes.Axes
        The axes for the plot
    show : bool
        True to show the plot immediately.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    m, n = x.shape

    if tile_width < 0:
        tile_width = int(np.round(np.sqrt(n)))
    tile_height = n / tile_width

    display_rows = int(np.floor(np.sqrt(m)))
    display_columns = int(np.ceil(m / display_rows))

    tile_height_padded = tile_height + padding * 2
    tile_width_padded = tile_width + padding * 2
    data = np.zeros((int(display_rows * tile_height_padded), int(display_columns * tile_width_padded)))

    for i in range(display_rows):
        for j in range(display_columns):
            tile = format_tile(x[i * display_rows + j, ], tile_width, padding)
            tile = tile.T
            data[int(i * tile_height_padded):int((i + 1) * tile_height_padded),
                 int(j * tile_width_padded):int((j + 1) * tile_width_padded)] = tile

    if axes:
        axes.imshow(data, cmap='gray', extent=[0, 1, 0, 1])
    else:
        plt.imshow(data, cmap='gray', extent=[0, 1, 0, 1])


def format_tile(x, width=-1, padding=0):
    """
    Format raw data to a 2-d array for plot.

    Parameters
    ----------
    x : ndarray
        Raw data, 1-d array.
    width : int
        Width of the image.
    padding : int
        Padding around the image.

    Returns
    -------
    ndarray
        The formatted 2-d array data for plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if width < 0:
        width = int(np.round(np.sqrt(len(x))))
    height = len(x) / width

    tile = np.ones((int(height + padding * 2), int(width + padding * 2)))

    for i in range(int(padding), int(height + padding)):
        tile[i, padding:(padding + width)] = x[((i - padding) * width):((i - padding) * width + width)]

    return tile