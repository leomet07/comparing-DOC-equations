import numpy as np

# this is all zero indexes
equation_functions = [
    lambda bands: [(bands[2] / bands[4]), (bands[3])],
    lambda bands: [np.log(bands[1] / bands[4]), (bands[0])],  # zero-indexed
    lambda bands: ((bands[2] / bands[4]), (bands[2])),
    lambda bands: (np.log(bands[2] / bands[3]), np.log(bands[3] / bands[4])),
    #
    lambda bands: ((bands[2] / bands[3]), (bands[3] / bands[4])),
    lambda bands: (np.log(bands[1] / bands[4]), (bands[1])),
    lambda bands: ((bands[1] / bands[4]), (bands[1])),
    lambda bands: ((bands[2] / bands[3]), (bands[2])),
    #
    lambda bands: ((bands[0] / bands[3]), (bands[1] / bands[3])),
    lambda bands: ((bands[0] / bands[3]), (bands[2] / bands[3])),
    lambda bands: ((bands[2] / bands[3]), (bands[1])),
]
