from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format

_BACKEND = 'deeplearning4j'


sys.stderr.write('Using deeplearning4j backend\n')

from jumpy.keras.dl4j_backend import *

def backend():
    """Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
