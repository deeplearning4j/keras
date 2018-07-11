# Keras: Deep Learning for humans

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)


## You have just found Keras - nd4j implementation (inference only)

### Installation

* Install [Jumpy](https://github.com/deeplearning4j/deeplearning4j/tree/master/jumpy)
* Install `inference_only` branch of deeplearning4j/keras:

```bash
git clone https://www.github.com/deeplearning4j/keras.git
git checkout inference_only
cd keras
python setup.py install
```

### Example


```python
from keras.layers import *
from keras.models import Model
import jumpy as jp

x = Input((3, 2))
y = Dense(10)(x)

print(y._keras_shape)  # >> (None, 3, 10)  Shape inference working, Yay!

model = Model(x, y)

input_arr = jp.zeros((100, 3, 2))

output_arr = model.predict(input_arr)

 # optional: bring back to numpy space, no copy!
output_arr = output_arr.numpy()

```
