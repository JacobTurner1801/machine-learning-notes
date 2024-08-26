# Computer Vision

This course will teach you about Convolutional Neural Networks and Computer vision tasks.

## The convolutional classifier

A convet used for image classification has two parts:

* Convolutional base
* Dense Head

![Convet](../machine-learning-notes/images/convet.png)

The base is used to extract features. It is primarily focused on the layers performing the convolutional operation, but often includes other types of layers as well.

The head is used to determine the class of the image. It is formed primarily of dense layers, but might include dropout layers.

* A feature could be a line, colour, texture, shape, pattern, or some complicated combination *

![features](../machine-learning-notes/images/cnnFeatures.png)

### Training the classifier

The goal of a CNN during training is the following:

* Which features to extract from an image (base)
* Which class goes with those features (head)

More often than not, CNNs are not trained from scratch, instead you take a pretrained base and apply an untrained head to it.

![train](../machine-learning-notes/images/cnntrain.png)

The base usually consists of only a few dense layers, and has shown to be rather accurate from relatively little data.

Note: Reusing a pretrained model is a technique known as transfer learning. Almost every image classifier uses it today due to its effectiveness.

### Code examples

Step 1: Load dataset

<details>
    <summary>Code example</summary>
    ```python
        # Imports
        import os, warnings
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image_dataset_from_directory

        # Reproducability
        def set_seed(seed=31415):
            np.random.seed(seed)
            tf.random.set_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        set_seed(31415)

        # Set Matplotlib defaults
        plt.rc('figure', autolayout=True)
        plt.rc('axes', labelweight='bold', labelsize='large',
               titleweight='bold', titlesize=18, titlepad=10)
        plt.rc('image', cmap='magma')
        warnings.filterwarnings("ignore") # to clean up output cells


        # Load training and validation sets
        ds_train_ = image_dataset_from_directory(
            '../input/car-or-truck/train',
            labels='inferred',
            label_mode='binary',
            image_size=[128, 128],
            interpolation='nearest',
            batch_size=64,
            shuffle=True,
        )
        ds_valid_ = image_dataset_from_directory(
            '../input/car-or-truck/valid',
            labels='inferred',
            label_mode='binary',
            image_size=[128, 128],
            interpolation='nearest',
            batch_size=64,
            shuffle=False,
        )

        # Data Pipeline
        def convert_to_float(image, label):
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image, label

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds_train = (
            ds_train_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        ds_valid = (
            ds_valid_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
    ```
</details>

Step 2: Define pretrained base

<details>
    <summary>code example</summary>
    ```python
        pretrained_base = tf.keras.models.load_model(
            '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
        )
        pretrained_base.trainable = False
    ```
</details>

Step 3: Attach head

<details>
    <summary>code example</summary>
    ```python
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            pretrained_base,
            layers.Flatten(),
            layers.Dense(6, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
    ```
</details>

Step 4: Train

<details>
    <summary>Code example</summary>
    ```python
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        )

        history = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=30,
            verbose=0,
        )
    ```
</details>

Now we can plot the training graphs
```python
    import pandas as pd

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

CNNs show one advantage of deep learning over traditional machine learning models: given the right network structure, the NN can learn how to engineer the features it needs to solve its problem.

## Convolution and ReLU

### Feature Extraction

There are 3 basic operations to perform feature extraction.

1. Filter an image for a particular feature (convolution)
2. Detect a feature within the filtered image (RelU)
3. Condense the image to enhance the features (maximum pooling)

In modern convets, it is not uncommon to be filtering for over 1000 features in parallel.

#### Filtering

```python
# filtering
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])
```

### Weights and Kernels

The weights a convet learns during training and primarily contained in the convolutional layers. These weights are called kernels, these are represented as small 2D arrays.

A kernel operates by scanning over an image and producing a weighted sum of pixel values. This way, a kernel acts sort of like a polarised lens, emphasizing or deemphasizing certain patterns of information.

kernels define how a convolutional layer is connected to the layer that follows, typically the kernel size is given as two odd numbers (width, height) so that there is a centre pixel, but it's not a requirement.

The kernels in a convolutional layer dictate what kind of features it creates. During training, a convet tries to learn what features it needs to solve the classification problem. This means finding the best values for the kernels.

![kernel](../machine-learning-notes/images/cnnkernel.png)

### Activations

The activations in the network we call feature maps. They are what results when we apply a filter to an image; they contain the visual features the kernel extracts.

## Detect with ReLU

After filtering, the feature maps pass through the activation function. Here we will use ReLU (see [Intro to deep learning](../machine-learning-notes/intro%20to%20deep%20learning.md))

```python
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])
```

You could think about the activation function as scoring pixel values according to some measure of importance. The ReLU activation says that negative values are not important and so sets them to 0. ("Everything unimportant is equally unimportant.")

Below is the ReLU applied to feature maps (kernels).

![cnnrelu](../machine-learning-notes/images/cnnreluimage.png)

## Maximum Pooling

### Condense with maximum pooling

Adding a condensing step to the code we had before, we get this:

```python


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    layers.MaxPool2D(pool_size=2),
    # More layers follow
])
```

A MaxPool2D layer is much like a Conv2D layer, except that it uses a simple max function instead of a kernel with the pool_size parameter analogous to kernel_size. The MaxPool2D layer doesn't have trainable weights like a convolutional layer does however.

After applying the activation function (detect), the feature map may end up with some dead space, which is simply large areas containing only 0's (black areas in an image) that don't contain relevant information. So we would like to condense the feature map to only retain the useful part of the image -- the feature itself.

This is what maximum pooling does, it takes a patch of activations in the original feature map and replaces them with the maximum activation in that patch.

![maxPool](../machine-learning-notes/images/cnnMaxPooling.png)

To use maximum pooling, you can use the tf.nn.pool function. It does the same thing as the MaxPool2D layer but it's easier to use.

```python
image_condense = tf.nn.pool(
    input=image_detect, # image in the Detect step above
    window_shape=(2, 2),
    pooling_type='MAX',
    # we'll see what these do in the next lesson!
    strides=(2, 2),
    padding='SAME',
)
```

### Translation Invariance

Previously, I said that the zero-pixels don't contain relevant information. That might be true however they do contain one piece of information, positional information (where the pixel is on the image).

When MaxPool2D removes some of these pixels, it removes some information in the feature map. This gives the convet the property of translation invariance. This means that a convet with maximum pooling tends not to distinguish features by their location in the image.

When two features are really close together in a feature map, it is possible, after repeated pooling, for the features to become indistinguishable from each other (they merge into one), but over long distances, the features tend to stay apart. 

The main advantage of this is that image classifiers don't learn a given features location in an image, and may be able to recognise the same feature in two different locations of an image. This means that you don't need to train the CNN to not care about the location and hence can get away with less training data.