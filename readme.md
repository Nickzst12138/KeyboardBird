# Project Overview

It is divided into two parts, the pytorch part and the c++ part. The convolutional neural network (CNN) is implemented through the PyTorch framework and C++, and the image classification is performed on the CIFAR-10 dataset using the ResNet architecture. It includes data enhancement technology (Cutout) and the python implementation part in Jupyter Notebook.

The c++ part uses C++ to implement the core functions of the convolution operation and the fully connected layer.

## File structure

```
project-root/
│
├── checkpoint/ # Folder for saving model checkpoints
├── dataset/ # File containing the CIFAR-10 dataset
├── utils/ # Utility scripts, including data loading and augmentation
├── final.cpp # Convolution and fully connected layer operations implemented in C++
├── train.ipynb # Jupyter Notebook for training models
├── test (1)(1).ipynb # Jupyter Notebook for testing models
├── readme.md # Project description document
```

---

## Detailed code description

### 1. final.cpp

**Function**:
The core operations in the neural network are implemented in C++, including 2D convolution, ReLU activation, fully connected layers, and loading weights for inference. This file shows how to implement the basic functions of convolutional neural networks from scratch without relying on deep learning libraries.

**Key functions**:

1. `conv2d`: performs two-dimensional convolution operations, supporting multi-channel input and output.

2. `relu`: ReLU activation function, used to introduce nonlinear characteristics, setting all negative values ​​to 0.

3. `fully_connected`: implementation of the fully connected layer, flattens the feature map after convolution, and inputs it into the fully connected layer for linear transformation.

4. `load_weights`: loads pre-trained model weights from a file for subsequent inference.



### 2. ResNet.py

**Function**:
Implements the definition of ResNet architecture and supports ResNet with different layers. ResNet is a deep residual network that introduces "skip connections" to alleviate the gradient vanishing problem in deep networks.

**Key modules**:

1. `BasicBlock`: Basic residual module for ResNet18 and ResNet34.

2. `Bottleneck`: Deeper modules for ResNet50 and above.

3. `_make_layer`: Build multiple residual blocks and adjust the network depth.


### 3. cutout.py

**Function**:
Implements the Cutout data augmentation technique, which increases data diversity by randomly removing parts of the image area, thereby improving the generalization ability of the model.

**Key classes**:
1. `Cutout`: Responsible for generating a random square mask and applying it to the input image.

**Code snippet and explanation**:

class Cutout(object):
def __init__(self, n_holes, length):
// Initialize the number of patches to be removed and the size of the patches
self.n_holes = n_holes
self.length = length

def __call__(self, img):
// Generate a random mask on the image
h, w = img.size(1), img.size(2)
mask = np.ones((h, w), np.float32)
for n in range(self.n_holes):
y, x = np.random.randint(h), np.random.randint(w)
y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + the length // 2, 0, w)
mask[y1: y2, x1: x2] = 0.
mask = torch.from_numpy(mask).expand_as(img)
img = img * mask
return img
```
- This class increases the randomness of training data and reduces overfitting by randomly generating patches (removed areas) on each image.

### 4. readData.py

**Function**:
Responsible for loading the CIFAR-10 dataset, performing data augmentation, and dividing the training set and validation set. Supports custom batch size, validation set ratio and other parameters.

**Key functions**:
1. `read_dataset`: Responsible for loading the CIFAR-10 dataset and performing preprocessing (such as random cropping, horizontal flipping, Cutout).

def read_dataset(batch_size=16, valid_size=0.2, num_workers=0, pic_path='dataset'):
// CIFAR-10 data enhancement, including random cropping, horizontal flipping and normalization
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
Cutout(n_holes=1, length=16),
])
// Test set preprocessing
transform_test = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

# How to run

1. **Environment setup**:
- Download the CIFAR-10 dataset to the `dataset/` folder, install the pytorch environment and the ipykernel kernel of jupyter.

2. **Train the model**:
- Open and run `train.ipynb` to complete the data loading, model definition and training steps.

3. **Test the model**:
- Use the `test (1)(1).ipynb` notebook to load the trained model and evaluate its performance on the test set.
The C++ part compiles an exe executable file
---