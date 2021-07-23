# molmapnets
> Neural networks for regression and classification for molecular data, using MolMap generated features.


```python
#all_slow
```

This package implements the neural network architects originally presented in the [MolMap](https://github.com/shenwanxiang/bidd-molmap) package, with two important differences:

- The package is written using literate programming so all functionalities are written and tested in Jupyter notebooks, and the implementation, testing, and documentation are done together at the same time. You can find the documentation on the [package website](https://riversdark.github.io/molmapnets/).
- The models are implemented in PyTorch.

## Install

First you need to install MolMap and ChemBench (you can find the detailed installation guide [here](https://github.com/shenwanxiang/bidd-molmap#installation)), then simply

`pip install molmapnets`

## How to use the package

We need `ChemBench` for the datasets, `MolMap` for feature extraction, and finally `molmapnets` for the neural networks.

```python
from chembench import dataset
from molmap import MolMap
```

    RDKit WARNING: [13:50:43] Enabling RDKit 2019.09.3 jupyter extensions


```python
from molmapnets.data import SingleFeatureData, DoubleFeatureData
from molmapnets.models import MolMapRegression
```

And for model training we also need `torch`

```python
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
torch.set_default_dtype(torch.float64)
```

Load and process data, using the `eSOL` dataset here for regression

```python
data = dataset.load_ESOL()
```

    total samples: 1128


```python
descriptor = MolMap(ftype='descriptor', metric='cosine',)
```

```python
descriptor.fit(verbose=0, method='umap', min_dist=0.1, n_neighbors=15,)
```

    2021-07-23 13:50:53,798 - INFO - [bidd-molmap] - Applying grid feature map(assignment), this may take several minutes(1~30 min)
    2021-07-23 13:50:56,904 - INFO - [bidd-molmap] - Finished


feature extraction

```python
X = descriptor.batch_transform(data.x)
```

    100%|##########| 1128/1128 [06:08<00:00,  2.78it/s]


Prepare data for model training

```python
esol = SingleFeatureData(data.y, X)
```

Train, validation, and test split

```python
train, val, test = random_split(esol, [904,112,112], generator=torch.Generator().manual_seed(7))
```

Batch data loader

```python
train_loader = DataLoader(train, batch_size=8, shuffle=True)
val_loader = DataLoader(val, batch_size=8, shuffle=True)
test_loader = DataLoader(test, batch_size=8, shuffle=True)
```

Initialise model

```python
model = MolMapRegression()

epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

Train model. The users are encouraged to tweak the training loop to achieve better performance

```python
for epoch in range(epochs):

    running_loss = 0.0
    for i, (xb, yb) in enumerate(train_loader):

        xb, yb = xb.to(device), yb.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward propagation
        pred = model(xb)

        # loss calculation
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('[Epoch: %2d] Training loss: %.3f' %
          (epoch + 1, running_loss / (i+1)))

print('Training finished')
```

    /Users/olivier/opt/anaconda3/envs/molmap/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  ../c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


    [Epoch:  1] Training loss: 4.530
    [Epoch:  2] Training loss: 1.803
    [Epoch:  3] Training loss: 1.541
    [Epoch:  4] Training loss: 1.209
    [Epoch:  5] Training loss: 1.092
    Training finished


Please refer to the package documentation for more detailed usage.
