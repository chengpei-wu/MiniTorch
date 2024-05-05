# MiniTorch

MiniTorch is a mini deep-learning package.

It includes the most fundamental and essential features of a deep learning framework:

- tensor computation (based on numpy)
- autograd (forward & backward)
- dataset & dataloader
- neural network blocks (Module, ModuleList, Linear...)
- functions (activation functions, loss functions)
- optimizers (SGD, Adam...)

# Autograd

```python
import mt

# scaler example
x = mt.Tensor(2, requires_grad=True)
print(x)
# 2

y = x * x + 2 * x  # d_y/d_x = 2*x + 2 === 2*2+2 === 6.0
print(y)
# 8

y.backward()
print(x.grad)
# 6.0

# matrix example
x = mt.random(10, 5)

w = mt.random(5, 2, requires_grad=True)
b = mt.random(2, requires_grad=True)

y = (x @ w) + b

z = y.sum()

z.backward()

print(y.grad)
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]]

print(w.grad)
# [[5.3130031  5.3130031 ]
#  [5.01509856 5.01509856]
#  [5.85540498 5.85540498]
#  [6.3287238  6.3287238 ]
#  [3.96215012 3.96215012]]

print(b.grad)
# [10. 10.]
```

# Training example (minst dataset)

```python
import numpy as np
from sklearn.metrics import accuracy_score

from mt.dataset.data_loading import DataLoader
from mt.dataset.load_dataset import load_dataset
from mt.nn.loss import cross_entropy
from mt.nn.model import MLP
from mt.optim.optimizers import SGD


def train(model, train_set, test_set, epoch, batch_size, learning_rate):
    # prepare
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=256)
    optimizer = SGD(model.parameters(), learning_rate)

    # train
    for epoch in range(epoch):
        total_loss = []
        train_acc = []
        for batch, labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = cross_entropy(logits, labels)
            total_loss.append(loss.values)
            loss.backward()
            optimizer.step()

            output = np.exp(logits.values - logits.values.max(axis=-1, keepdims=True))
            pred = output / output.sum(axis=-1, keepdims=True)
            acc = accuracy_score(np.argmax(pred, 1), np.argmax(labels.values, 1))
            train_acc.append(acc)

        test_acc = []
        for test_batch, test_labels in test_loader:
            logits = model(test_batch)
            output = np.exp(logits.values - logits.values.max(axis=-1, keepdims=True))
            pred = output / output.sum(axis=-1, keepdims=True)

            acc = accuracy_score(np.argmax(pred, 1), np.argmax(test_labels.values, 1))
            test_acc.append(acc)

        print(f'epoch: {epoch}, train_loss: {np.mean(total_loss):.3f} | train_accuracy: {np.mean(train_acc):.3f}| '
              f'test_accuracy: {np.mean(test_acc):.3f}')


if __name__ == '__main__':
    # load minst dataset
    train_set, test_set = load_dataset('minst')

    # construct MLP model
    model = MLP(in_feats=784, hid_feats=64, out_feats=10, num_layer=3)

    batch_size = 16
    learning_rate = 0.01

    train(
        model=model,
        train_set=train_set,
        test_set=test_set,
        epoch=100,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

# training process:
'''
epoch: 0, train_loss: 0.335 | train_accuracy: 0.896| test_accuracy: 0.939
epoch: 1, train_loss: 0.144 | train_accuracy: 0.956| test_accuracy: 0.957
epoch: 2, train_loss: 0.104 | train_accuracy: 0.969| test_accuracy: 0.958
epoch: 3, train_loss: 0.081 | train_accuracy: 0.976| test_accuracy: 0.963
epoch: 4, train_loss: 0.066 | train_accuracy: 0.979| test_accuracy: 0.961
epoch: 5, train_loss: 0.064 | train_accuracy: 0.979| test_accuracy: 0.962
epoch: 6, train_loss: 0.052 | train_accuracy: 0.984| test_accuracy: 0.962
epoch: 7, train_loss: 0.046 | train_accuracy: 0.985| test_accuracy: 0.965
epoch: 8, train_loss: 0.044 | train_accuracy: 0.985| test_accuracy: 0.965
epoch: 9, train_loss: 0.039 | train_accuracy: 0.988| test_accuracy: 0.958
epoch: 10, train_loss: 0.035 | train_accuracy: 0.988| test_accuracy: 0.963
epoch: 11, train_loss: 0.037 | train_accuracy: 0.989| test_accuracy: 0.964
......

'''
# 
```

# Todo

- normalization
- dropout
- conv_layers
- GPU acceleration
- distributed training
- ...
- 
