from mt.nn.module import Module, ModuleList
from mt.nn.layer import Linear


class MLP(Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layer):
        super().__init__()
        self.layers = ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.layers.append(Linear(in_feats, hid_feats, activation="relu"))
            elif i == num_layer - 1:
                self.layers.append(Linear(hid_feats, out_feats))
            else:
                self.layers.append(Linear(hid_feats, hid_feats, activation="relu"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
