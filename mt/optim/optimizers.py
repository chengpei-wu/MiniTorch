class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter['value'].values -= parameter['value'].grad * self.lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter['value'].zero_grad()
