class Dataset:
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
