from tqdm import tqdm

from mt.dataset.load_dataset import load_dataset
from mt.dataset.dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size=1):
        self.X = dataset.X
        self.y = dataset.y
        self.batch_size = batch_size
        self.index = 0
        self.length = len(dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.index >= self.length:
            self.index = 0
            raise StopIteration

        batch = self.X[self.index:self.index + self.batch_size], self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch


if __name__ == '__main__':
    train_set, test_set = load_dataset('minst')
    train_loader = DataLoader(train_set, batch_size=64)
    for batch, label in tqdm(train_loader):
        print(batch, label)
        pass
