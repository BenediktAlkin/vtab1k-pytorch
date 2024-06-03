from torch.utils.data.sampler import BatchSampler


class InfiniteBatchSampler(BatchSampler):
    def __init__(self, epochs, sampler, batch_size, drop_last):
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        assert isinstance(epochs, int) and 0 < epochs
        self.epochs = epochs

    def __iter__(self):
        epoch = 0
        while True:
            for batch in super().__iter__():
                yield batch
            epoch += 1
            if epoch == self.epochs:
                break

    def __len__(self):
        raise NotImplementedError
