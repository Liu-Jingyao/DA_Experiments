import datasets


class SizedIterableDataset(datasets.IterableDataset):
    @classmethod
    def convert_to_sized(cls, obj, rows_num):
        obj.len = rows_num
        obj.num_rows = rows_num
        obj.__class__ = SizedIterableDataset
        return obj

    def __len__(self):
        return self.len