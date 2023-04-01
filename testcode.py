import torch
import transformers

if __name__ == '__main__':
    import torch

    a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 2], [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 2]])
    keep = torch.Tensor([[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]])

    # create boolean mask
    keep[:,-1] = 0
    mask = (keep != 0)
    max_len = len(keep[0])

    # select non-zero elements from a
    a_list = []
    for i in range(a.size(0)):
        new_a_i = a[i][mask[i]]
        new_a_i = torch.nn.functional.pad(new_a_i, (0, max_len - len(new_a_i) - 1), mode='constant', value=0)
        new_a_i = torch.cat((new_a_i, a[i,-1].unsqueeze(0)))
        a_list.append(new_a_i)
    a = torch.stack(a_list)

    print(a)