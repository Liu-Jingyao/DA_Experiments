import tokenizers
import torch
import transformers

if __name__ == '__main__':
    a = torch.Tensor([1,2,3,4])
    b = [3,4]
    a = a.tolist()
    a[2:3] = b
    print(a)
    tokenizers.Tokenizer.encode()