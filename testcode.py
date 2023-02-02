import torch
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["happy"]

# tokens = DistilBertTokenizerFast.convert_tokens_to_ids(tokenizer, sequences)
t = torch.Tensor([1,1,1])[None, :]
print(t.squeeze())