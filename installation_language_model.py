from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer

model = AutoModel.from_pretrained('microsoft/deberta-v2-xlarge')
model.save_pretrained('deberta-v2-xlarge')

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
tokenizer.save_pretrained('deberta-v2-xlarge')