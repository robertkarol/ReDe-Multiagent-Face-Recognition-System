from RecognitionModel import RecognitionModel

model = RecognitionModel(normalize=True, encode_labels=False)
model.load_data_from_compressed('5-celebrity-faces-embeddings2.npz', is_embeddings_file=True)
model.train()
print(model.test() * 100)