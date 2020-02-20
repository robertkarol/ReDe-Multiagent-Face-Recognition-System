from RecognitionModel import RecognitionModel
from DatasetHelpers import DatasetHelpers

model = RecognitionModel(normalize=True, encode_labels=False)
model.load_data_from_compressed('locals/5-celebrity-faces-embeddings2.npz', is_embeddings_file=True)
model.train()
print(model.test() * 100)
RecognitionModel.save_model_as_binary(model, "model-pickle")
model = RecognitionModel.load_model_from_binary("model-pickle")
print(model.test() * 100)
print(model.predict_from_faces_images(DatasetHelpers.load_images("locals/Me/val/robi")))