from RecognitionModel import RecognitionModel
from DatasetHelpers import DatasetHelpers


for i in range(1,7):
    model = RecognitionModel(normalize=True, encode_labels=False)
    #model.load_data_from_compressed('locals/5-celebrity-faces-embeddings2.npz', is_embeddings_file=True)
    #model.load_data_from_directory('locals/' + str(i) + '-train-images-per-class')
    model.load_data_from_directory('locals/' + 'lfw-dataset' + str(i))
    model.train()
    print(model.test() * 100)
    #RecognitionModel.save_model_as_binary(model, "model-pickle-" + str(i))
'''model = RecognitionModel.load_model_from_binary("model-pickle")
print(model.test() * 100)
print(model.predict_from_faces_images(DatasetHelpers.load_images("locals/Me/val/robi")))'''