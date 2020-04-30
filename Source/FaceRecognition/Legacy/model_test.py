import time

from RecognitionModel import RecognitionModel
from DatasetHelpers import DatasetHelpers

'''for i in range(7, 8):
    for j in range(1, 8, 2):
        model = RecognitionModel(normalize=False, encode_labels=False)
        #model.load_data_from_compressed('locals/5-celebrity-faces-embeddings2.npz', is_embeddings_file=True)
        #model.load_data_from_directory('locals/' + str(i) + '-train-images-per-class')
        model.load_data_from_directory('../' + 'lfw-dataset' + str(i))
        model.train(neighbors=j)
        test = model.test()
        with open("res.txt", "a") as f:
            f.write("%d %d\n" % (i, j))
            f.write(str(test * 100) + '\n')
        print(test * 100)
        RecognitionModel.save_model_as_binary(model, "model-pickle-knn" + str(i) + str(j))
        del model'''
'''model = RecognitionModel.load_model_from_binary("model-pickle-knn" + str(i) + str(j))
        # print(model.test() * 100)
        threshold = 0.49999999
        while threshold <= 1:
            print(i, " ", j, " ", threshold)
            test = model.test_threshold(threshold)
            print(test)
            with open("results2.txt", "a") as f:
                f.write("%d %d %s\n" % (i, j, str(threshold)))
                f.write(str(test) + '\n')
            threshold += 0.05'''
'''for i in range(1, 7):

    for j in range(1, i + 1):
        model = RecognitionModel(normalize=False, encode_labels=False)
        # model.load_data_from_compressed('locals/5-celebrity-faces-embeddings2.npz', is_embeddings_file=True)
        # model.load_data_from_directory('locals/' + str(i) + '-train-images-per-class')
        model.load_data_from_directory('../' + 'lfw-datasetb' + str(i))
        model.train(neighbors=j)
        test = model.test()
        with open("results-knn-l2.txt", "a") as f:
            f.write("%d %d\n" % (i, j))
            f.write(str(test * 100))
        print(test * 100)
        del model
    # RecognitionModel.save_model_as_binary(model, "model-pickle-" + str(i))'''
'''model = RecognitionModel.load_model_from_binary("model-pickle")
print(model.test() * 100)
print(model.predict_from_faces_images(DatasetHelpers.load_images("locals/Me/val/robi")))'''

model = RecognitionModel.load_model_from_binary("model-pickle-knn75")
print(model.test())
model.retrain_from_dataset('../retrain')
print(model.test())
images_to_predict = []
images_to_predict.extend(DatasetHelpers.load_images('../retrain/val/mindy_kaling'))
images_to_predict.extend(DatasetHelpers.load_images('../retrain/val/robi'))
images_to_predict.extend(DatasetHelpers.load_images('../retrain/val/madonna'))
start_time = time.time()
print(model.predict_from_faces_images(images_to_predict))
elapsed_time = time.time() - start_time
print(elapsed_time)
elapsed_time = 0
for image in images_to_predict:
    start_time = time.time()
    model.predict_from_faces_images([image])
    elapsed_time += time.time() - start_time
print(elapsed_time)