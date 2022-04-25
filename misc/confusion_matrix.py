import sklearn.metrics
import experiment_runner.datasets_config as dc
import experiment_runner.feature_extractors_config as fe
import tensorflow as tf

# init model
model = tf.keras.models.load_model('../saved_experiments/example_NTU-3/gru')
# init dataset
dataset = dc.Dataset.Training('../datasets/UCF-3', fe.MobileNetV2Extractor(), 50)
# dataset to list
ds = list(dataset.test_dataset)
X = []
true_y = []
for i in ds:
    for j in zip(*i):
        X.append(j[0])
        true_y.append(j[1].numpy()[0])

# get predictions
res = model.predict_on_batch(tf.convert_to_tensor(X))
pred_y = []
for r in res:
    pred_y.append(r.argmax())

# print confusion matrix
print(sklearn.metrics.confusion_matrix(true_y, pred_y))
