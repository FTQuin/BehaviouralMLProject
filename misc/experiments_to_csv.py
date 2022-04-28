import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import tensorflow as tf
import experiment_runner.datasets_config as dc
import experiment_runner.feature_extractors_config as fe

write_to_path = os.path.abspath('../saved_experiments')
os.chdir(os.path.join('../saved_experiments'))

extractors = ["MobileNetV2Extractor"]
activation_function = ['relu', 'tanh', 'sigmoid']
optimizers = ['adam', 'sgd', 'adagrad']
seq_len = ['10', '20', '50']
models = ['gru1', 'gru2', 'lstm1', 'lstm2']

hyper_params = [extractors, activation_function, optimizers, seq_len, models]
indexes = pd.MultiIndex.from_product(hyper_params,
                                     names=['extractor', 'activation_function', 'optimizer', 'seq_len',
                                            'model'])
df = pd.DataFrame(index=indexes, columns=['accuracy', 'time', 'f1', 'recall', 'precision'], dtype='float64')

# init dataset
ex = fe.MobileNetV2Extractor()     ## HERE
datasets = {10: None, 40: None, 50: None}
labels = []
for seq in [10, 20, 50]:
    dataset = dc.Dataset.Training('../../../datasets/UCF-3', ex, seq)      ## HERE
    # dataset to list
    ds = list(dataset.test_dataset)
    X = []
    true_y = []
    for i in ds:
        for j in zip(*i):
            X.append(j[0])
            true_y.append(j[1].numpy()[0])
    datasets[seq] = [X, true_y]
    labels = dataset.labels
    del dataset
    del ds
del ex.feature_extractor
del ex

def get_preds(mod_name, seq_len):
    # init model
    model = tf.keras.models.load_model('../' + mod_name[6:-1])
    # get predictions
    res = model.predict_on_batch(tf.convert_to_tensor(datasets[seq_len][0]))
    pred_y = []
    for r in res:
        pred_y.append(r.argmax())
    return datasets[seq_len][1], pred_y


os.chdir(os.path.abspath(f'./MobilenetGridSearchEpoch25_NTU-6'))     ## HERE
os.chdir(os.path.abspath(f'./TxtFiles'))

for file in os.listdir():
    with open(file, 'r') as f:
        exp_name = f.readline()
        exp_name_split = exp_name.split(':')[1].strip().split('_')
        text = f.read()
        timeing = text.split('\n')[-5].split('=')[1]
        extractor = exp_name_split[1]
        act_func = exp_name_split[4]
        optimizer = exp_name_split[6]
        window = exp_name_split[9]
        mdl = exp_name_split[-1]
        acc = float(text.split('=')[1].split('\n')[0].replace('[', '').replace(']', '').split(',')[1])
        df.loc[(extractor, act_func, optimizer, window, mdl), 'accuracy'] = acc
        df.loc[(extractor, act_func, optimizer, window, mdl), 'time'] = timeing
        true, pred = get_preds(exp_name, int(window))
        df.loc[(extractor, act_func, optimizer, window, mdl), 'f1'] = f1_score(true, pred, average='weighted')
        df.loc[(extractor, act_func, optimizer, window, mdl), 'recall'] = recall_score(true, pred, average='weighted')
        df.loc[(extractor, act_func, optimizer, window, mdl), 'precision'] = precision_score(true, pred, average='weighted')
        print(exp_name)
        print(classification_report(true, pred, labels=labels))


# print(df['MobileNetV2Extractor', 'relu', 'sgd', :, 'gru1'].mean())
# print(df['MobileNetV2Extractor', 'relu', 'sgd', :, 'gru2'].mean())
# print(df['MobileNetV2Extractor', 'relu', 'sgd', :, 'lstm1'].mean())
# print(df['MobileNetV2Extractor', 'relu', 'sgd', :, 'lstm2'].mean())
#
# print()
# print(df['MobileNetV2Extractor', 'tanh', 'sgd', :, 'gru1'].mean())
# print(df['MobileNetV2Extractor', 'tanh', 'sgd', :, 'gru2'].mean())
# print(df['MobileNetV2Extractor', 'tanh', 'sgd', :, 'lstm1'].mean())
# print(df['MobileNetV2Extractor', 'tanh', 'sgd', :, 'lstm2'].mean())
#
# print()
# print(df['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'gru1'].mean())
# print(df['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'gru2'].mean())
# print(df['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'lstm1'].mean())
# print(df['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'lstm2'].mean())
df.to_csv(f'../../mobilenet_output_ntu-6.csv', index=True)

# datasets = ["NTU-6", "UCF-3"]
# extractors = ["MovenetExtractor", "InceptionExtractor", "MobileNetV2Extractor"]
# activation_function = ['relu', 'tanh', 'sigmoid']
# optimizers = ['adam', 'sgd', 'adagrad']
# seq_len = [10, 20, 50]
# models = ['gru1', 'gru2', 'lstm1', 'lstm2']
# hyper_params = [datasets, extractors, activation_function, optimizers, seq_len, models]
# indexes = pd.MultiIndex.from_product(hyper_params,
#                                      names=['dataset', 'extractor', 'activation_function', 'optimizer',
#                                             'seq_len', 'model'])
# s_main = pd.DataFrame(index=indexes, columns=['accuracy', 'time', 'f1', 'recall', 'precision'], dtype='float64')
# df = pd.read_csv('./saved_experiments/movenet_output_ntu-6.csv', index_col=list(range(5)))
# s_main.loc['NTU-6'].loc[['MovenetExtractor']] = df
# df = pd.read_csv('./saved_experiments/mobilenet_output_ntu-6.csv', index_col=list(range(5)))
# s_main.loc['NTU-6'].loc[['MobileNetV2Extractor']] = df
# df = pd.read_csv('./saved_experiments/inception_output_ntu-6.csv', index_col=list(range(5)))
# s_main.loc['NTU-6'].loc[['InceptionExtractor']] = df
# df = pd.read_csv('./saved_experiments/inception_output.csv', index_col=list(range(5)))
# s_main.loc['UCF-3'].loc[['InceptionExtractor']] = df
# df = pd.read_csv('./saved_experiments/movenet_output.csv', index_col=list(range(5)))
# s_main.loc['UCF-3'].loc[['MovenetExtractor']] = df
# df = pd.read_csv('./saved_experiments/mobilenet_output.csv', index_col=list(range(5)))
# s_main.loc['UCF-3'].loc[['MobileNetV2Extractor']] = df
#
# for n in s_main.index.names:
#     print(f'\n===={n}====')
#     s_main.index.get_level_values(n).unique()
#     for i in s_main.index.get_level_values(n).unique():
#         print(f'==={i}===')
#         for m in s_main.droplevel(n).index.names:
#             # print(f'\n=={m}==')
#             for j in s_main.droplevel(n).index.get_level_values(m).unique():
#                 # print(j)
#                 print(s_main.xs(i, level=n).xs(j, level=m).mean())
#         # print('average')
#         print(s_main.xs(i, level=n).mean())
