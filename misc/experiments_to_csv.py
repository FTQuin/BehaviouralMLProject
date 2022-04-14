import os
import pandas as pd

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
s = pd.Series(index=indexes, dtype='float64')


os.chdir(os.path.abspath(f'./MobilenetGridSearchEpoch25_NTU-6'))

os.chdir(os.path.abspath(f'./TxtFiles'))
accuracy_list = set()
models_list = set()

for file in os.listdir():
    for opt, param, model in zip(activation_function, optimizers, models):
        with open(file, 'r') as f:
            exp_name = f.readline().split(':')[1].strip().split('_')
            text = f.read()
            extractor = exp_name[1]
            act_func = exp_name[4]
            optimizer = exp_name[6]
            window = exp_name[9]
            mdl = exp_name[-1]
            acc = float(text.split('=')[1].split('\n')[0].replace('[', '').replace(']', '').split(',')[1])
            s[extractor, act_func, optimizer, window, mdl] = acc

print(s['MobileNetV2Extractor', 'relu', 'sgd', :, 'gru1'].mean())
print(s['MobileNetV2Extractor', 'relu', 'sgd', :, 'gru2'].mean())
print(s['MobileNetV2Extractor', 'relu', 'sgd', :, 'lstm1'].mean())
print(s['MobileNetV2Extractor', 'relu', 'sgd', :, 'lstm2'].mean())

print()
print(s['MobileNetV2Extractor', 'tanh', 'sgd', :, 'gru1'].mean())
print(s['MobileNetV2Extractor', 'tanh', 'sgd', :, 'gru2'].mean())
print(s['MobileNetV2Extractor', 'tanh', 'sgd', :, 'lstm1'].mean())
print(s['MobileNetV2Extractor', 'tanh', 'sgd', :, 'lstm2'].mean())

print()
print(s['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'gru1'].mean())
print(s['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'gru2'].mean())
print(s['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'lstm1'].mean())
print(s['MobileNetV2Extractor', 'sigmoid', 'sgd', :, 'lstm2'].mean())
s.to_csv(f'../../mobilenet_output_ntu-6.csv', index=True)