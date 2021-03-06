<div align="center">

![-](docs/readme_images/tru_img.png)
# Human Activity Recognition </h1>
### A Comparative Study between Different Pre-processing Approaches and Classifiers</h3>  
</div>

## Table of Contents
1. ➤ [Required Libraries](#required-libraries)
2. ➤ [Quickstart](#quickstart)
3. ➤ [Adding new Dataset](#adding-new-dataset)
   - [Processing Dataset](#processing-dataset)
4. ➤ [Adding new Model Architecture](#adding-new-model-architecture)
5. ➤ [Adding new Feature Extractor](#adding-new-feature-extractor)

![-](docs/readme_images/line-break.png)

## Required Libraries
- tensorflow 2.8
- cv2
- numpy
- pandas

requirements.txt provided:
```pip install -r requirements.txt```

![-](docs/readme_images/line-break.png)

## Quickstart
Example preprocessed video features are available in feature directory, make
usage of **UCF-101** or **NTU** dataset and **MovenetExtractor** or
**MobilenetV2Extractor** if you would like to try without processing your own data 

1. Open run_experiments.py or run_experiments.ipynb
   - **Modifiable Parameters**
     - _EXPERIMENT_NAME_: name of experiment
     - _EXPERIMENT_PARAMS_:
       - **name**
       - **batch_size**
       - **epochs**
     - _DATASETS_PARAMS_:
       - **dataset_path**: path of the original dataset
       - **train\_test\_split**
       - **seq_len**
     - _EXTRACTOR_PARAMS_:
       - **feature extractor** from feature\_extractors\_config.py
           - MobileNetV2Extractor
           - MovenetExtractor
           - InceptionV3Extractor
     - _MODEL_PARAMS_:
       - **model** from models\_config.py
       - model params specific to model
         - **activation_function** 
         - **loss_function**
         - **optimizer**
2. Run File
   - The models will train based on the parameters you set and will be saved in the saved_experiments directory under the name  experiment\_name parameter 
3. Results
   - Results can be viewed in tensorboard by launching it with log directory set to the current experiments log directory. **ex.** tensorboard --logdir=saved_experiments/test_experiment/logs/
4. Live Demo
   - Open live_demo.ipynb
   - Choose experiment and model that you would like to demo
   - Ensure other hyperparameters such as feature_extractor are the same as they were when the model was trained
   - Run the rest of the cells

![-](docs/readme_images/line-break.png)

## Adding new Dataset
1. Structure video datasets in the following format

    Dataset_name
   - action_label_0
     - video_name1
     - video_name2
     - ...
   - action_label_1
       - video_name1
       - video_name2
       - ...
        
![Dataset Image Example](docs/readme_images/dataset_example.png "Dataset Example" )

![-](docs/readme_images/line-break.png)

### Processing Dataset

1. Go to preprocess_data.py
2. Once you have chosen your parameters simply run the file and it will create a folder with the extracted features 
under the **features** directory

- **Modifiable Parameters**
   - feature_extractors currently available from (feature_extractors_config.py) :
       - MobileNetV2Extractor
       - MovenetExtractor
       - InceptionV3Extractor
   - dataset_path 
       - `../datasets/UCF-101`
       - `../datasets/NTU`
 
Inside of your features folder you will have the folder name for the extractor that was used, the folders
for each action label and zip files containing the extracted features in a csv

![Extracted Features Image Example](docs/readme_images/features_example.png "Dataset Example" )

![-](docs/readme_images/line-break.png)

## Adding new Model Architecture
1. open models_config.py
   - create a function that returns a model or modify existing models to your liking

```py
@staticmethod
def gru1(output_size, activation_function='relu',
         loss_function="sparse_categorical_crossentropy", optimizer="adam"):
    model = keras.Sequential([
        keras.layers.GRU(16, return_sequences=True),
        keras.layers.GRU(8),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(16, activation=activation_function),
        keras.layers.Dense(8, activation=activation_function),
        keras.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(loss=loss_function, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
    return model
```

![-](docs/readme_images/line-break.png)

## Adding new Feature Extractor
1. Create a new class in feature\_extractors\_config.py which implements the **ExtractorAbstract** class
   - `pre_process_extract_video(frames)`
     - any processing done to the raw data before extraction + extraction
   - `live_extract(frame)`
     - extraction method used for live inference

