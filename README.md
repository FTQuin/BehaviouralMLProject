REQUIRED LIBRARIES
---

- tensorflow 2.8
- cv2
- numpy
- pandas

# How to Use
## Quickstart
** Preprocessed video features available in feature file, make usage of UCF-101 dataset and MovenetExtractor or 
MobilenetV2Extractor if you would like to try without processing your own data ** 
1. Open run_experiments.py
- **Modifiable Parameters**
  - EXPERIMENT_NAME: name of experiment
  - EXPERIMENT_PARAMS:
    - **name**
    - **batch_size**
    - **epochs**
  - DATASETS_PARAMS:
    - **dataset** from datasets_config.py, **ex**: datasets.UCF
    - Dataset hyperparameters
      - **train\_test\_split**
      - **seq_len**
  - EXTRACTOR_PARAMS:
    - **feature extractor** from feature\_extractors\_config.py
      - feature_extractors currently available from (feature_extractors_config.py) :
          - MobileNetV2Extractor
          - MovenetExtractor
          - InceptionV3Extractor
      - feature extractor params specific to that extractor, **ex**: movenet can take a **threshold value** {'threshold': 60}
  - MODEL_PARAMS:
    - **model** from models\_config.py
    - model params specific to model
      - **activation_function** 
      - **loss_function**
      - **optimizer**
3. Run File
- The models will train based on the parameters you set and will be saved in the saved_experiments directory under the name  experiment\_name parameter 
4. Results
- Results can be viewed in tensorboard by launching it with log directory set to the current experiments log directory. **ex.** tensorboard --logdir=saved_experiments/test_experiment/logs/
5. Live Demo
- Open live_demo.ipynb
- Choose experiment and model that you would like to demo
- Ensure other hyperparameters such as feature_extractor are the same as they were when the model was trained
- Run the rest of the cells

## How to Use
1. Preprocess raw videos into structured folder in the following format
- Dataset_name
    - action_label
        - video1
        - video2
        - ...
    - action_label2
        - video1
        - video2
        - ...

2. Once the dataset has been processed, add your folder to the **datasets** directory

![Dataset Image Example](./readme_images/dataset_example.png "Dataset Example" )

3. Go to preprocess_data.py
 - **Modifiable Parameters**
    - feature_extractors currently available from (feature_extractors_config.py) :
        - MobileNetV2Extractor
        - MovenetExtractor
        - InceptionV3Extractor
    - dataset_path 
        - UCF-101
        - NTU
        
 - Once you have chosen your parameters simply run the file and it will create a folder with the extracted features 
under the **features** directory

![Extracted Features Image Example](./readme_images/features_example.png "Dataset Example" )

 - Inside of your features folder you will have the folder name for the extractor that was used, the folders
for each action label and zip files containing the extracted features in a csv

4. Once you have your extracted features, head to run_experiment.py file 
and follow the steps in quickstart

## Adding new Model Architecture
1. open models_config.py
   - File contains 2 classes of models, GRU or LSTMS
   - In either class you can create a new model as a function or modify existing models to your liking
2. To create a new model create a new function and have it return a keras model
    - set model hyperparameters as inputs to the function

![Model Config Image Example](./readme_images/model_example.png "Model Example" ) 

## Adding new Dataset
1. Add the raw data to the datasets directory
2. Create class for dataset in datasets\_config.py
   - create custom loading and saving methods

3. run preprocess_data.py with the extractor you would like to use
## Adding new Feature Extractor
1. Create a new class in feature\_extractors\_config.py which implements the **ExtractorAbstract** class
   - pre\_process\_features
     - any processing done to the raw data before extraction + extraction
   - post\_process\_features
     - any processing done to the features after extraction
   - live\_extract
     - extraction method used for live inference
