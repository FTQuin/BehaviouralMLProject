LIBRARIES
---

- tensorflow 2.8
- cv2
- numpy
- pandas

# How to Use
## Quickstart
1. Open run_experiments.py
2. Change Hyperparameters to liking
- EXPERIMENT_NAME: name of experiment
- EXPERIMENT_PARAMS:
  - **batch_size**
  - **epochs**
- DATASETS_PARAMS:
  - dataset from datasets_config.py, **ex**: datasets.UCF
    - Dataset hyperparameters, **ex**: train\_test\_split
- EXTRACTOR_PARAMS:
  - feature extractor from feature\_extractors\_config.py
    - feature extractor params specific to that extractor, **ex**: movenet needs a threshold
- MODEL_PARAMS:
  - model from models\_config.py
    - model params specific to model, **ex**: activation_function 
3. Run File
- The models will train based on the parameters you set and will be saved in the saved_experiments directory under the name  experiment\_name parameter 
4. Results
- Results can be viewed in tensorboard by launching it with log directory set to the current experiments log directory. **ex.** tensorboard --logdir saved_experiments/test_experiment/logs/
5. Live Demo
- Open live_demo.ipynb
- Set directory in cell 2 to the directory of the model you would like to demo
- Ensure other hyperparameters such as SEQ_LEN are the same as they were when the model was trained
- Run the rest of the cells

## Adding new Model Architecture
1. open models_config.py
2. Create a new function and have it return a keras model
   - set model hyperparameters as inputs to the function
## Adding new Dataset
1. Add the raw data to the datasets directory
2. Create class for dataset in datasets\_config.py
   - create custom loading and saving methods
3. run preprocess_data.py with the extractor you would like to use
4. 
## Adding new Feature Extractor
1. Create a new class in feature\_extractors\_config.py which implements the **ExtractorAbstract** class
   - pre\_process\_features
     - any processing done to the raw data before extraction + extraction
   - post\_process\_features
     - any processing done to the features after extraction
