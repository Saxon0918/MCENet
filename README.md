**Code Files**

This repository also includes several Python files used for training the model. Below is a description of each script:

- main.py: Runs the main function to train the model.

***modules***

- models.py: Define and implement the Multi-head Cross-modal Information Enhancement (MCIE) model, which integrates
  multiple modalities of data to improve the diagnostic accuracy for Alzheimer's disease by leveraging cross-modal
  information enhancement.
- multihead_attention.py: Define the structural components of model, specifically implementing the multi-head attention
  mechanism used for integrating cross-modal information.
- transformer.py: Define the structural components of model

***src***

- eval_metrics.py: Calculate evaluation metrics for model performance.
- preprocess_data.py: Preprocess the ADNI dataset by standardizing, registering, and segmenting the data for further
  analysis.
- train.py: Train the model using the preprocessed data and specified parameters.
- utils.py: Provide utility functions for model management.

***baselines***

- mlp.py: Implement the Multi-Layer Perceptron (MLP) model for Alzheimer's disease diagnosis.
- svm.py: Implement the Support Vector Machine (SVM) model for Alzheimer's disease diagnosis.


**Data Files**

The dataset used in this study is sourced from the Alzheimer's Disease Neuroimaging Initiative (ADNI), which includes
multimodal neuroimaging data (MRI and PET), clinical assessments, and genetic information. This dataset is widely used
to investigate the biomarkers and progression of Alzheimer's disease.


