# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf


## Model Details

- Person or organization developing model: Raghuveer Bhat Rajaprakash
- Model date: January 2026
- Model version: 1.0.0
- Algorithm: Random Forest Classifier with default hyperparameters
- Features: Categorical features (Workclass, Education, Marital-Status, Occupation, Relationship, Race, Sex, Native-Country) processed via One-Hot Encoding. Target label is binary (<=50K, >50K).
- Fairness: No specific fairness constraints or re-weighting were applied during training.
- Paper or other resource for more information: https://archive.ics.uci.edu/dataset/20/census+income


## Intended Use

The primary intended use of this model is for research and educational purposes to demonstrate machine learning pipelines on census data.


## Training Data

The model was trained on the Census Income Dataset (also known as the Adult Dataset), extracted from the 1994 Census database.
- Data: 80% of the Census Income Dataset.
- Source: https://archive.ics.uci.edu/dataset/20/census+income
- Features: The dataset includes attributes such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, and native country.
- Target Label: salary (<=50k, >50k)
- Preprocessing: Categorical features encoded via One-Hot Encoding. For target variable, we use Label Binarizer.


## Evaluation Data

- Data: 20% hold-out test set
- Preprocessing: The evaluation data was preprocessed using the exact same One-Hot Encoder and Label Binarizer fitted on the training data.

## Metrics

Metrics Used:
- Precision: Proportion of predicted high-earners that were correct
- Recall: Proportion of actual high-earners that were identified
- F1 Score: Harmonic mean of precision and recall

Model Performance:
- Precision: 0.7310
- Recall: 0.6295
- F1 Score: 0.6765


## Ethical Considerations

Significant performance disparities exist across racial groups and education levels, reflecting historical biases in the 1994 data.
The 1994 dataset does not reflect modern economic conditions.


## Caveats and Recommendations

- Caveats: The model was trained using the default parameters of the RandomForestClassifier in scikit-learn. No hyperparameter tuning (e.g., GridSearch) was performed, so the model is likely not fully optimized. Also there is a significant class imbalance in the dataset (far more <=50K examples than >50K). This biases the model towards the majority class and affects recall on the minority class.
- Recommendations: Future iterations should perform comprehensive hyperparameter tuning to maximize performance. Additionally, the data imbalance must be explicitly handled using techniques such as oversampling, or applying class weights during training.
