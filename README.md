# Hypothesis Prediction Using XLM-Roberta

## Overview
This project implements a multilingual Natural Language Inference (NLI) pipeline using the XLM-Roberta model. It predicts the relationship between a premise and a hypothesis (contradiction, neutral, entailment) across multiple languages. The workflow leverages PyTorch and HuggingFace Transformers for model training, evaluation, and prediction.

## Results & Performance
- Achieved **>91% accuracy** on the test submission data.
- Kaggle Data: [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data)
- My Code on Kaggle: [Hypothesis Prediction Using XLM-Roberta](https://www.kaggle.com/code/aunanya875/hypothesis-prediction-using-xlm-roberta)

## Features
- Multilingual NLI using XLM-Roberta (XNLI)
- Stratified train/eval split by language and label
- Weighted loss for handling class imbalance
- Custom PyTorch Dataset and Sampler for balanced multilingual batches
- Training loop with accuracy tracking and model checkpointing
- Evaluation with classification report and confusion matrix
- Test set prediction and CSV submission generation
- Data visualization for language distribution

## Project Structure
- `hypothesis-prediction-using-xlm-roberta.ipynb`: Main Jupyter notebook containing all code and workflow
- `LICENSE`: Project license
- `README.md`: Project documentation

## Requirements
- Python 3.7+
- PyTorch
- Transformers (HuggingFace)
- pandas, numpy, seaborn, matplotlib, scikit-learn

Install dependencies (in notebook or terminal):
```python
!pip install transformers torch pandas numpy seaborn matplotlib scikit-learn
```

## Data
- **Train/Eval Data**: `train.csv` from Kaggle's "Contradictory, My Dear Watson" competition
- **Test Data**: `test.csv` from the same source

## Usage
1. **Open the notebook**: `hypothesis-prediction-using-xlm-roberta.ipynb`
2. **Run all cells**: The notebook will install dependencies, load data, preprocess, train, evaluate, and generate predictions.
3. **Model Training**: The notebook trains XLM-Roberta on the NLI task, saving the best model based on evaluation accuracy.
4. **Evaluation**: Prints classification report and confusion matrix for validation set.
5. **Test Prediction**: Loads test data, makes predictions, and saves results to `submission.csv`.

## Key Code Sections
- **Imports & Setup**: Installs and imports required libraries
- **Data Loading & Visualization**: Loads CSVs, analyzes and visualizes language distribution
- **Model & Tokenizer**: Loads XLM-Roberta model and tokenizer
- **Custom Dataset & Sampler**: Defines classes for balanced multilingual batching
- **Training Loop**: Trains model, tracks metrics, saves best checkpoint
- **Evaluation & Reporting**: Evaluates model, prints metrics
- **Test Prediction**: Generates predictions for test set

## Customization
- Adjust `MODEL_NAME` to use different transformer models
- Change `max_length`, `batch_size`, or `num_epochs` for experimentation
- Modify data paths for local or cloud environments

## Results
- The notebook provides detailed metrics and visualizations for model performance across languages.
- Final predictions are saved in `submission.csv` for Kaggle submission.

## License
This project is licensed under the terms of the LICENSE file.

## References
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Kaggle Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson)
- [XLM-Roberta Model](https://huggingface.co/joeddav/xlm-roberta-large-xnli)