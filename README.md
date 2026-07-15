# Wind Turbine Generator Failure Prediction
Predicting wind turbine generator failures using machine learning to enable predictive maintenance and reduce unplanned downtime.

## Overview
Wind turbine generators are prone to unexpected failures that lead to costly downtime and maintenance. This project builds a binary classification model to predict generator failure using sensor and operational data, helping identify at-risk turbines before failure occurs.

## Dataset
- **Size:** ~40,000 records
- **Type:** Binary classification (Healthy vs. Failure)
- **Class Imbalance:** ~17:1 (majority class: healthy, minority class: failure)
- **Preprocessing:** Cleaned and prepared for modeling; oversampling applied to address class imbalance

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib/Seaborn
- **Environment:** Google Colab / Spyder

## ML Models Used
| Model | Accuracy |

| **SVM (LinearSVC)** | **0.9909** |

| Logistic Regression | 0.9753 |

| Random Forest | 0.9651 |

| Naive Bayes | 0.8761 |

> Random Forest was tuned using `RandomizedSearchCV` for faster hyperparameter optimization on the large dataset. SVM (LinearSVC) delivered the best overall accuracy.

## Project Workflow
1. **Data Preprocessing** – cleaning, handling missing values, feature preparation
2. **Handling Class Imbalance** – oversampling the minority (failure) class
3. **Model Training** – trained and compared 4 algorithms (SVM, Logistic Regression, Random Forest, Naive Bayes)
4. **Hyperparameter Tuning** – RandomizedSearchCV applied to Random Forest
5. **Model Evaluation** – accuracy compared across all models; SVM selected as best performer

## Results
Given the class imbalance, accuracy alone isn't the full picture — SVM (LinearSVC) achieved the highest accuracy at **99.09%**, closely followed by Logistic Regression at 97.53%. Random Forest and Naive Bayes trailed behind, with Naive Bayes performing weakest at 87.61%.

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Open and run the notebook
jupyter notebook "The Wind Energy.ipynb"
```

## Project Structure
```
├── The Wind Energy.ipynb   # Main notebook: preprocessing, modeling, evaluation
├── README.md                # Project documentation
```

## Future Improvements
- Add precision, recall, and F1-score for deeper evaluation (especially on the minority/failure class)
- Deploy the best model as a simple web app or API for real-time prediction
- Experiment with deep learning approaches (e.g., LSTM for time-series sensor data)

## Author
**Nikil**
Final Year B.E. CSE (Data Science/ML) | Chennai, India
[LinkedIn](#) · [GitHub](#)
