# Finding-Donors-for-CharityML
##Project Overview

This project applies supervised learning techniques to data collected from the U.S. census to help CharityML, a fictional charity organization, identify individuals most likely to donate. CharityML aims to provide financial support for those eager to learn machine learning. After sending nearly 32,000 letters, the organization found that all donations came from individuals earning more than $50,000 annually. To expand its donor base while reducing mailing costs, CharityML plans to target residents of California who are most likely to donate.

Project Objectives

Explore factors affecting the likelihood of charity donations.

Develop a simple classifier as a baseline for comparison.

Train and test multiple supervised learning models on preprocessed census data.

Optimize the best-performing model based on accuracy, a modified F-score metric, and efficiency.

Reduce mailing costs while maximizing potential donations.

Evaluate the final model’s performance and interpret its predictions.

Data and Features

The dataset contains approximately 32,000 records, each with 13 features extracted from census data. These features include:

Demographics: Age, race, sex, marital status, and native country.

Education: Level of education and number of years completed.

Employment: Work class, occupation, and relationship status.

Financial Information: Capital gains, capital losses, and hours worked per week.

Target Variable: Income class (either <=50K or >50K).

The dataset is a modified version of the one published in Ron Kohavi’s research paper, Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid.

Methodology

Data Preprocessing:

Handle missing values.

Encode categorical variables.

Normalize numerical features.

Model Selection:

Train and evaluate different supervised learning models (e.g., Logistic Regression, Support Vector Machines, AdaBoost).

Compare models based on accuracy and F-score.

Hyperparameter Tuning:

Use GridSearchCV to optimize model parameters.

Choose the best model for final evaluation.

Performance Analysis:

Measure accuracy and F-score before and after optimization.

Analyze feature importance and model efficiency.

Implementation

The project is implemented using Python 3.x and requires the following libraries:

NumPy & Pandas: Data manipulation and analysis.

Matplotlib & Seaborn: Data visualization.

Scikit-Learn: Machine learning models and evaluation metrics.

The primary code is in the finding_donors.ipynb notebook, with supporting visualization scripts in visuals.py. An HTML export (report.html) contains the executed notebook with outputs.

Running the Project

To run the project, navigate to the main directory (finding_donors/) and execute:

jupyter notebook finding_donors.ipynb

This opens the Jupyter Notebook interface to interact with the code and results.
jupyter notebook finding_donors.ipynb

Conclusion

This project successfully demonstrates how supervised learning can optimize donor outreach for a charity. By identifying key features that indicate donation likelihood, the model reduces mailing costs while increasing potential contributions. Future work could explore deep learning techniques or additional demographic insights to further enhance predictions.

