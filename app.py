import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import graphviz

# Introduction
st.title("Predicting Survival of Heart Failure Patients")
st.write("""
This dataset focuses on cardiovascular diseases (CVDs), which are the leading cause of death worldwide, claiming approximately 17.9 million lives annually and accounting for about 31% of global deaths. Among CVDs, heart failure is a prevalent condition, and the dataset includes 12 specific features that can be used to forecast mortality due to heart failure.

Preventative measures against most cardiovascular diseases primarily involve addressing behavioral risk factors, including tobacco use, unhealthy eating habits, obesity, lack of physical activity, and excessive alcohol consumption, through strategies implemented at the population level.

Individuals afflicted with cardiovascular disease or those at a heightened risk of developing it, owing to the presence of risk factors like hypertension, diabetes, hyperlipidemia, or an existing illness, necessitate early detection and effective management. Here, the application of a machine learning model could prove highly beneficial.
""")

# Title
st.title("Heart Failure Clinical Records Dataset")

st.subheader("1. Import libraries and read the data")
st.text("To begin, we import the following libraries")
commands = """
%pip install scikit-learn
%pip install ucimlrepo
%pip install seaborn
%pip install --upgrade matplotlib
%pip install streamlit
%pip install graphviz
%pip install pydotplus
"""

st.code(commands, language="plaintext")

# Display import statements
st.subheader("Import Statements:")
imports = """
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
"""

st.code(imports, language="python")

st.subheader("Import the data using Pandas")
# load dataset
st.code("heart_failure_df = pd.read_csv('resources/heart_failure_clinical_records_dataset.csv')")

heart_failure_df = pd.read_csv('resources/heart_failure_clinical_records_dataset.csv')

# Display the dataset on Streamlit
st.subheader("Dataset Overview")
st.write(heart_failure_df)

# Display data exploration information
st.subheader("Data Exploration:")
st.markdown("## 2. Explore the data")
st.markdown("Print the feature names to make sure you have the right dataset.")

# Print the features and dataframe size
st.code("""
# print the features
print(heart_failure_df.columns)

# Check the Size of the dataframe
size = heart_failure_df.shape
print("The size of this dataframe is: ", size)
""", language="python")
st.subheader("Feature names")
heart_failure_df.columns

st.subheader("Data size")
size = heart_failure_df.shape
st.text("The size of this dataframe is: ")
size


# Split the data
st.header("3. Data Splitting")
st.code("""
# split the data
X = heart_failure_df.drop('DEATH_EVENT', axis=1)  # Independent variables (features)
y = heart_failure_df['DEATH_EVENT'] # Dependent variable (label)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
""")

# Split the data
X = heart_failure_df.drop('DEATH_EVENT', axis=1)  # Independent variables (features)
y = heart_failure_df['DEATH_EVENT']  # Dependent variable (label)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X

y
# Build and train the models
st.header("4. Model Training")
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, y_train)

# Create predictions
y_pred = clf.predict(X_test)

# Check the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
st.write(f"Decision Tree Model Accuracy: {accuracy}")

# Build and train the Random Forest model
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Calculate and display the accuracy for Random Forest
accuracy_rf = metrics.accuracy_score(y_test, y_pred_rf)
st.write(f"Random Forest Model Accuracy: {accuracy_rf}")

# Generate and display the confusion matrix for Decision Tree
st.header("Decision Tree Model")
cm_dt = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix for Decision Tree:")
st.write(cm_dt)

# Generate and display the confusion matrix for Random Forest
st.header("Random Forest Model")
cm_rf = confusion_matrix(y_test, y_pred_rf)
st.write("Confusion Matrix for Random Forest:")
st.write(cm_rf)

# Plot the confusion matrices for Decision Tree and Random Forest
st.header("Confusion Matrix Visualizations")

st.text("To have a better understanding I will plot the confusion matrix for both the Decision Tree and Random Forest")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix for Decision Tree Model
disp_dt = metrics.ConfusionMatrixDisplay(cm_dt, display_labels=['Survival', 'Death'])
disp_dt.plot(ax=axes[0])
axes[0].set_title('Confusion Matrix for Decision Tree Model')

# Random Forest Confusion Matrix
disp_rf = metrics.ConfusionMatrixDisplay(cm_rf, display_labels=['Survival', 'Death'])
disp_rf.plot(ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')

plt.tight_layout()  # Adjust layout to prevent overlapping

# Display the figure using Streamlit
st.subheader("Confusion Matrices:")
st.pyplot(fig)

# Decision Tree Confusion Matrix
dt_text = """
**Decision Tree Confusion Matrix:**
- Out of the instances predicted as "Survival," 45 were correctly identified, while 8 instances were incorrectly predicted as "Death."
- On the other hand, 20 instances predicted as "Death" were correct, but 17 instances were incorrectly predicted as "Survival."
"""

# Random Forest Confusion Matrix
rf_text = """
**Random Forest Confusion Matrix:**
- Among the instances predicted as "Survival," 48 were correct, but 5 instances were incorrectly predicted as "Death."
- Regarding instances predicted as "Death," 21 were correct, while 16 instances were incorrectly predicted as "Survival."
"""

# Conclusions
conclusions_text = """
**Conclusions:**
**Decision Tree Model:**
- Overall Accuracy: (TP + TN) / (TP + TN + FP + FN) = (20 + 45) / (20 + 45 + 8 + 17) ≈ 0.71 (71%)
- The model correctly predicted 71% of the instances.
- The model has more False Negatives (17) than False Positives (8).

**Random Forest Model:**
- Overall Accuracy: (TP + TN) / (TP + TN + FP + FN) = (21 + 48) / (21 + 48 + 5 + 16) ≈ 0.75 (75%)
- The model correctly predicted 75% of the instances.
- The model has more False Negatives (16) than False Positives (5).
"""

# Comparison
comparison_text = """
**Comparison:**
The Random Forest model has a slightly higher overall accuracy compared to the Decision Tree model.
Both models seem to have a higher number of False Negatives, indicating that they might be more conservative in predicting positive instances.
"""
# Display the text using Streamlit
st.subheader("Model Evaluation:")
st.markdown(dt_text, unsafe_allow_html=True)
st.markdown(rf_text, unsafe_allow_html=True)
st.markdown(conclusions_text, unsafe_allow_html=True)
st.markdown(comparison_text, unsafe_allow_html=True)



st.subheader("The Decision Tree")
# Plot the decision tree
def plot_decision_tree(clf, feature_names):
    # Generate the DOT data for the decision tree
    dot_data = export_graphviz(
        clf,
        precision=True,
        out_file=None,  # Set to None to return the result as a string
        feature_names=feature_names,
        class_names=['Survival', 'Death'],  # Use a list of class names
        filled=True,
        rounded=True
    )
    # Display the graph
    st.graphviz_chart(dot_data)

# Assuming clf is your decision tree classifier and X is your feature data
plot_decision_tree(clf, list(X.columns))
# Step 4: Choose 2 other ML techniques from the ScikitLearn library
st.header("Additional Models")

# Support Vector Machines (SVM)
st.header("Support Vector Machines (SVM)")
# Support Vector Machines (SVM) Explanation
st.subheader("Support Vector Machines (SVM)")
st.write("""
Support Vector Machines (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. The primary goal of SVM is to find the optimal hyperplane that separates data points of different classes while maximizing the margin between the classes. It works by mapping data points into a high-dimensional feature space and finding the optimal hyperplane that best segregates the data. This hyperplane is chosen so that it maximizes the distance between the nearest data points of all the classes, known as the margin. The data points that are closest to the hyperplane are called support vectors, and they are crucial in determining the hyperplane's position and orientation. SVM is effective in handling complex decision boundaries and is known for its ability to generalize well to new, unseen data.
""")

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(y_test, svm_y_pred)
st.write("SVM Accuracy:", svm_accuracy)

# ROC curve for SVM
fpr, tpr, thresholds = roc_curve(y_test, svm_y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
plt.legend(loc="lower right")
fig_svm = plt.gcf()

# Display the figure using st.pyplot()
st.pyplot(fig_svm)
# Gradient Boosting Machines (GBM)
st.header("Gradient Boosting Machines (GBM)")

# Gradient Boosting Machines (GBM) Explanation
st.subheader("Gradient Boosting Machines (GBM)")
st.write("""
Gradient Boosting Machines (GBM) is an ensemble learning technique that combines the predictions of several individual models to create a stronger predictive model. It operates by building a series of weak learners, typically decision trees, in a sequential manner. Each new tree is trained to correct the errors made by the previous ones. GBM works by optimizing a predefined loss function, aiming to minimize the errors between the actual and predicted values. It assigns higher weights to data points that were previously misclassified, enabling subsequent models to focus more on these difficult-to-predict instances. The final prediction is generated by summing the predictions from all the individual models. GBM is known for its high predictive power and flexibility, making it a popular choice for various machine learning tasks, including regression and classification problems.
""")

# Train the Gradient Boosting model
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gbm_model.fit(X_train, y_train)
gbm_y_pred = gbm_model.predict(X_test)
gbm_accuracy = accuracy_score(y_test, gbm_y_pred)
st.write("GBM Accuracy:", gbm_accuracy)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
st.write("SVM Accuracy:", svm_accuracy)

# Display confusion matrices side by side
st.subheader("Confusion matrices SVM and GBM")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SVM Confusion Matrix
disp_svm = ConfusionMatrixDisplay(confusion_matrix(y_test, svm_y_pred), display_labels=['Survival', 'Death'])
disp_svm.plot(ax=axes[0])
axes[0].set_title('SVM Confusion Matrix')

# Gradient Boosting Confusion Matrix
disp_gbm = ConfusionMatrixDisplay(confusion_matrix(y_test, gbm_y_pred), display_labels=['Survival', 'Death'])
disp_gbm.plot(ax=axes[1])
axes[1].set_title('Gradient Boosting Confusion Matrix')

st.pyplot(fig)

# Display conclusions
st.write("""
Support Vector Machines (SVM) Confusion Matrix:
- Instances predicted as "Survival" had 49 correct predictions, but 4 were incorrectly predicted as "Death."
- For instances predicted as "Death," 21 were correct, and 16 were incorrectly predicted as "Survival."

Gradient Boosting Model Confusion Matrix:
- Correct predictions for instances predicted as "Survival" were 45, with 8 instances incorrectly predicted as "Death."
- Among instances predicted as "Death," 21 were correct, but 16 instances were incorrectly predicted as "Survival."

Conclusion:
    Both the SVM and GBM have a similar number of true negatives and true positives.
    The SVM model has fewer false positives (4) compared to the Gradient Boosting model (8).
    The Gradient Boosting model has a slightly higher number of false negatives (16) compared to the SVM model (16).
""")