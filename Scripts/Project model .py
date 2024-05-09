#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation and analysis
import seaborn as sns
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from sklearn.preprocessing import StandardScaler  # Feature scaling
from io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# Importing modules for machine learning from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# Importing evaluation metrics
from sklearn.metrics import classification_report,f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, auc
from sklearn.preprocessing import LabelEncoder
import shap
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#import data
loan_df = pd.read_csv("C:/Users/oracl/OneDrive/Desktop/Project/Data/Loan Default Prediction Dataset - NIKHIL.csv")


# In[3]:


#preview dataset
loan_df.head()


# In[4]:


#check the shape dataset
loan_df.shape


# In[5]:


#drop the first column loan ID as we would not be using it 
loan_df=loan_df.drop('LoanID', axis = 1)


# In[6]:


#check for the description and data types in the dataset
loan_df.info()


# In[7]:


#preview dataset
loan_df.describe()


# In[8]:


#check for the number of missing values in each column
print(loan_df.isnull().sum())


# In[9]:


loan_df.isna().sum()


# In[10]:


loan_df.duplicated().sum()


# In[11]:


#we have no missing values so we continue with the processing


# In[12]:


sns.set_palette("Set1")


# In[13]:


#our target variable for this exercise would be dafault so we would check for the distribution of defaulters
loan_df['Default'].value_counts()


# In[14]:


#we have 225,694 non defaulters and about 29,653 defaulters


# In[15]:


#we start the exploratory data analysis by visualizing the number of non-defaulters and defaulters


# In[16]:


# Calculate the percentage of each Default value
default_percentage = (loan_df['Default'].value_counts() / loan_df['Default'].count()) * 100

# Create a bar plot for the 'Default' column distribution as percentage
sns.barplot(x=default_percentage.index, y=default_percentage.values)

plt.title('Distribution of Default')
plt.xlabel('Default Status')  # Adding an x-label for clarity
plt.ylabel('Percentage')
plt.show()


# In[17]:


#Over 80% did not default and less than 20% of the applicants defaulted


# In[18]:


#visualizing the distribution of defaulters by their Age
sns.boxplot(data=loan_df, y='Age', x='Default')

plt.title('Distribution of Age by Default Status')
plt.xlabel('Default Status') 
plt.ylabel('Age')  
plt.show()


# In[19]:


#this shows the distribution of income
# Calculate bins based on the range of Income
bins = np.linspace(loan_df['Income'].min(), loan_df['Income'].max(), 10)

# Setup the FacetGrid
g = sns.FacetGrid(loan_df, col="Education", hue="Default", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Income', bins=bins, ec="k")  # 'ec' stands for edge color


g.add_legend()
plt.show()


# In[20]:


# Group by 'Default', count 'Education' values, and reshape the data
education_counts = loan_df.groupby('Default')['Education'].value_counts().unstack(fill_value=0)

# Print the resulting DataFrame
print(education_counts)


# In[21]:


# Include NaN values in the count
education_counts_including_na = loan_df['Education'].value_counts(dropna=False)

# Print the results including NaN values
print(education_counts_including_na)


# In[22]:


g = sns.catplot(data=loan_df, x='Education', y='Income', kind="box", height=5, aspect=2)
g.set_axis_labels("Education", "Income") 
g.fig.suptitle('Income Distribution by Education Level')
g.fig.subplots_adjust(top=0.9) 

plt.show()


# In[23]:


# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=loan_df, x='Education', hue='Default', palette='Set1') 

plt.title('Counts of Education Grouped by Default')
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Default Status') 


plt.xticks(rotation=45) 
plt.tight_layout()  

plt.show()


# In[24]:


# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=loan_df, x='EmploymentType', hue='Default', palette='Set2') 
plt.title('Counts of Employment Status Grouped by Default')
plt.xlabel('Employment Type') 
plt.ylabel('Count')
plt.legend(title='Default Status') 

# Rotate x-axis labels for better readability if necessary
plt.xticks(rotation=45)  
plt.tight_layout() 

plt.show()


# In[25]:


# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=loan_df, x='MaritalStatus', hue='Default', palette='bright') 

plt.title('Counts of Marital Status Grouped by Default')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(title='Default Status', loc='upper right')  

plt.xticks(rotation=45)  
plt.tight_layout()  

plt.show()


# In[26]:


# Calculate bins for histogram
bins = np.linspace(loan_df.CreditScore.min(), loan_df.CreditScore.max(), 10)

# Create the histogram plot using Seaborn's displot
plot = sns.displot(loan_df, x='CreditScore', bins=bins, hue='Default', element="step", palette="viridis")

# Adding enhancements for better visualization
plot.fig.suptitle('Credit Score Distribution by Default Status', y=1.03)  # Add a title to the plot and adjust vertical spacing
plt.xlabel('Credit Score')  # Label for the x-axis
plt.ylabel('Count')         # Label for the y-axis

# Adjust legend title and location
plot._legend.set_title('Default Status')
plt.legend(title='Default Status', labels=['Not Defaulted', 'Defaulted'], loc='upper left')

# Display the plot
plt.show()


# In[27]:


# Sample Data Loading
# loan_df = pd.read_csv('path_to_your_data.csv')

# Define a mapping for 'yes' and 'no' to binary values
binary_map = {'Yes': 1, 'No': 0}

# Columns to convert
columns_to_convert = ['HasMortgage', 'HasDependents', 'HasCoSigner']

# Check for nulls and unexpected values
for column in columns_to_convert:
    print(f"Unique values before mapping in {column}: {loan_df[column].unique()}")
    if loan_df[column].isnull().any():
        print(f"Warning: {column} contains null values. These will become NaN after mapping if not handled.")

# Apply mapping to convert 'yes' and 'no' to binary values
for column in columns_to_convert:
    loan_df[column] = loan_df[column].map(binary_map).fillna(loan_df[column])

# Validation of the results
for column in columns_to_convert:
    print(f"Unique values after mapping in {column}: {loan_df[column].unique()}")

# Display the first few rows to confirm changes
loan_df.head()


# In[28]:


# Identify non-numeric columns
non_numeric_columns = loan_df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns identified for encoding:", non_numeric_columns)

# Apply one-hot encoding
loan_df_encoded = pd.get_dummies(loan_df, columns=non_numeric_columns)

# Display the first few rows to confirm changes and see the newly created columns
loan_df_encoded.head()


# In[29]:


# Apply one-hot encoding
loan_df_encoded = pd.get_dummies(loan_df, columns=non_numeric_columns)

# Convert all columns to integers
loan_df_encoded = loan_df_encoded.astype(int)

# Drop specific dummy columns to avoid multicollinearity and reduce dimensionality
loan_df_encoded.drop(
    columns=['Education_PhD', 'EmploymentType_Full-time', 'LoanPurpose_Other'], 
    axis=1, 
    inplace=True
)

# Checking the modified DataFrame's structure and memory info
loan_df_encoded.info()


# In[30]:


#we check to see the variable correlation relationships
# Calculate the correlation matrix
correlation_matrix = loan_df_encoded.corr()

# Extract 'Default' correlations
default_correlation = correlation_matrix['Default'].drop('Default')  # Drop self-correlation

# Convert to DataFrame for easier plotting
default_corr_df = pd.DataFrame(default_correlation).reset_index()
default_corr_df.columns = ['Feature', 'Correlation Coefficient']

# Plotting
plt.figure(figsize=(15, 6))
sns.barplot(data=default_corr_df, x='Feature', y='Correlation Coefficient')
plt.xticks(rotation=90)  # Rotate the feature names for better visibility
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation Between Features and Target")
plt.show()


# In[31]:


#The relationships between the variables and the default appear top be non linear.
#We would use all features for the first experiment


# In[32]:


# Select Target and Features
y = loan_df_encoded['Default']
X = loan_df_encoded.drop(columns=['Default'])

# Print the shapes of X and y to confirm their dimensions
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# In[33]:


#normalize data
# Instantiate the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and then transform it
X_scaled = scaler.fit_transform(X)

# Display the first entry of the scaled features
print(X_scaled[0:1])


# In[34]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=14)

# Print the shapes of the training and testing data
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# In[35]:


#modelbuilding


# In[36]:


#Logistics Regression
log_reg1 = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
log_reg1


# In[37]:


# Predicting the test set results
y_pred = log_reg1.predict(X_test)


# In[38]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train, log_reg1.predict(X_train)))
print("Test set Accuracy: ",accuracy_score(y_test, y_pred))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test, y_pred, average='weighted')))


# In[39]:


# Classification Report
print(classification_report(y_test, y_pred))


# In[40]:


# Assuming 'log_reg1' is your logistic regression model and 'X_test', 'y_test' are your test datasets
y_pred = log_reg1.predict(X_test)

# Generate the confusion matrix
confusion_matrix_lr = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Extracting TP, FP, TN, FN from the confusion matrix
# For a binary classification the matrix shape is 2x2
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Output the results
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[41]:


# Predict probabilities on the test set
ylog_probs1 = log_reg1.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, ylog_probs1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[42]:


# XG BOOST CLASSIFIER
xgb1 = XGBClassifier()
# Fitting the model with the training Data
xgb1.fit(X_train, y_train)


# In[43]:


# Making Predictions
yhat_xgb1 = xgb1.predict(X_test)


# In[44]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train, xgb1.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat_xgb1))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test, yhat_xgb1, average='weighted')))


# In[45]:


# Classification Report
print(classification_report(y_test, yhat_xgb1))


# In[46]:


# Generate the confusion matrix using actual labels and predictions
confusion_matrix_lr = confusion_matrix(y_test, yhat_xgb1)

# Display the confusion matrix visually
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.title('Confusion Matrix for XGBoost Model')
plt.show()

# Extract TP, FP, TN, FN from the confusion matrix for binary classification
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print out the confusion matrix values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[47]:


# Predict probabilities on the test set
yxgb_probs1 = xgb1.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, yxgb_probs1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve(GNB Model)')
plt.legend(loc="lower right")
plt.show()


# In[48]:


# Gaussian Naive Bayes
gnb1 = GaussianNB()
gnb1.fit(X_train, y_train)
gnb_yhat1 = gnb1.predict(X_test)


# In[49]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train, gnb1.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, gnb_yhat1))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test, gnb_yhat1, average='weighted')))


# In[50]:


# Classification Report
print(classification_report(y_test, gnb_yhat1))


# In[51]:


# Assuming y_test is your true labels and gnb_yhat1 are the predictions from a model
confusion_matrix_lr = confusion_matrix(y_test, gnb_yhat1)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.show()

# confusion_matrix_lr is a 2x2 numpy array when dealing with binary classification
# [TN, FP]
# [FN, TP]

# Extracting True Negatives, False Positives, False Negatives, and True Positives
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print these values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[52]:


# Predict probabilities on the test set
ygnb_probs1 = gnb1.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, ygnb_probs1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve(GNB Model)')
plt.legend(loc="lower right")
plt.show()


# In[53]:


#Decision Tree
loanTree1 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
# Training our model
loanTree1.fit(X_train, y_train)


# In[54]:


# Prediction
yhat_tree1 = loanTree1.predict(X_test)


# In[55]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train, loanTree1.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat_tree1))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test, yhat_tree1, average='weighted')))


# In[56]:


# Classification Report
print(classification_report(y_test, yhat_tree1))


# In[57]:


# Assuming y_test contains the actual labels and yhat_tree1 contains the predictions from a decision tree model
confusion_matrix_lr = confusion_matrix(y_test, yhat_tree1)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.show()

# Extract True Negatives, False Positives, False Negatives, and True Positives from the matrix
# For a binary classification, the confusion matrix looks like this:
# [TN, FP]
# [FN, TP]
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print these values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[58]:


# Predict probabilities on the test set
ytree_probs1 = loanTree1.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, ytree_probs1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[59]:


#Decision Tree
dot_data = StringIO()
# Define feature names as a list
featureNames = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
                'HasMortgage', 'HasDependents', 'HasCoSigner', "Education_Bachelor's", 'Education_High School', "Education_Master's",
                'EmploymentType_Part-time', 'EmploymentType_Self-employed', 'EmploymentType_Unemployed', 'MaritalStatus_Divorced',
                'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Auto', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home']

# Export Decision Tree to Graphviz format
out = tree.export_graphviz(loanTree1, feature_names=featureNames, out_file=dot_data, class_names=[
                           '0', '1'], filled=True, special_characters=True, rotate=False)

# Create graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Write graph to a PNG file
filename = "LoanTree.png"
graph.write_png(filename)

# Display the PNG image using Matplotlib
img = mpimg.imread(filename)
plt.figure(figsize=(20, 30))
plt.imshow(img, interpolation='nearest')
plt.show()


# In[60]:


#Comapring results
model_names = ['Logistic Regression', 'Xg Boost',
              'Gaussian Naive Bayes', 'Decision Trees']
test_accuracy = [0.8857, 0.8858, 0.8855, 0.8852]
f1_scores = [0.8391, 0.8469, 0.8329, 0.8313]
auc_roc_scores = [0.73, 0.74, 0.74, 0.71]
compare_df = pd.DataFrame({
    "Model": model_names,
    "Test Accuracy": test_accuracy,
    "F1-Score": f1_scores,
    "AUC-ROC": auc_roc_scores
})
compare_df.head()


# In[61]:


ax = compare_df.plot(x="Model", kind="bar", figsize=(7, 6), width=0.2)
ax.set_ylabel("Scores (%)")
ax.set_title("Model Comparison")
plt.xticks(rotation=90)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -15), textcoords='offset points', rotation=90)
plt.tight_layout()
plt.show()


# In[62]:


# Define model names and performance metrics
model_names = ['Logistic Regression', 'Xg Boost', 'Gaussian Naive Bayes', 'Decision Trees']
test_accuracy = [88.65, 88.58, 88.63, 88.52]  # Convert to percentages
f1_scores = [83.91, 84.69, 83.79, 83.13]
auc_roc_scores = [75.00, 74.00, 75.00, 71.00]

# Create DataFrame
compare_df = pd.DataFrame({
    "Model": model_names * 3,  # Repeat model names for each metric
    "Value": test_accuracy + f1_scores + auc_roc_scores,
    "Metric": ["Test Accuracy"] * 4 + ["F1 Score"] * 4 + ["AUC-ROC"] * 4
})

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Create a figure for the plot
plt.figure(figsize=(10, 6))

# Create a bar plot
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=compare_df)

# Set titles and labels
plt.title('Model Comparison')
plt.ylabel('Percentage')
plt.xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Function to add labels on the bars
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

# Add labels to each bar
add_labels(ax)

# Display the plot
plt.tight_layout()
plt.show()


# In[63]:


#Improve the Model
#We motice that the classes are not balanced. We balance the classes using Smote
# Apply SMOTE to generate synthetic samples for the minority class
X_2 = loan_df_encoded.drop('Default', axis =1)
X_2.head()


# In[64]:


y_2 = loan_df_encoded.Default
y_2.head()


# In[65]:


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_2, y_2)


# In[66]:


X_resampled.shape


# In[67]:


y_resampled.shape


# In[68]:


# Calculate the percentage for each class in the resampled target variable
class_distribution = y_resampled.value_counts(normalize=True) * 100

# Create a DataFrame from the distribution for easier plotting
distribution_df = pd.DataFrame({'Default': class_distribution.index, 'Percentage': class_distribution.values})

# Plotting
plt.figure(figsize=(8, 4))
sns.barplot(x='Default', y='Percentage', data=distribution_df)
plt.title('Distribution of Default After SMOTE')
plt.xlabel('Default Status')
plt.ylabel('Percentage (%)')
plt.show()


# In[69]:


#The classes are balanced and split 50/50


# In[70]:


#Feature Selection
#Implement Mutual Information
#We use MI Scores to see relationship between features and Target
# Calculate Mutual Information for classification
mi_scores = mutual_info_classif(X_resampled, y_resampled)

# Print MI scores for each feature
for feature_name, score in zip(X_resampled.columns, mi_scores):
    print(f"{feature_name}: {score}")


# In[71]:


# Select features with MI score > 0.01
selected_features = X_resampled.columns[mi_scores >= 0.02]

print("Selected Features:")
print(selected_features)


# In[72]:


#Select only features with MI score >= 0.02 and we also included Income since we know this from domain knowledge
X_resampled = X_resampled[['Age', 'Income','MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm',
       'HasMortgage', 'HasDependents', 'HasCoSigner', "Education_Bachelor's",
       'Education_High School', "Education_Master's",
       'EmploymentType_Part-time', 'EmploymentType_Self-employed',
       'EmploymentType_Unemployed', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Auto',
       'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home']]
X_resampled.shape


# In[73]:


# normalize data
X_resampled = StandardScaler().fit(X_resampled).transform(X_resampled)
X_resampled[0:1]


# In[74]:


# Train Test Split
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.4, random_state=14)
print('Train set:', X_train_resampled.shape,  y_train_resampled.shape)
print('Test set:', X_test_resampled.shape,  y_test_resampled.shape)


# In[75]:


#Model building


# In[76]:


#Logistics Regression
log_reg2 = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_resampled, y_train_resampled)
log_reg2


# In[77]:


yhat_lr2 = log_reg2.predict(X_test_resampled)


# In[78]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train_resampled, log_reg2.predict(X_train_resampled)))
print("Test set Accuracy: ",accuracy_score(y_test_resampled, yhat_lr2))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test_resampled, yhat_lr2, average='weighted')))


# In[79]:


# Classification Report
print(classification_report(y_test_resampled, yhat_lr2))


# In[80]:


# Assuming y_test_resampled contains the resampled test data labels 
# and yhat_lr2 contains the predictions from a logistic regression model
confusion_matrix_lr = confusion_matrix(y_test_resampled, yhat_lr2)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Extract True Negatives, False Positives, False Negatives, and True Positives from the matrix
# In binary classification, confusion matrix structure:
# [TN, FP]
# [FN, TP]
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print these values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[81]:


# Predict probabilities on the test set
ylog_probs2 = log_reg2.predict_proba(X_test_resampled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_resampled, ylog_probs2)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[82]:


# XG BOOST
xgb2 = XGBClassifier()
# Fitting the model with the training Data
xgb2.fit(X_train_resampled, y_train_resampled)


# In[83]:


# Making Predictions
yhat_xgb2 = xgb2.predict(X_test_resampled)


# In[84]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train_resampled, xgb2.predict(X_train_resampled)))
print("Test set Accuracy: ", accuracy_score(y_test_resampled, yhat_xgb2))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test_resampled, yhat_xgb2, average='weighted')))


# In[85]:


# Classification Report
print(classification_report(y_test_resampled, yhat_xgb2))


# In[86]:


# Assuming y_test_resampled contains the resampled test data labels 
# and yhat_xgb2 contains the predictions from an XGBoost model
confusion_matrix_lr = confusion_matrix(y_test_resampled, yhat_xgb2)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.title("Confusion Matrix for XGBoost Model")
plt.show()

# Extract True Negatives, False Positives, False Negatives, and True Positives from the matrix
# In binary classification, confusion matrix structure:
# [TN, FP]
# [FN, TP]
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print these values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[87]:


# Predict probabilities on the test set
yxgb_probs2 = xgb2.predict_proba(X_test_resampled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_resampled, yxgb_probs2)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve(GNB Model)')
plt.legend(loc="lower right")
plt.show()


# In[88]:


# Gaussian Naive Bayes
gnb2 = GaussianNB()
gnb2.fit(X_train_resampled, y_train_resampled)
gnb_yhat2 = gnb2.predict(X_test_resampled)


# In[89]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train_resampled, gnb2.predict(X_train_resampled)))
print("Test set Accuracy: ", accuracy_score(y_test_resampled, gnb_yhat2))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test_resampled, gnb_yhat2, average='weighted')))


# In[90]:


# Classification Report
print(classification_report(y_test_resampled, gnb_yhat2))


# In[91]:


# Generate the confusion matrix from the model's predictions
confusion_matrix_lr = confusion_matrix(y_test_resampled, gnb_yhat2)

# Display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.show()

# Extracting TP, FP, TN, FN from the confusion matrix
# For a binary classification the matrix shape is 2x2
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Output the results
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[92]:


# Predict probabilities on the test set
ygnb_probs2 = gnb2.predict_proba(X_test_resampled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_resampled, ygnb_probs2)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve(GNB Model)')
plt.legend(loc="lower right")
plt.show()


# In[93]:


#Decision Tree
loanTree2 = DecisionTreeClassifier(criterion="entropy", max_depth=20)
# Training our model
loanTree2.fit(X_train_resampled, y_train_resampled)


# In[94]:


# Prediction
yhat_tree2 = loanTree2.predict(X_test_resampled)


# In[95]:


# Checking our Training and Test set accuracy
print("Train set Accuracy: ", accuracy_score(
    y_train_resampled, loanTree2.predict(X_train_resampled)))
print("Test set Accuracy: ", accuracy_score(y_test_resampled, yhat_tree2))
#F1 Score 
print('F1 Score: {:.4f}'.format(
    f1_score(y_test_resampled, yhat_tree2, average='weighted')))


# In[96]:


# Classification Report
print(classification_report(y_test, yhat_tree1))


# In[97]:


# Assuming y_test_resampled contains the resampled test data labels 
# and yhat_tree2 contains the predictions from a decision tree model
confusion_matrix_lr = confusion_matrix(y_test_resampled, yhat_tree2)

# Display the confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix_lr).plot()
plt.title("Confusion Matrix for Decision Tree Model")
plt.show()

# Extract True Negatives, False Positives, False Negatives, and True Positives from the matrix
# In binary classification, the confusion matrix structure is typically:
# [TN, FP]
# [FN, TP]
TN, FP, FN, TP = confusion_matrix_lr.ravel()

# Print these values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# In[98]:


# Predict probabilities on the test set
ytree_probs2 = loanTree2.predict_proba(X_test_resampled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_resampled, ytree_probs2)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[99]:


#Comapring all model results
model_names = ['Logistic Regression Model 1', 'Xg Boost Model 1',
               'Gaussian Naive Bayes Model 1', 'Decision Trees Model 1', 'Logistic Regression Model 2', 'Xg Boost Model 2',
               'Gaussian Naive Bayes Model 2', 'Decision Trees Model 2']
test_accuracy = [0.8857, 0.8858, 0.8855,
                 0.8852, 0.8913, 0.9181, 0.8751, 0.8515]
f1_scores = [0.8329, 0.8469, 0.8329, 0.8313, 0.8912, 0.9179, 0.8745, 0.8516]
auc_roc_scores = [0.73, 0.74, 0.74, 0.71, 0.95, 0.96, 0.93, 0.87]
compare_df = pd.DataFrame({
    "Models": model_names,
    "Test Accuracy": test_accuracy,
    "F1-Score": f1_scores,
    "AUC-ROC": auc_roc_scores
})
compare_df


# In[100]:


# Set up the figure and axes
fig, ax = plt.subplots(figsize=(15, 8))

# Define bar width and positions
bar_width = 0.25
index = np.arange(len(compare_df['Models']))

# Create bars for each metric
bars1 = ax.bar(index, compare_df['Test Accuracy'], bar_width, label='Test Accuracy', color='b')
bars2 = ax.bar(index + bar_width, compare_df['F1-Score'], bar_width, label='F1 Score', color='r')
bars3 = ax.bar(index + 2 * bar_width, compare_df['AUC-ROC'], bar_width, label='AUC-ROC', color='g')

# Add labels, title and axes ticks
ax.set_xlabel('Models')
ax.set_ylabel('Scores (%)')
ax.set_title('Comparison of Model Metrics')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(compare_df['Models'])
ax.legend()

# Label with the numeric value above each bar
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Improve layout and display the plot
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:




