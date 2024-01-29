#!/usr/bin/env python
# coding: utf-8

# # Load and understanding the data

# In[1]:


# import pandas
import pandas as pd              
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)

# define support function
def inspect_data(df, col=None, n_rows=5):
    # check data shape
    print(f'data shape: {df.shape}')
    
    # define columns
    if col is None:
        col = df.columns
    
    # check data head, use display function to display dataframe
    display(df[col].head(n_rows))
    
def check_missing(df, cut_off=0, sort=True):
    freq=df.isnull().sum()
    percent=df.isnull().sum()/df.shape[0]*100
    types=df.dtypes
    unique=df.apply(pd.unique).to_frame(name='Unique Values')['Unique Values']
    unique_counts = df.nunique(dropna=False)
    df_miss=pd.DataFrame({'missing_percentage':percent,'missing_frequency':freq,'types':types,'count_value':unique_counts,
                          'unique_values':unique})
    if sort:df_miss.sort_values(by='missing_frequency',ascending=False, inplace=True)
    return df_miss[df_miss['missing_percentage']>=cut_off]

# load file
# app_test = pd.read_csv('application_test.csv')
app_train = pd.read_csv('application_train.csv')
# bureau = pd.read_csv('bureau.csv')
# bur_bal = pd.read_csv('bureau_balance.csv')
# cc_bal = pd.read_csv('credit_card_balance.csv')
# ins_pay = pd.read_csv('installments_payments.csv')
# pc_bal = pd.read_csv('POS_CASH_balance.csv')
# prev_app = pd.read_csv('previous_application.csv')
# samp_sub = pd.read_csv('sample_submission.csv')


# In[2]:


inspect_data(app_train)


# In[3]:


check_missing(app_train)


# # Feature engineering and selection

# In[4]:


# drop column who have high threshold of missing value (20% or more)
threshold = len(app_train) * 0.8
model_base = app_train.dropna(axis=1, thresh=threshold)


# In[5]:


inspect_data(model_base)


# In[6]:


check_missing(model_base)


# In[7]:


# drop column realted to 'EXT_SOURCE_1' column
model_base = model_base.drop(columns=['EXT_SOURCE_3','EXT_SOURCE_2'])


# In[8]:


inspect_data(model_base)


# In[9]:


# drop column who have all unique value
unique_cols = [col for col in model_base.columns if model_base[col].nunique() == len(model_base)]
model_base = model_base.drop(columns=unique_cols)


# In[10]:


inspect_data(model_base)


# In[11]:


check_missing(model_base)


# In[12]:


# drop column with only 1 unique value
single_value_cols = [col for col in model_base.columns if model_base[col].nunique() == 1]
model_base = model_base.drop(columns=single_value_cols)


# In[13]:


inspect_data(model_base)


# In[14]:


check_missing(model_base)


# In[15]:


# checking column with 1 dominant category, 80% will be the threshold
for col in model_base.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = model_base[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        print(value_counts_percentage)
        print('\n')


# In[16]:


# drop column with 1 dominant category automaticaly, 80% will be the threshold
for col in model_base.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = model_base[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        model_base = model_base.drop(columns=col)


# In[17]:


inspect_data(model_base)


# In[18]:


check_missing(model_base)


# In[19]:


# converting some column type to object
model_base['FLAG_DOCUMENT_2'] = model_base['FLAG_DOCUMENT_2'].astype('object')
model_base['FLAG_DOCUMENT_3'] = model_base['FLAG_DOCUMENT_3'].astype('object')
model_base['FLAG_DOCUMENT_4'] = model_base['FLAG_DOCUMENT_4'].astype('object')
model_base['FLAG_DOCUMENT_5'] = model_base['FLAG_DOCUMENT_5'].astype('object')
model_base['FLAG_DOCUMENT_6'] = model_base['FLAG_DOCUMENT_6'].astype('object')
model_base['FLAG_DOCUMENT_7'] = model_base['FLAG_DOCUMENT_7'].astype('object')
model_base['FLAG_DOCUMENT_8'] = model_base['FLAG_DOCUMENT_8'].astype('object')
model_base['FLAG_DOCUMENT_9'] = model_base['FLAG_DOCUMENT_9'].astype('object')
model_base['FLAG_DOCUMENT_10'] = model_base['FLAG_DOCUMENT_10'].astype('object')
model_base['FLAG_DOCUMENT_11'] = model_base['FLAG_DOCUMENT_11'].astype('object')
model_base['FLAG_DOCUMENT_12'] = model_base['FLAG_DOCUMENT_12'].astype('object')
model_base['FLAG_DOCUMENT_13'] = model_base['FLAG_DOCUMENT_13'].astype('object')
model_base['FLAG_DOCUMENT_14'] = model_base['FLAG_DOCUMENT_14'].astype('object')
model_base['FLAG_DOCUMENT_15'] = model_base['FLAG_DOCUMENT_15'].astype('object')
model_base['FLAG_DOCUMENT_16'] = model_base['FLAG_DOCUMENT_16'].astype('object')
model_base['FLAG_DOCUMENT_17'] = model_base['FLAG_DOCUMENT_17'].astype('object')
model_base['FLAG_DOCUMENT_18'] = model_base['FLAG_DOCUMENT_18'].astype('object')
model_base['FLAG_DOCUMENT_19'] = model_base['FLAG_DOCUMENT_19'].astype('object')
model_base['FLAG_DOCUMENT_20'] = model_base['FLAG_DOCUMENT_20'].astype('object')
model_base['FLAG_DOCUMENT_21'] = model_base['FLAG_DOCUMENT_21'].astype('object')
model_base['LIVE_CITY_NOT_WORK_CITY'] = model_base['LIVE_CITY_NOT_WORK_CITY'].astype('object')
model_base['FLAG_MOBIL'] = model_base['FLAG_MOBIL'].astype('object')
model_base['FLAG_EMP_PHONE'] = model_base['FLAG_EMP_PHONE'].astype('object')
model_base['FLAG_WORK_PHONE'] = model_base['FLAG_WORK_PHONE'].astype('object')
model_base['FLAG_CONT_MOBILE'] = model_base['FLAG_CONT_MOBILE'].astype('object')
model_base['FLAG_PHONE'] = model_base['FLAG_PHONE'].astype('object')
model_base['FLAG_EMAIL'] = model_base['FLAG_EMAIL'].astype('object')
model_base['REG_REGION_NOT_LIVE_REGION'] = model_base['REG_REGION_NOT_LIVE_REGION'].astype('object')
model_base['REG_REGION_NOT_WORK_REGION'] = model_base['REG_REGION_NOT_WORK_REGION'].astype('object')
model_base['LIVE_REGION_NOT_WORK_REGION'] = model_base['LIVE_REGION_NOT_WORK_REGION'].astype('object')
model_base['REG_CITY_NOT_LIVE_CITY'] = model_base['REG_CITY_NOT_LIVE_CITY'].astype('object')
model_base['REG_CITY_NOT_WORK_CITY'] = model_base['REG_CITY_NOT_WORK_CITY'].astype('object')


# In[20]:


check_missing(model_base)


# In[21]:


model_base['REGION_RATING_CLIENT'] = model_base['REGION_RATING_CLIENT'].astype('object')
model_base['REGION_RATING_CLIENT_W_CITY'] = model_base['REGION_RATING_CLIENT_W_CITY'].astype('object')


# In[22]:


check_missing(model_base)


# In[23]:


# checking the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = model_base.corr()

# create a heatmap
plt.figure(figsize=(30, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 6})
plt.title('Correlation Heatmap')
plt.show()


# In[24]:


# Filter the correlation matrix based on the threshold > 0.4 or < -0.4
filtered_matrix = correlation_matrix[(correlation_matrix > 0.4) | (correlation_matrix < -0.4)]

# create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 8})
plt.title('Correlation Heatmap (|Correlation| > 0.4)')
plt.show()


# In[25]:


# drop column that have high corelation with other column (correlation coefficient >0.4 or <-0.4), choose 1 column to stay
model_base = model_base.drop(columns=['CNT_CHILDREN','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','OBS_60_CNT_SOCIAL_CIRCLE',
                                      'DEF_60_CNT_SOCIAL_CIRCLE'])


# In[26]:


correlation_matrix = model_base.corr()

# Filter the correlation matrix based on the threshold
filtered_matrix = correlation_matrix[(correlation_matrix > 0.4) | (correlation_matrix < -0.4)]

# create a heatmap
plt.figure(figsize=(24 , 20))
sns.heatmap(filtered_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 8})
plt.title('Correlation Heatmap (|Correlation| > 0.4)')
plt.show()


# In[27]:


inspect_data(model_base)


# In[28]:


check_missing(model_base)


# In[29]:


# drop categorical column with high cardinality and by expert judgement
model_base = model_base.drop(columns=['ORGANIZATION_TYPE'])


# In[30]:


# convert the negative value to positive
model_base['DAYS_LAST_PHONE_CHANGE'] = model_base['DAYS_LAST_PHONE_CHANGE'].abs()
model_base['DAYS_EMPLOYED'] = model_base['DAYS_EMPLOYED'].abs()
model_base['DAYS_REGISTRATION'] = model_base['DAYS_REGISTRATION'].abs()
model_base['DAYS_ID_PUBLISH'] = model_base['DAYS_ID_PUBLISH'].abs()


# In[31]:


check_missing(model_base)


# In[32]:


inspect_data(model_base)


# In[33]:


# handling missing value
'''
# defining columns with missing value
categorical_columns = []
numerical_columns = ['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK',
                     'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','OBS_30_CNT_SOCIAL_CIRCL','DEF_30_CNT_SOCIAL_CIRCLE',
                    'CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE']

# impute categorical columns with mode because it helps to preserve the overall distribution of categorical data and is 
# suitable when the missing values are expected to occur at random.
for col in categorical_columns:
    mode_value = model_base[col].mode()[0]
    model_base[col].fillna(mode_value, inplace=True)

# impute numerical columns with median because it less sensitive to extreme values (outliers)
for col in numerical_columns:
    median_value = model_base[col].median()
    model_base[col].fillna(median_value, inplace=True)
'''
    
model_base = model_base.dropna(subset=['AMT_REQ_CREDIT_BUREAU_YEAR','DEF_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE',
                                       'CNT_FAM_MEMBERS'])


# In[34]:


inspect_data(model_base)


# In[35]:


check_missing(model_base)


# In[36]:


# drop column with only 1 unique value
single_value_cols = [col for col in model_base.columns if model_base[col].nunique() == 1]
model_base = model_base.drop(columns=single_value_cols)


# In[37]:


# checking column with 1 dominant category, 80% will be the threshold
for col in model_base.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = model_base[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        print(value_counts_percentage)
        print('\n')


# In[38]:


# drop column with 1 dominant category automaticaly, 80% will be the threshold
for col in model_base.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = model_base[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        model_base = model_base.drop(columns=col)


# In[39]:


# checking column with 1 dominant category, 80% will be the threshold
for col in model_base.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = model_base[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        print(value_counts_percentage)
        print('\n')


# In[40]:


# Filter the correlation matrix based on the threshold > 0.4 or < -0.4
filtered_matrix = correlation_matrix[(correlation_matrix > 0.4) | (correlation_matrix < -0.4)]

# create a heatmap
plt.figure(figsize=(36, 30))
sns.heatmap(filtered_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 8})
plt.title('Correlation Heatmap (|Correlation| > 0.4)')
plt.show()


# In[41]:


correlation_matrix = model_base.corr()

# create a heatmap
plt.figure(figsize=(30, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 6})
plt.title('Correlation Heatmap')
plt.show()


# In[42]:


inspect_data(model_base)


# In[43]:


check_missing(model_base)


# # Scaling and encoding

# In[44]:


# define categorical columns
categorical_cols = [col for col in model_base.select_dtypes(include='object').columns.tolist()]

# develop onehot encoding dataframe
onehot = pd.get_dummies(model_base[categorical_cols], drop_first=True)


# In[45]:


onehot.head()


# In[46]:


onehot.shape


# In[47]:


# import library
from sklearn.preprocessing import StandardScaler

model_num = model_base.drop(columns=['TARGET'])

# define numerical columns
numerical_cols = [col for col in model_num.columns.tolist() if col not in categorical_cols]

# develop standardscaler dataframe
ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(model_num[numerical_cols]), columns=numerical_cols)


# In[48]:


std.head()


# In[49]:


std.shape


# In[50]:


# resetting the index of each dataframe to make sure the row numbers doesn't add up
df_reset = model_base.reset_index(drop=True)
onehot_reset = onehot.reset_index(drop=True)
std_reset = std.reset_index(drop=True)

# develop dataframe for machine learning modeling
df_model = pd.concat([onehot_reset, std_reset, df_reset[['TARGET']]], axis=1)


# In[51]:


# check the model dataframe
df_model.head()


# In[52]:


df_model.shape


# # Train test split

# In[53]:


# define features (X) and target variable (Y)
X = df_model
Y = df_model['TARGET']

# get the column names as feature names
feature_names = X.columns.tolist()


# In[54]:


# import library
from sklearn.model_selection import train_test_split

# split the data into training and testing sets (80% training, 20% testing for n between 100,000 to 1,000,000)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[55]:


# check dataframe shape
X_train.shape, X_test.shape


# # Oversampling with SMOTE

# In[56]:


# count the occurrences of each loan category
loan_category_counts = X_train['TARGET'].value_counts()

# set colors for different loan categories
colors = ['green', 'red']

# plot the distribution of loan label
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Target')

# set x-axis ticks to only show 0 and 1
plt.xticks([0, 1])

# add values on top of the bars
for i, count in enumerate(loan_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

plt.show()


# In[57]:


# import library
from imblearn.over_sampling import SMOTE

# conducting oversampling using SMOTE
smote = SMOTE(random_state=42)
X_train, Y_train = smote.fit_resample(X, Y)


# In[58]:


# count the occurrences of each loan category
loan_category_counts = X_train['TARGET'].value_counts()

# set colors for different loan categories
colors = ['green', 'red']

# plot the distribution of loan label
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Target')

# set x-axis ticks to only show 0 and 1
plt.xticks([0, 1])

# add values on top of the bars
for i, count in enumerate(loan_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

plt.show()


# # Train, test, evaluate, and hyperparameter tuning

# In[59]:


# droping target column
X_train = X_train.drop('TARGET', axis=1)
X_test = X_test.drop('TARGET', axis=1)


# In[60]:


# check dataframe shape
X_train.shape, X_test.shape


# In[61]:


# import library
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# initialize different models
results = {}
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# initialize dictionary to store classification reports
classification_reports = {}
model_names = []
accuracies = []

# train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, Y_train)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    classification_rep = classification_report(
        Y_test, Y_pred, target_names=['1', '0'], zero_division=1  # handle zero division
    )

    # store the classification report in the dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test, Y_pred)

    model_names.append(model_name)
    accuracies.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)


# In[62]:


# create a bar plot to visualize accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color='skyblue')

# add annotations in the middle of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

# set labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')

# set y-axis limits to 0-1 for accuracy percentage
plt.ylim(0.6, 1.2)

# rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.tight_layout()

# add horizontal axis line at y=0.9
plt.axhline(0.9, color='black', linewidth=0.8)

plt.show()


# In[63]:


# initialize model
import numpy as np

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, Y_train)

y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[64]:


# import library
from sklearn.metrics import roc_curve, roc_auc_score

# initialize auc
fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[65]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[66]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[67]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# In[68]:


# generating feature importances
arr_feature_importances = rfc.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features


# In[69]:


# showing top 10 feature importance on bar chart
df_top_features = df_all_features.head(10).sort_values(by='importance', ascending=True)

# plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(df_top_features['feature'], df_top_features['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')

# annotating the bars with the importance values
for index, value in enumerate(df_top_features['importance']):
    plt.text(value, index, f'{value:.4f}', va='center')

plt.show()


# In[70]:


# uncommet block code below for hyperparameter tuning random forest
'''
# Hyperparameter tuning on RandomForest
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid,
                                   n_iter=10, scoring='accuracy', cv=5, random_state=42)
random_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
test_accuracy = random_search.best_estimator_.score(X_test, Y_test)
print("Test Accuracy:", test_accuracy)
'''


# In[71]:


lr = LogisticRegression(random_state=42)
lr.fit(X_train, Y_train)

y_pred_proba = lr.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[72]:


fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[73]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[74]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[75]:


model = LogisticRegression()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# In[76]:


# uncommet block code below for hyperparameter tuning logistic regresion
'''
# Hyperparameter tuning on logistic regresion
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Accuracy:", test_accuracy)
'''


# In[77]:


dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, Y_train)

y_pred_proba = dtc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[78]:


fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[79]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[80]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[81]:


model = DecisionTreeClassifier()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# In[82]:


# uncommet block code below for hyperparameter tuning decision tree
'''
# Hyperparameter tuning on Decision Tree
# Define the parameter grid
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Create the Decision Tree model
dtc = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Accuracy:", test_accuracy)
'''


# In[83]:


gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, Y_train)

y_pred_proba = gbc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[84]:


fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[85]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[86]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[87]:


model = GradientBoostingClassifier()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# In[88]:


# uncommet block code below for hyperparameter tuning gradient boosting
'''
# Hyperparameter tuning on Gradient Boosting
# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set
test_accuracy = grid_search.best_estimator_.score(X_test, Y_test)
print("Test Accuracy:", test_accuracy)
'''


# In[89]:


knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)

y_pred_proba = knc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[90]:


fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[91]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[92]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[93]:


model = KNeighborsClassifier()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# In[94]:


# uncommet block code below for hyperparameter tuning k nearest
'''
# Hyperparameter tuning on k-Nearest Neighbors
# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance (L1), 2 for Euclidean distance (L2)
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=knc, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Accuracy:", test_accuracy)
'''


# # Prediction with RandomForest Classification Model

# In[95]:


# load the test data
app_test = pd.read_csv('application_test.csv')


# In[96]:


inspect_data(app_test)


# In[97]:


# Find common columns
common_columns = set(app_test.columns).intersection(model_base.columns)

# Create a new DataFrame with common columns
data_test = pd.DataFrame({col: pd.concat([app_test[col], model_base[col]], ignore_index=True) for col in common_columns})

# Filter rows from app_test only and keep the original index
data_test = data_test.loc[:len(app_test)-1]


# In[98]:


inspect_data(data_test)


# In[99]:


check_missing(data_test)


# In[100]:


# change data type as in fit process
data_test['FLAG_DOCUMENT_3'] = data_test['FLAG_DOCUMENT_3'].astype('object')
data_test['FLAG_PHONE'] = data_test['FLAG_PHONE'].astype('object')
data_test['REG_CITY_NOT_WORK_CITY'] = data_test['REG_CITY_NOT_WORK_CITY'].astype('object')
data_test['REGION_RATING_CLIENT'] = data_test['REGION_RATING_CLIENT'].astype('object')
data_test['REGION_RATING_CLIENT_W_CITY'] = data_test['REGION_RATING_CLIENT_W_CITY'].astype('object')


# In[101]:


inspect_data(data_test)


# In[102]:


check_missing(data_test)


# In[103]:


# change negative value to absolute as in fit process
data_test['DAYS_LAST_PHONE_CHANGE'] = data_test['DAYS_LAST_PHONE_CHANGE'].abs()
data_test['DAYS_EMPLOYED'] = data_test['DAYS_EMPLOYED'].abs()
data_test['DAYS_REGISTRATION'] = data_test['DAYS_REGISTRATION'].abs()
data_test['DAYS_ID_PUBLISH'] = data_test['DAYS_ID_PUBLISH'].abs()


# In[104]:


inspect_data(data_test)


# In[105]:


check_missing(data_test)


# In[106]:


# handling missing value
numerical_columns = ['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK',
                     'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE']

for col in numerical_columns:
    median_value = data_test[col].median()
    data_test[col].fillna(median_value, inplace=True)


# In[107]:


inspect_data(data_test)


# In[108]:


check_missing(data_test)


# In[109]:


# define categorical columns
categorical_cols = [col for col in data_test.select_dtypes(include='object').columns.tolist()]

# develop onehot encoding dataframe
onehot_test = pd.get_dummies(data_test[categorical_cols], drop_first=True)


# In[110]:


onehot_test.head()


# In[111]:


inspect_data(onehot_test)


# In[112]:


# develop standard scaler for numerical columns
numerical_cols = [col for col in data_test.columns.tolist() if col not in categorical_cols]

ss_test = StandardScaler()
std_test = pd.DataFrame(ss_test.fit_transform(data_test[numerical_cols]), columns=numerical_cols)


# In[113]:


inspect_data(std_test)


# In[114]:


std_test.head()


# In[115]:


# concat the test data
onehot_rst = onehot_test.reset_index(drop=True)
std_rst = std_test.reset_index(drop=True)

data_pred = pd.concat([onehot_rst, std_rst], axis=1)


# In[116]:


inspect_data(data_pred)


# In[117]:


# adding features that exist in fit porcess but missing in test data, fill value with 0
data_pred['CODE_GENDER_XNA'] = 0
data_pred['NAME_INCOME_TYPE_Maternity leave'] = 0


# In[118]:


inspect_data(data_pred)


# In[119]:


check_missing(data_pred)


# In[120]:


# handling missing value due to features added, uncomment if needed
# data_pred.fillna(0, inplace=True)


# In[121]:


# common features extraction
common_features = set(data_pred.columns).intersection(rfc.feature_names_in_)

# filtering columns
data_pred = data_pred[rfc.feature_names_in_]


# In[122]:


inspect_data(data_pred)


# In[123]:


check_missing(data_pred)


# In[127]:


# making prediction
predict = pd.Series(rfc.predict(data_pred), name="TARGET").astype(int)

# concat the SK ID CURR with the result and set the index
results = pd.concat([app_test['SK_ID_CURR'], predict], axis=1)

results.head()


# In[128]:


inspect_data(results)


# In[129]:


check_missing(results)


# In[130]:


# uncomment code below to save result to csv
#results.to_csv("predict_application.csv", index = False)

