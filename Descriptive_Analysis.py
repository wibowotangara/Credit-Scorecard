#!/usr/bin/env python
# coding: utf-8

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


# In[4]:


# change target value

decode_map = {0: "No Difficulties", 1: "Payment Difficulties"}
def decode_sentiment(label):
    return decode_map[int(label)]

app_train['TARGET'] = app_train['TARGET'].apply(lambda x: decode_sentiment(x))


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# count the occurrences of each loan category
loan_category_counts = app_train['TARGET'].value_counts()

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


# In[6]:


loan_category_counts = app_train['TARGET'].value_counts()

# Set colors for different loan categories
colors = ['green', 'red']

# Plot a pie chart for the distribution of loan label
plt.pie(loan_category_counts.values, labels=loan_category_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Target')

plt.show()


# In[7]:


sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
sns.set_context('paper', font_scale=1)

ax.set_title('Clients Repayment Abilities By Gender\n', fontweight='bold', fontsize=14)
sns.countplot(x='CODE_GENDER', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'}, ax=ax)

plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[8]:


gender = app_train.groupby(by=['CODE_GENDER', 'TARGET'], as_index=False)['TARGET'].count()
gender = gender.rename(columns={'TARGET': 'SK_ID_CURR'})
gender['Percentage'] = gender.groupby('CODE_GENDER')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Gender')
gender = gender.sort_values(by=['CODE_GENDER', 'SK_ID_CURR'], ascending=[True, False])
gender.style.background_gradient(cmap='Blues')


# In[9]:


sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
sns.set_context('paper', font_scale=1)

ax.set_title('Clients Repayment Abilities by Car Ownership\n', fontweight='bold', fontsize=14)
sns.countplot(x='FLAG_OWN_CAR', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'}, ax=ax)

plt.xlabel('Car Ownership Status')
plt.ylabel('Count')
plt.show()


# In[10]:


car = app_train.groupby(by=['FLAG_OWN_CAR', 'TARGET'], as_index=False)['TARGET'].count()
car = car.rename(columns={'TARGET': 'SK_ID_CURR'})
car['Percentage'] = car.groupby('FLAG_OWN_CAR')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Car Ownership')
car = car.sort_values(by=['FLAG_OWN_CAR', 'SK_ID_CURR'], ascending=[True, False])
car.style.background_gradient(cmap='Blues')


# In[11]:


sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
sns.set_context('paper', font_scale=1)

ax.set_title('Clients Repayment Abilities by Realty Ownership\n', fontweight='bold', fontsize=14)
sns.countplot(x='FLAG_OWN_REALTY', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'}, ax=ax)

plt.xlabel('Realty Ownership Status')
plt.ylabel('Count')
plt.show()


# In[12]:


real = app_train.groupby(by=['FLAG_OWN_REALTY', 'TARGET'], as_index=False)['TARGET'].count()
real = real.rename(columns={'TARGET': 'SK_ID_CURR'})
real['Percentage'] = real.groupby('FLAG_OWN_REALTY')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Realty Ownership')
real = real.sort_values(by=['FLAG_OWN_REALTY', 'SK_ID_CURR'], ascending=[True, False])
real.style.background_gradient(cmap='Blues')


# In[13]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', data=app_train, hue='TARGET', 
                palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'})

plt.title('Relationship Between Income and Loan Amount')
plt.xlabel('Income Amount')
plt.ylabel('Loan Amount')

plt.show()


# In[14]:


plt.figure(figsize=(10, 8))
sns.boxplot(x='TARGET', y='CNT_FAM_MEMBERS', data=app_train, 
            palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'})

plt.title('Relationship Between Family Members and Repayment Capabilities')
plt.xlabel('Payment Status')
plt.ylabel('Number of Family Members')

plt.show()


# In[15]:


plt.figure(figsize=(10, 8))
sns.violinplot(x='TARGET', y='CNT_FAM_MEMBERS', data=app_train, 
               palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'}, inner='quartile')

plt.title('Relationship Between Family Members and Repayment Capabilities')
plt.xlabel('Payment Status')
plt.ylabel('Number of Family Members')

plt.show()


# In[16]:


plt.figure(figsize=(10, 8))
sns.countplot(x='NAME_FAMILY_STATUS', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'})
    
plt.title('Relationship Between Family Status and Repayment Capabilities')
plt.xlabel('Family Status')
plt.ylabel('Count')
plt.legend(title='Payment Difficulties')

plt.show()


# In[17]:


family = app_train.groupby(by=['NAME_FAMILY_STATUS', 'TARGET'], as_index=False)['TARGET'].count()
family = family.rename(columns={'TARGET': 'SK_ID_CURR'})
family['Percentage'] = family.groupby('NAME_FAMILY_STATUS')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Family Status')
family = family.sort_values(by=['NAME_FAMILY_STATUS', 'SK_ID_CURR'], ascending=[True, False])
family.style.background_gradient(cmap='Blues')


# In[18]:


plt.figure(figsize=(10, 8))
sns.countplot(x='NAME_EDUCATION_TYPE', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'})
    
plt.title('Relationship Between Education and Repayment Capabilities')
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Payment Difficulties')

plt.show()


# In[19]:


edu = app_train.groupby(by=['NAME_EDUCATION_TYPE', 'TARGET'], as_index=False)['TARGET'].count()
edu = edu.rename(columns={'TARGET': 'SK_ID_CURR'})
edu['Percentage'] = edu.groupby('NAME_EDUCATION_TYPE')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Education')
edu = edu.sort_values(by=['NAME_EDUCATION_TYPE', 'SK_ID_CURR'], ascending=[True, False])
edu.style.background_gradient(cmap='Blues')


# In[20]:


plt.figure(figsize=(10, 8))
sns.countplot(x='NAME_INCOME_TYPE', data=app_train, hue='TARGET', 
              palette={'No Difficulties': 'green', 'Payment Difficulties': 'red'})
    
plt.title('Relationship Between Income Type and Repayment Capabilities')
plt.xlabel('Income Type')
plt.ylabel('Count')
plt.legend(title='Payment Difficulties')

plt.show()


# In[21]:


job = app_train.groupby(by=['NAME_INCOME_TYPE', 'TARGET'], as_index=False)['TARGET'].count()
job = job.rename(columns={'TARGET': 'SK_ID_CURR'})
job['Percentage'] = job.groupby('NAME_INCOME_TYPE')['SK_ID_CURR'].transform(lambda x: x / x.sum() * 100)

print('Clients Repayment Abilities By Income Type')
job = job.sort_values(by=['NAME_INCOME_TYPE', 'SK_ID_CURR'], ascending=[True, False])
job.style.background_gradient(cmap='Blues')

