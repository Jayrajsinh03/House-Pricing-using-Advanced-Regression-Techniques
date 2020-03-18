#!/usr/bin/env python
# coding: utf-8

# In[88]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the Dataset

# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/Jordan3296/Kaggle-Competitions/master/House%20Prices%3A%20Advanced%20Regression%20Techniques/train.csv')


# In[3]:


df.head()


# In[5]:


df.columns


# Creating a heatmap of Features 

# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=True)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


p = df.columns.to_series().groupby(df.dtypes).groups
print(p)


# Data Visualization for Important Features

# In[11]:


#YrSold : Year Sold

df['YrSold'].value_counts().plot(kind = 'bar')


# In[12]:


#MoSold: Month Sold

df['MoSold'].value_counts().plot(kind = 'bar')


# In[13]:


#MiscVal: $value of miscellaneous feature

df['MiscVal'].value_counts().plot(kind = 'bar')


# In[14]:


#PoolArea: Pool area in square ft
df['PoolArea'].value_counts().plot(kind = 'bar')


# In[15]:


#3SsnPorch: 3 season porch area in square feet
df['3SsnPorch'].value_counts().plot(kind='bar')


# In[17]:


#Fireplaces: Number of Fireplaces in a house
df.Fireplaces.value_counts().plot(kind = 'bar')


# In[18]:


#TotRmsAbvGrd: Total rooms above Grade
df.TotRmsAbvGrd.value_counts().plot(kind = 'bar')


# In[19]:


df.KitchenAbvGr.value_counts().plot(kind = 'bar')


# In[20]:


df.BedroomAbvGr.value_counts().plot(kind = 'bar')


# In[21]:


#HalfBath: Half baths above grade
df.HalfBath.value_counts().plot(kind = 'bar')


# In[22]:


#FullBath: Full bathrooms above grade
df.FullBath.value_counts().plot(kind = 'bar')


# In[23]:


#BsmntHalfBath: Basement Half Bathrooms
df.BsmtHalfBath.value_counts().plot(kind = 'bar')


# In[24]:


#BsmtFullBath: Basement Full Bathrooms
df.BsmtFullBath.value_counts().plot(kind = 'bar')


# In[25]:


#MSSubClass: The building class

df.MSSubClass.value_counts().plot(kind = 'bar')


# In[26]:


#LowQuanlFinSF: Low quality finished square feet (all floors)
df.LowQualFinSF.value_counts().plot(kind = 'bar')


# In[27]:


#Overallcond: Overall condition rating
df.OverallCond.value_counts().plot(kind = 'bar')


# Number of Null values in each feature

# In[28]:


df.isnull().sum()


# In[29]:


null_counts = df.isnull().sum()
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)


# In[30]:


# Fill Missing Values 

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[31]:


df.drop(['Alley'], axis = 1, inplace = True)


# In[32]:


df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[33]:


df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[34]:


df.drop(['GarageYrBlt'], axis = 1, inplace = True)


# In[35]:


df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[36]:


df.drop(['PoolQC','Fence','MiscFeature'], axis = 1, inplace = True)


# In[37]:


df.drop(['Id'], axis = 1, inplace = True)


# In[38]:


df.shape


# In[40]:


df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[41]:


s = df.isnull().sum()


# In[42]:


s.plot.pie(subplots = True, figsize =(18,8))


# In[43]:


df['BsmtFinType2'].isnull().sum()


# In[44]:


df['BsmtExposure'].isnull().sum()


# In[46]:


df['BsmtFinType1'].isnull().sum()


# In[47]:


df['Electrical'].isnull().sum()


# In[48]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])


# Creating a Heatmap 

# In[49]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[50]:


df.dropna(inplace = True)


# In[51]:


df.shape


# In[52]:


df.head()


# In[53]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[54]:


len(columns)


# In[55]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[56]:


main_df = df.copy()


# In[57]:


main_df.shape


# # Working Test dataset

# In[58]:


test_df = pd.read_csv('https://raw.githubusercontent.com/Jordan3296/Kaggle-Competitions/master/House%20Prices%3A%20Advanced%20Regression%20Techniques/test.csv')


# In[59]:


test_df.shape


# In[60]:


test_df.head()


# In[61]:


# Check null values
test_df.isnull().sum()


# Observing Null values with Features

# In[62]:


null_counts = test_df.isnull().sum()
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)


# In[63]:


## Fill Missing Values
test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])
test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])

test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])

test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])


# In[64]:


##Droping variable who is having more number of missing values
test_df.drop(['Alley'],axis=1,inplace=True)
test_df.drop(['GarageYrBlt'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)
test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[65]:


s = test_df.isnull().sum()
s.plot.pie(subplots=True, figsize=(18, 8))


# In[66]:


test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])


# In[67]:


s = test_df.isnull().sum()
s.plot.pie(subplots=True, figsize=(18, 8))


# In[68]:


test_df['BsmtFinType2'].isnull().sum()


# In[69]:


test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])


# In[70]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[71]:


test_df.shape


# In[72]:


df.shape


# In[74]:


final_df=pd.concat([df,test_df])


# In[75]:


final_df.shape


# In[76]:


final_df['SalePrice']


# In[77]:


final_df=category_onehot_multcols(columns)


# In[78]:


final_df.shape


# In[79]:


#Removing all the duplicate columns

final_df = final_df.loc[:,~final_df.columns.duplicated()]


# In[80]:


final_df.shape


# In[82]:


final_df.head()


# In[81]:


df_Train = final_df.iloc[:1460,:]
df_Test = final_df.iloc[1460:,:]


# In[83]:


df_Test.drop(['SalePrice'], axis = 1, inplace = True)


# In[84]:


df_Test.shape


# In[85]:


X_train = df_Train.drop(['SalePrice'], axis = 1)
Y_train = df_Train['SalePrice']


# In[89]:


# Applying the algorithm

import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_train, Y_train)


# In[90]:


# Saving as a "pickle" file so we don't have to save it frequently 

import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[91]:


y_pred = classifier.predict(df_Test) 


# In[92]:


y_pred


# In[117]:


# Create Sample Submission file & Submit

pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('C:/Users/Jordan/Desktop/Data Science/Kaggle Competitions/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred], axis = 1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index = False)


# In[118]:


pred.head(20)


# In[ ]:




