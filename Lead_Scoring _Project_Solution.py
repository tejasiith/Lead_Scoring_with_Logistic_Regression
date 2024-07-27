#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Lead Scoring Project
# 
# ### Problem Statement -
# 
# X Education wants to improve its lead-to-sale conversion rate by identifying potential leads ('Hot Leads') with a higher likelihood of conversion. The goal is to build a lead scoring model using a dataset of approximately 9000 past leads. The model should assign a lead score to each lead, where higher scores indicate a higher probability of conversion. The target lead conversion rate is 80%. The dataset contains various attributes such as Lead Source, Total Time Spent on the Website, Total Visits, Last Activity, etc., and the target variable is 'Converted' (1 for converted, 0 for not converted). Categorical variables may contain a level called 'Select', which needs to be handled as a null value.

# In[1]:


# Surpress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Step 1:- Read the Dataset

# In[122]:


leads= pd.read_csv(r'C:\Users\DELL\Documents\PROJECTS\Capstone Project\Lead Scoring Assignment\Leads.csv')
leads


# In[123]:


leads.head()


# In[124]:


leads.shape


# In[125]:


leads.describe()


# In[126]:


leads.info()


# In[127]:


leads.columns


# ### Step 2: Data Cleaning and Preparation

# In[128]:


leads.isnull().sum()


# As you can see there are a lot of columns which have high number of missing values. Clearly, these columns are not useful. Since, there are 9000 datapoints in our dataframe, let's eliminate the columns having greater than 3000 missing values as they are of no use to us.

# In[129]:


# Drop all the columns in which greater than 3000 missing values are present
for col in leads.columns:
    if leads[col].isnull().sum()>3000:
        leads.drop(col, 1, inplace= True)


# In[130]:


# Check the number of null values again
leads.isnull().sum()


# As you might be able to interpret, the variable "City" won't be of any use of in our analysis. So it's best that we drop it.

# In[131]:


leads.drop(['City'], axis=1, inplace=True)


# In[132]:


# Same goes for variable "Country"
leads.drop(['Country'], axis=1, inplace=True)


# In[133]:


# Let's now check the percentage of missing values in each column
round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[134]:


# Check the number of null values again
leads.isnull().sum()


# In[135]:


# Get the value counts of all the columns
for column in leads:
    print(leads[column].astype('category').value_counts())
    print('____________________________________________')


# The following three columns now have the level 'Select'. Let's check them once again.

# In[136]:


leads['Lead Profile'].astype('category').value_counts()


# In[137]:


leads['How did you hear about X Education'].value_counts()


# In[138]:


leads['Specialization'].value_counts()


# Clearly the levels `Lead Profile` and `How did you hear about X Education` have a lot of rows which have the value `Select` which is of no use to the analysis so it's best that we drop them.

# In[139]:


leads.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# Also notice that when you got the value counts of all the columns, there were a few columns in which only one value was majorly present for all the data points. These include `Do Not Call`, `Search`, `Magazine`, `Newspaper Article`, `X Education Forums`, `Newspaper`, `Digital Advertisement`, `Through Recommendations`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque`. Since practically all of the values for these variables are `No`, it's best that we drop these columns as they won't help with our analysis.

# In[140]:


leads.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis=1, inplace=True)


# In[141]:


leads.drop(['What matters most to you in choosing a course'], axis=1, inplace=True)


# In[142]:


# Check the number of null values again

leads.isnull().sum()


# Now, there's the column `What is your current occupation` which has a lot of null values. Now you can drop the entire row but since we have already lost so many feature variables, we choose not to drop it as it might turn out to be significant in the analysis. So let's just drop the null rows for the column `What is you current occupation`.

# In[143]:


leads = leads[~pd.isnull(leads['What is your current occupation'])]


# In[144]:


leads.isnull().sum()


# In[145]:


# Drop the null value rows in the column 'TotalVisits'

leads = leads[~pd.isnull(leads['TotalVisits'])]


# In[146]:


leads.isnull().sum()


# In[147]:


# Drop the null values rows in the column 'Lead Source'

leads = leads[~pd.isnull(leads['Lead Source'])]


# In[148]:


leads.isnull().sum()


# In[149]:


# Drop the null values rows in the column 'Specialization'

leads = leads[~pd.isnull(leads['Specialization'])]


# In[150]:


leads.isnull().sum()


# Now our data doesn't have any null values. Let's now check the percentage of rows that we have retained.

# In[151]:


print(len(leads.index))
print(len(leads.index)/9240)


# We still have around 69% of the rows which seems good enough.

# In[152]:


# Let's look at the dataset again

leads.head()


# Now, clearly the variables `Prospect ID` and `Lead Number` won't be of any use in the analysis, so it's best that we drop these two variables.

# In[153]:


leads.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)


# In[154]:


leads.head()


# ### Dummy variable creation
# 
# The next step is to deal with the categorical variables present in the dataset. So first take a look at which variables are actually categorical variables.

# In[155]:


# Check the columns which are of type 'object'

temp = leads.loc[:, leads.dtypes == 'object']
temp.columns


# In[156]:


# Create dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
leads = pd.concat([leads, dummy], axis=1)


# In[157]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(leads['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
leads = pd.concat([leads, dummy_spl], axis = 1)


# In[158]:


# Drop the variables for which the dummy variables have been created

leads = leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[159]:


# Let's take a look at the dataset again

leads.head()


# ### Test-Train Split
# 
# The next step is to split the dataset into training an testing sets.

# In[160]:


from sklearn.model_selection import train_test_split


# In[161]:


# Put all the feature variables in X

X = leads.drop(['Converted'], 1)
X.head()


# In[162]:


# Put the target variable in y

y = leads['Converted']

y.head()


# In[163]:


# Split the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test= train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=100)


# ### Scaling
# 
# Now there are a few numeric variables present in the dataset which have different scales. So let's go ahead and scale these variables.

# In[164]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler


# In[165]:


# Scale the three numeric features prsent in the dataset
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# ### Looking at the correlations
# 
# Let's now look at the correlations. Since the number of variables are pretty high, it's better that we look at the table instead of plotting a heatmap

# In[166]:


# Looking at the correlation table

leads.corr()


# ## Step 2: Model Building
# 
# Let's now move to model building. As you can see that there are a lot of variables present in the dataset which we cannot deal with. So the best way to approach this is to select a small set of features from this pool of variables using RFE.

# In[167]:


# Import 'LogisticRegression' and create a LogisticRegression object

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[168]:


# Import RFE and select 15 variables

from sklearn.feature_selection import RFE
rfe = RFE(estimator=logreg, n_features_to_select=15) 
rfe.fit(X_train, y_train)


# In[169]:


# Let's take a look at which features have been selected by RFE
list(zip(X_train.columns,rfe.support_, rfe.ranking_))


# In[170]:


# Put all the columns selected by rfe in the variables "col"
col = X_train.columns[rfe.support_]


# Now we have all the variables selected by rfe and since we care about statistics part i.e. the p-values and VIFs, let's use these variables to create logistic regression model using statsmodels.

# In[171]:


# Select only the columns selected by RFE
X_train = X_train[col]


# In[172]:


# Import statsmodels
import statsmodels.api as sm


# In[173]:


# Fit a logistic regression model on X_train after adding a constat and output the summary
X_train_sm = sm.add_constant(X_train)
logm2= sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res=logm2.fit()
res.summary()


# There are quite a variables which have a p-value greater than `0.05`. we will need to take care of them. But first let's also look at the VIFs.

# In[174]:


# Import variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[175]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features']=X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# VIFs seem to be in a decent range except for three variables. 
# 
# Let's first drop the variable `Lead Source_Reference` since it has a high p-value as well as a high VIF.

# In[176]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# In[177]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[178]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# The VIFs are now all less than 5. So let's drop the ones with the high p-values beginning with `Last Notable Activity_Had a Phone Conversation`.

# In[179]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[180]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Drop `What is your current occupation_Housewife`.

# In[181]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[182]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# Drop `What is your current occupation_Working Professional`.

# In[183]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# In[184]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# All the p-values are now in the appropriate range. Let's also check the VIFs again in case we had missed something.

# In[185]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We are good to go!

# ## Step 3: Model Evaluation
# 
# Now, both the p-values and VIFs seem decent enough for all the variables. So let's go ahead and make predictions using this final set of features.

# In[186]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred[:10]


# In[187]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating a dataframe with the actual conversion flag and the predicted probabilities

# In[188]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# #### Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0

# In[189]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[190]:


# Import metrics from sklearn for evaluation

from sklearn import metrics


# In[191]:


# Create confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[192]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[193]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[194]:


# Calculate the sensitivity

TP/(TP+FN)


# In[195]:


# Calculate the specificity

TN/(TN+FP)


# In[196]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[197]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[198]:


# Import matplotlib to plot the ROC curve

import matplotlib.pyplot as plt


# In[199]:


# Call the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# The area under the curve of the ROC is 0.86 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[200]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[201]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[202]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As you can see that around 0.42, you get the optimal values of the three metrics. So let's choose 0.42 as our cutoff now.

# In[203]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[204]:


# Let's check the accuracy now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[205]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[206]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[207]:


# Calculate Sensitivity

TP/(TP+FN)


# In[208]:


# Calculate Specificity

TN/(TN+FP)


# This cutoff point seems good to go!

# ## Step 4: Making Predictions on the Test Set
# 
# Let's now make predicitons on the test set.

# In[209]:


# Scale the test set as well using just 'transform'

X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[210]:


# Select the columns in X_train for X_test as well

X_test = X_test[col]
X_test.head()


# In[211]:


# Add a constant to X_test

X_test_sm = sm.add_constant(X_test[col])


# In[212]:


# Check X_test_sm

X_test_sm


# In[213]:


# Drop the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 'Last Notable Activity_Had a Phone Conversation'], 1, inplace = True)


# In[214]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(sm.add_constant(X_test))


# In[215]:


y_test_pred[:10]


# In[216]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[217]:


# Let's see the head

y_pred_1.head()


# In[218]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[219]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[220]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[221]:


# Check 'y_pred_final'

y_pred_final.head()


# In[222]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[223]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[224]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[225]:


# Check y_pred_final

y_pred_final.head()


# In[226]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[227]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[228]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[229]:


# Calculate sensitivity
TP / float(TP+FN)


# In[230]:


# Calculate specificity
TN / float(TN+FP)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




