
# coding: utf-8

# In[1]:


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#This feature list was going to be used for selectkbest to give back the highest scoring features 
features_list = ['poi','salary','bonus',
                 'deferral_payments','deferred_income',
                 'director_fees','exercised_stock_options',
                 'expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances','long_term_incentive',
                 'other','restricted_stock','restricted_stock_deferred',
                 'shared_receipt_with_poi','to_messages',
                 'total_payments','total_stock_value']  

    
    
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:


### Task 2: Remove outliers
#plot that shows a possible outlier
import matplotlib.pyplot 
import numpy as np
data = featureFormat(data_dict, features_list)
for point in data:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[4]:


#Exploring for the  possible outlier 

salary = 0
for key, value in data_dict.items():
    salary = data_dict[key]['salary'] 
    if salary >= 5000000:
        if salary != 'NaN':
            print key
            print salary


# In[5]:


#Using the pop method to remove "Total"
data_dict.pop('TOTAL')


# In[6]:


#Data variable that was updated for the removal of the total outlier
data = featureFormat(data_dict, features_list)


# In[7]:


#A plot showing the bonus and salary of employees once the "Total" outlier was removed
for point in data:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[8]:


#Exploring for any possible outlier
bonus = 0
for key, value in data_dict.items():
    bonus = data_dict[key]['bonus'] 
    if bonus >= 4000000:
        if bonus != 'NaN':
            print key
            print bonus


# In[9]:


#Exploring for any possible outlier
salary = 0
for key, value in data_dict.items():
    salary = data_dict[key]['salary'] 
    if salary >= 1000000:
        if salary != 'NaN':
            print key
            print salary


# In[10]:


#A plot that shows the loan advances and total stock value
for point in data:
    total_stock_value = point[-1]
    loan_advances = point[11]
    matplotlib.pyplot.scatter( total_stock_value, loan_advances )

matplotlib.pyplot.xlabel("total_stock_value")
matplotlib.pyplot.ylabel("loan_advances")
matplotlib.pyplot.show()


# In[11]:


#Exploring for any possible outlier
loanADV = 0
for key, value in data_dict.items():
    loanADV = data_dict[key]['loan_advances'] 
    if loanADV >= 5000000:
        if loanADV != 'NaN':
            print key
            print loanADV


# In[12]:


#Exploring for any possible outlier
totalStockValue = 0
for key, value in data_dict.items():
    totalStockValue = data_dict[key]['total_stock_value']    
    if totalStockValue >= 20000000:
        if totalStockValue != 'NaN':
            print key
            print totalStockValue
            
        


# In[13]:


#Feature Creation 
#Creating a feature that sums up the value of salary and bonus. 

for k,v in data_dict.items():
    if v['bonus'] != 'NaN' and  v['salary'] != 'NaN':
      
        v['Salary_plus_bonus'] = v['bonus'] + v['salary']
        print k
        print v['Salary_plus_bonus']
        
    elif v['bonus']== 'NaN':
        v['Salary_plus_bonus'] = v['salary']
    elif v['salary'] == 'NaN':
        v['Salary_plus_bonus'] = v['bonus']
    else:
        v['Salary_plus_bonus'] == 0
        v['Salary_plus_bonus'] = v['bonus'] + v['salary']
        print k
        print v['Salary_plus_bonus']
       


# In[14]:


#This feature list was used for selectkbest when salary plus bonus was created.
features_list = ['poi','salary','bonus','Salary_plus_bonus',
                 'deferral_payments','deferred_income',
                 'director_fees','exercised_stock_options',
                 'expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances','long_term_incentive',
                 'other','restricted_stock','restricted_stock_deferred',
                 'shared_receipt_with_poi','to_messages',
                 'total_payments','total_stock_value']    


# In[15]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
from sklearn.cross_validation import train_test_split
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[16]:


#using selectkbest to find best features
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




new = SelectKBest(k=2).fit(features, labels)
print new.scores_


# In[17]:


#Using select k best scores I updated the features list to include the ones that had higher values. 
#My feature creation that I created got a score of 22.888. 
#Removed anything scoring below a 10
#removed deferral payments, director fees, expenses,from messages,'from_poi_to_this_person'
#from_this_person_to_poi'loan_advances''other','restricted_stock',
#'restricted_stock_deferred','shared_receipt_with_poi','to_messages','total_payments',


# In[18]:


#Features that had a score of more than 10
#List of Seven Features 
features_list = ['poi','salary','bonus',
                 'Salary_plus_bonus',
                 'deferred_income',
                 'exercised_stock_options',
                 'long_term_incentive',
                 'total_stock_value']    


# In[19]:


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[20]:


new = SelectKBest(k=2).fit(features, labels)
print new.scores_


# In[ ]:





# In[21]:


#Using the Gaussian NB Classifier with Seven Features
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)


# In[22]:


#Using the KNeighbors Classifier with Seven Features
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
test_classifier(clf, my_dataset, features_list)


# In[23]:


#Using the Decision Tree Classifier with Seven Features
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)


# In[24]:


#List of Six Features 
features_list = ['poi','salary','bonus',
                 'Salary_plus_bonus',
                 'deferred_income',
                 'exercised_stock_options',
                 'total_stock_value']    


# In[25]:


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[26]:


new = SelectKBest(k=2).fit(features, labels)
print new.scores_


# In[27]:


#Using the Gaussian NB Classifier with Six Features 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)


# In[28]:


#Using the KNeighbors Classifier with Six Features 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
test_classifier(clf, my_dataset, features_list)


# In[29]:


#Using the Decision Tree Classifier with Six Features 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)


# In[30]:


#attempting to do selectkbest again..features that had score above 12 before were chosen this time

#List of Five Features 
features_list = ['poi','salary','bonus',
                 'Salary_plus_bonus',
                 'exercised_stock_options',
                 'total_stock_value'] 


# In[31]:


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[32]:


new = SelectKBest(k=2).fit(features, labels)
print new.scores_


# In[33]:


### Task 4: Try a varity of classifiers


#Using the Gaussian NB Classifier with Five Features

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)


# In[34]:


#Using the KNeighbors Classifier with Five Features
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
test_classifier(clf, my_dataset, features_list)


# In[35]:


#Using the Decision Tree Classifier with Five Features 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)


# In[ ]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#test_classifier(clf, my_dataset, features_list)

#train_test_split(features, labels, test_size=0.3, random_state=42)
#test_classifier(clf, my_dataset, features_list)


# In[36]:


#KNeighbors GridSearch  Tune 1 with Five Features 
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
parameters = {'algorithm': ('auto','ball_tree','kd_tree','brute')}

KN = KNeighborsClassifier()
clf = GridSearchCV(KN, parameters)
cv = cross_validation.StratifiedShuffleSplit(labels, 100, random_state = 42)
a_grid_search = GridSearchCV(clf, parameters, cv = cv, scoring = 'recall')

test_classifier(clf, my_dataset, features_list)


# In[37]:


#KNeighbors GridSearch Tune 2 with Five Features
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
parameters = {'n_neighbors':[3,4,5,6,8,10,12,13,14]}

KN = KNeighborsClassifier()
clf = GridSearchCV(KN, parameters)
cv = cross_validation.StratifiedShuffleSplit(labels, 100, random_state = 42)
a_grid_search = GridSearchCV(clf, parameters, cv = cv, scoring = 'recall')

test_classifier(clf, my_dataset, features_list)


# In[38]:


#KNeighbors GridSearch Tune 3 with Five Features
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
parameters = {'weights': ('uniform', 'distance')}

KN = KNeighborsClassifier()
clf = GridSearchCV(KN, parameters)
cv = cross_validation.StratifiedShuffleSplit(labels, 100, random_state = 42)
a_grid_search = GridSearchCV(clf, parameters, cv = cv, scoring = 'recall')

test_classifier(clf, my_dataset, features_list)


# In[39]:


#Top Scoring Classifier that passes the .3 for precision and recall

#Using the Decision Tree Classifier with Five Features 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)










# In[ ]:



#stratified best for our dataset because of the imbalance of poi and sample


# In[40]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




