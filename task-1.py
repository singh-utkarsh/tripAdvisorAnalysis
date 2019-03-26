# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
Dataset = pd.read_csv("lasvegas_tripadvisor.csv",header = 0)

# check if the dataset contains missing values
Dataset.isnull().any().any()

# see the statistics and columns description of dataset
Dataset.describe()
Dataset.info()

#Basic Visualisations 
def plot_by(df, column_name, size=(20, 15), sortit=True, horizontal=True):
    by_user_country = df.copy()
    x = by_user_country.groupby(column_name)["Nr. reviews"].agg(sum)
    kind = "barh" if horizontal else "bar"
    if sortit:
        x.sort_values().plot(kind=kind, figsize=size, title="By " + column_name)
    else:
        x.plot(kind=kind, figsize=size, title="By " + column_name)
plot_by(Dataset, "User continent", size=(10, 5))
plot_by(Dataset, "User country", size=(15, 12))
plot_by(Dataset, "Traveler type", size=(10, 5))
plot_by(Dataset, "Review weekday", size=(10, 5), sortit=False, horizontal=False)
plot_by(Dataset, "Score", size=(10, 5), horizontal=False)


#Convert the columns datatype to categorical which have categorical values
for col in ['User country','Period of stay','Traveler type','Pool',
        'Gym','Tennis court','Spa','Casino','Free internet',
        'Hotel name','Hotel stars','User continent','Review month','Review weekday']:
    Dataset[col]=Dataset[col].astype('category')


#Dependent and independent variables
X = Dataset[Dataset.columns.difference(['Score'])]
y = Dataset.iloc[:,4:5].values

#Removing the error in dataset
X.set_value(75,'Member years',np.median(X[['Member years']]))
#X['Hotel stars'] = X['Hotel stars'].apply({'3,5':5, '4,5':5,3:3,4:4,5:5}.get)
#Try  combinations of '3,5' as 3 and 5 with '4,5' as 4 and 5 respectively

# Encoding categorical data
X = pd.get_dummies(X, columns=X.columns[X.dtypes == 'category'], drop_first=True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_plot = X_test
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Restoring the column names of X_train,X_test
X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = pd.DataFrame(X_test,columns = X.columns)

# Fitting SVR to the dataset
from sklearn import svm
regressor = svm.SVR(kernel = 'rbf',epsilon=0.7)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test) 

# =============================================================================
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# =============================================================================
#mean squared error,mean absolute error,mean absolute percentage error in prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)

def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mape = mean_absolute_percentage_error(y_test,y_pred)


# -----------------------------Feature Importance----------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Create a random forest classifier
rgr = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the classifier
rgr.fit(X_train, y_train)

# Appending the name and gini importance of each feature
feature_imp = []
for feature in zip(X.columns, rgr.feature_importances_):
    feature_imp.append(feature)


# Create a selector object that will use the random forest regressor to identify
# features that have an importance of more than 0.05(5%)
sfm = SelectFromModel(rgr, threshold=0.05)

# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])

Feature_imp_dict = {'Helpful votes':feature_imp[0][1],'Hotel stars':feature_imp[1][1],
                    'Member years':feature_imp[2][1],'Nr. Hotel Reviews':feature_imp[3][1],
                    'Nr. Reviews':feature_imp[4][1],'Nr. rooms':feature_imp[5][1]}
Feature_imp_dict.update((x,y*100) for x,y in Feature_imp_dict.items())

#---------------------------Visualising the results------------------------------
l = [1,2,3,4,5,6]
tick_label = ['Helpful votes','Hotel stars','Member years','Nr. Hotel Reviews',
              'Nr. Reviews','Nr. rooms'] 
h = [Feature_imp_dict['Helpful votes'],Feature_imp_dict['Hotel stars'],Feature_imp_dict['Member years'],Feature_imp_dict['Nr. Hotel Reviews'],
     Feature_imp_dict['Nr. Reviews'],Feature_imp_dict['Nr. rooms']]

plt.figure(figsize = (9,4))
plt.bar(l,h,tick_label=tick_label, align = 'center',
        width=0.5,color=['red','blue','orange','indigo','violet','green'])
plt.xlabel('Features')
plt.ylabel('Percentage of relevance for the model')
plt.title('Most important features in the model')



#Influence of Member years on TripAdvisor score.
z = list(X_plot['Member years'])
count_0,sum_0,count_1,sum_1,count_2,sum_2,count_3,sum_3,count_4,sum_4,count_5,sum_5=0,0,0,0,0,0,0,0,0,0,0,0
for i,data in enumerate(z):
    if(data == 0):
        count_0+=1
        sum_0+=y_pred[i]
    elif(data == 1):
        count_1+=1
        sum_1+=y_pred[i]
    elif(data == 2):
        count_2+=1
        sum_2+=y_pred[i]
    elif(data == 3):
        count_3+=1
        sum_3+=y_pred[i]
    elif(data == 4):
        count_4+=1
        sum_4+=y_pred[i]
    elif(data == 5):
        count_5+=1
        sum_5+=y_pred[i]
Avg_scores = [sum_0/count_0,sum_1/count_1,sum_2/count_2,sum_3/count_3,
              sum_4/count_4,sum_5/count_5]
plt.plot([0,1,2,3,4,5],Avg_scores,marker='s',color='b')
plt.xlabel("Member Years",fontsize=10, fontweight='bold')
plt.ylabel("Trip Advisor Score",fontsize=10, fontweight='bold')
plt.title("Influence of member years on Trip advisor score",fontsize=14, fontweight='bold')