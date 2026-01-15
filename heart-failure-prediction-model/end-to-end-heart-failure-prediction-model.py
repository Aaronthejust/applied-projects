#######################################################################

#########---------PROJECT DETAILS-------------#################

'''End to end data science project Heart failure prediction
In this project, I will show you how to do an end to end data 
science project starting from importing the required libraries 
and packages to the prediction using machine learning algorithms
and visualizing the different accuracies

The project is about heart failure and it is binary classification, 
which means we either have 1 for death or 0 for survival.

In the first part, we will analyze our dataset that we import 
from kaggle with the different exploratory analysis techniques 
like data visualization and dataframes with pandas, matplotlib 
and seaborn for both categorical and continuous variables.

Then we split our dataset for machine learning prediction using 
different classification algorithms.

Finally, we visualize the different accuracies using a bar 
chart and compare between the different algorithms. '''

#########################################################################

# importing basic libraries for data-analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing training testing splitting model and standardscalar for data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing supervised machine learning models i.e classification models
from sklearn.linear_model import LogisticRegression #as we will have binary classification
from sklearn.svm import SVC #support vector machine algorithm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#importing acc score metric from sklearn metrics to analyse our model performance
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset/classification_datasets/heart_failure_clinical_records_dataset.csv')
print(df.dtypes)    #shows datatype of each column
print(df.shape)     #we have 299 rows and 13 columns
print(df.columns)   #shows list of columns names of dataset used 
print(df.head(10))  #by default head shows first 5 records here i passed 10 so it will show first 10 records

#now lets split our dataset into two categories to make it easier for analysis

#as we have some categorical columns and some continous

categorical_columns = df[["anaemia","diabetes","high_blood_pressure", "sex", "smoking"]] #our categorical data here all column have binary category either 0 or 1
#Castegorical column explaination
''' 0 mean no anaemia 1 mean yes patient has anaemia
    0 mean no diabetes 1 mean yes patient has diabetes
    0 mean no high blood pressure 1 mean yes patient has high blood pressure
    0 for female 1 for male
    0 mean no smoking 1 mean yes patient do smoking'''
#we have death_event also as categorical columns but that is our label/target column not feature one

print(type(categorical_columns)) #type of categorical_columns is dataframe

continous_columns = df[["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine","serum_sodium", "time"]] 

#lets check if we have any null values in our dataset
print(df.isna().sum()) #no null values. this command shows count of null values here 0 for all columns
print(pd.set_option('display.max_rows', 300,),df) #method to print all records without skipping anything

print(df.isna()) #shows false as no null. no missing data in our dataset

#same function is perform by
print(df.isnull())
print(df.isnull().sum())

'''describe function tells about the descriptive statistics
   it tells about min value max value mean standard deviation
   that helps us to understand the variability in data
   '''
print(continous_columns.describe())
print(categorical_columns.describe())
print(df.groupby("DEATH_EVENT").count().T) #gives us detail about death event column with all other feature column name
#as it gives that deaths are 1 which mean yes which mean death are 96
#while 0 i.e  no are 203 so it means our dataset is unbalanced

age = df[["age"]] #here inside two brackets its type is datafram
#if we use one bracket it will give us series
platelets = df[["platelets"]] 

#lets visualize  the unbalanced data it to see the difference 
# through scatter plot graph the color will depend on death event(0,1)
show_graphs = False
if show_graphs:
    plt.figure(figsize=(13,7))
    plt.scatter(platelets,age, c = df["DEATH_EVENT"], s = 100, alpha= 0.8)
    # Scatter plot of Platelets vs Age
    # platelets = x-axis, age = y-axis
    # c = df["DEATH_EVENT"] colors the points based on death outcome (0 = alive, 1 = died) single bracket mean we take it as series
    # s = 100 sets the dot size how big is dot size or how small
    # alpha = 0.8 makes the dots slightly transparent the greater alpha the darker the color of dots
    plt.xlabel("Platelets", fontsize = 20)
    plt.ylabel("Age", fontsize = 20)
    plt.title("Visualizing the  Unbalanced Data", fontsize = 22)
    plt.show() #all the zeros are in purple and all 1 in yellow

    #graph for visualizing relationship between all variable and DEATH_EVENT

    plt.figure(figsize=(13,7))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
    plt.xticks(rotation=45, ha = "right", fontsize = 5)   # Rotate x-axis labels 45 degrees
    #plt.yticks(rotation=30)   # Rotate y-axis labels 45 degrees
    plt.title("Relationship between all Variable of dataframe and DEATH_EVENT", fontsize = 22)
    plt.show()

print(df.corr())
#if 1 it means correlated if close to zero or zero mean not correlated
#if closer to -1 that means the negatively correlated mean by increasing one other decreases
#so DEATH_EVENT has positive correlation with age and serum creatinine
#and DEATH_EVENT has negative correlation with time, ejection_fraction and serum sodium

#####-------DATA Visualization---------########

#lets create two lists of categorical and continous data
categorical_data = ["anaemia", "diabetes", "high_blood_pressure","sex", "smoking"]
continous_data = ["age", "creatinine_phosphokinase", "ejection_fraction","platelets","serum_creatinine", "serum_sodium", "time"]

plt.figure(figsize=(13,10))
'''The enumerate() function in Python is a built-in function 
used to iterate over an iterable (like a list, tuple, or string)
 while simultaneously keeping track of the index of each element.
   It returns an enumerate object, which produces pairs of 
  (index, value) tuples.''' 
for i, cat in enumerate(categorical_data):
    plt.subplot(2,3,i+1) #create a matrix of plot which has 2 rows and three columns where i+1 is adding graphs to it
    sns.countplot(data=df, x = cat, hue = "DEATH_EVENT")
plt.show()
print("\nNames of Categorical Data list at each index:")
for i, cat in enumerate(categorical_data):
    print(i, cat)
#so except for category Sex all other shows greater at 0 and less at 1
#while in sex for male i.e 1 it is greater death event that female

#now plotting the impact of continous variable with death_event
plt.figure(figsize=(17,15))
for j, con in enumerate(continous_data):
    plt.subplot(3,3, j+1)
    sns.histplot(data = df, x = con, hue="DEATH_EVENT", multiple="stack")
#our subplots are overlapping so to adjust we use following
plt.subplots_adjust(
    hspace=0.5, # Increase vertical space
    wspace=0.4, # Increase horizontal space
    top=0.9,    # Adjust top margin
    bottom=0.1, # Adjust bottom margin
    left=0.1,   # Adjust left margin
    right=0.9   # Adjust right margin
)
plt.show()
print("\nNames of Continous Data list at each index:")
for j, con in enumerate(continous_data):
    print(j,con)

# for continous we use histogram for categorical we use countplots

#now using boxplot to get minimum max mean quartile etc

plt.figure(figsize=(8,8))
sns.boxplot(data=df, x = "sex", y= "age", hue= "DEATH_EVENT") #sex categorical data while age continous data on y axis
plt.title("The impact of Sex and Age on DEATH_EVENT", fontsize = 22)
plt.show()
#key aspects from boxplot
# so for males when deathevent equals to 1 the median is 65 while 
# the median is approx 60 when deathevent equal to 0 
# for females both for survied and dead having median value equal to 60 

''' Analysing the survival statistics on smoking'''

smokers = df[df["smoking"] == 1]
non_smokers = df[df["smoking"] == 0]

non_survived_smokers = smokers[smokers["DEATH_EVENT"] == 1]
survived_non_smokers = non_smokers[non_smokers["DEATH_EVENT"] == 0]

non_survived_non_smokers = non_smokers[non_smokers["DEATH_EVENT"] == 1]
survived_smokers = smokers[smokers["DEATH_EVENT"] == 0]

smoking_data = [len(non_survived_smokers), len(survived_non_smokers), len(non_survived_non_smokers), len(survived_smokers)]
smoking_labels = ["non_survived_smokers", "survived_non_smokers", "non_survived_non_smokers", "survived_smokers"]

plt.figure(figsize=(9,9))
plt.pie(smoking_data, labels=smoking_labels, autopct='%.1f%%', startangle=90)  #autopct Shows percentage on the slices (one decimal place)
circle = plt.Circle((0,0), 0.7, color = "white") #Creates a white circle centered at (0,0) with radius 0.7 To convert the pie chart into a donut chart style.
p = plt.gcf() #gcf() means Get Current Figure So we can access and modify the chart after drawing it.
p.gca().add_artist(circle) #gca() means Get Current Axes Actually places the white circle inside to make the donut shape.
plt.title("Survival status on Smoking", fontsize = 22)
plt.show()


print(type(non_smokers))
print(smokers[smokers["DEATH_EVENT"] == 1])
print((len(non_survived_smokers) / 299) *100)
print(len(smokers[smokers["DEATH_EVENT"] == 1]))
print(smoking_data)
print(smoking_labels)

#Now lets try same survival status for the sex

male = df[df["sex"] == 1]
female = df[df["sex"] == 0]

non_survived_male = male[male["DEATH_EVENT"] == 1]
survived_male = male[male["DEATH_EVENT"] == 0]

non_survived_female = female[female["DEATH_EVENT"] == 1]
survived_female = female[female["DEATH_EVENT"] == 0]

sex_data = [len(non_survived_male), len(survived_male), len(non_survived_female), len(survived_female)]
sex_labels = ["non_survived_male", "survived_male", "non_survived_female", "survived_female"]

plt.figure(figsize=(10,10))
plt.pie(sex_data, labels=sex_labels, autopct='%.1f%%', startangle=90)
circle = plt.Circle((0,0), 0.5, color = "white")
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival status on Sex", fontsize = 22)
plt.show()

#Now analysing survial status on diabetes


with_diabetes = df[df["diabetes"] == 1]
without_diabetes = df[df["diabetes"] == 0]

non_survived_with_diabetes = with_diabetes[with_diabetes["DEATH_EVENT"] == 1]
survived_with_diabetes = with_diabetes[with_diabetes["DEATH_EVENT"] == 0]

non_survived_without_diabetes = without_diabetes[without_diabetes["DEATH_EVENT"] == 1]
survived_without_diabetes = without_diabetes[without_diabetes["DEATH_EVENT"] == 0]

diabtes_data = [len(non_survived_with_diabetes), len(survived_with_diabetes), len(non_survived_without_diabetes), len(survived_without_diabetes)]
diabetes_labels = ["non_survived_with_diabetes", "survived_with_diabetes", "non_survived_without_diabetes", "survived_without_diabetes"]

plt.figure(figsize=(10,10))
plt.pie(diabtes_data, labels=diabetes_labels, autopct='%.1f%%', startangle=90)
circle = plt.Circle((0,0), 0.3, color = "white")
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival status on Diabetes", fontsize = 22)
plt.show()


#Now analysing survial status on anaemia 

with_anaemia = df[df["anaemia"] == 1]
without_anaemia = df[df["anaemia"] == 0]

non_survived_with_anaemia  = with_anaemia[with_anaemia["DEATH_EVENT"] == 1]
survived_with_anaemia = with_anaemia[with_anaemia["DEATH_EVENT"] == 0]

non_survived_without_anaemia = without_anaemia[without_anaemia["DEATH_EVENT"] == 1]
survived_without_anaemia = without_anaemia[without_anaemia["DEATH_EVENT"] == 0]

anaemia_data = [len(non_survived_with_anaemia), len(survived_with_anaemia), len(non_survived_without_anaemia), len(survived_without_anaemia)]
anaemia_label = ["non_survived_with_anaemia", "survived_with_anaemia", "non_survived_without_anaemia", "survived_without_anaemia"]

plt.figure(figsize=(8,8))
plt.pie(anaemia_data, labels= anaemia_label, autopct='%.1f%%', startangle=90)
circle = plt.Circle((0,0), 0.4, color = "white")
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival rate on Anaemia", fontsize = 22)
plt.show()

#Survival status on high_blood_pressure

with_high_blood_pressure = df[df["high_blood_pressure"] == 1]
without_high_blood_pressure = df[df["high_blood_pressure"] == 0]

non_survived_with_high_blood_pressure  = with_high_blood_pressure[with_high_blood_pressure["DEATH_EVENT"] == 1]
survived_with_high_blood_pressure = with_high_blood_pressure[with_high_blood_pressure["DEATH_EVENT"] == 0]

non_survived_without_high_blood_pressure = without_high_blood_pressure[without_high_blood_pressure["DEATH_EVENT"] == 1]
survived_without_high_blood_pressure = without_high_blood_pressure[without_high_blood_pressure["DEATH_EVENT"] == 0]

high_BP_data = [len(non_survived_with_high_blood_pressure), len(survived_with_high_blood_pressure), len(non_survived_without_high_blood_pressure), len(survived_without_high_blood_pressure)]
high_BP_label = ["non_survived_with_high_BP", "survived_with_high_BP", "non_survived_without_high_BP", "survived_without_high_BP"]

plt.figure(figsize=(8,8))
plt.pie(high_BP_data, labels= high_BP_label, autopct='%.1f%%', startangle=90)
circle = plt.Circle((0,0), 0.4, color = "white")
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Survival rate on High Blood Pressure", fontsize = 22)
plt.show()

''' Data Modeling and Prediction using continous data
    Here we are going for training our Machine Learning
    Algorithm Training and then predicting values on the basis
    of training to see how well our model trained and what accuracy
    comes out of different model training '''

x = df[["age", "creatinine_phosphokinase", "ejection_fraction","platelets","serum_creatinine", "serum_sodium", "time"]]
y = df["DEATH_EVENT"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#data scaling

scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

print(x_train_scaled)
print(x_test_scaled)

#This is a list to save value of accuracy score of all the models used for comparison using a barplot
accuracy_list = []

#Model 1 Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train_scaled, y_train)
lr_prediction = lr_model.predict(x_test_scaled)
lr_accuracy = (round(accuracy_score(lr_prediction, y_test), 4) * 100) #here rounding lr accuracy result upto 4 decimal places and then converting into percentage
accuracy_list.append(lr_accuracy)

#Model 2 Support Vector Machine SVM
svc_model = SVC()
svc_model.fit(x_train_scaled, y_train)
svc_prediction = svc_model.predict(x_test_scaled)
svc_accuracy = (round(accuracy_score(svc_prediction, y_test), 4) * 100)
accuracy_list.append(svc_accuracy)

#Model 3 KNearestNeighbour KNN
#using this model to find the optimal value of K
#We test multiple K values to find the most accurate, balanced 
# and optimal K before training the final KNN model.
knn_list = []
for k in range(1, 50):
    knn_model = KNeighborsClassifier(n_neighbors= k)
    knn_model.fit(x_train_scaled, y_train)
    knn_prediction = knn_model.predict(x_test_scaled)
    knn_accuracy = (round(accuracy_score(knn_prediction, y_test), 4) * 100)
    knn_list.append(knn_accuracy)
k = np.arange(1, 50)
plt.plot(k, knn_list)
plt.xlabel("K value (Number of Neighbors)")
plt.ylabel("Accuracy (%)")
plt.title("Finding the Best K for KNN")
plt.grid(True)
plt.show()

''' The graph helps us identify the best K value for the KNN 
    model. Accuracy increases at first, becomes stable and 
    highest near K ≈ 8 to 12, then decreases for larger K values. 
    So we pick the K that gives the best accuracy.'''

knn_model = KNeighborsClassifier(n_neighbors= 8)
knn_model.fit(x_train_scaled, y_train)
knn_prediction = knn_model.predict(x_test_scaled)
knn_accuracy = (round(accuracy_score(knn_prediction, y_test), 4) * 100)
accuracy_list.append(knn_accuracy)

#Model 4 Decision Tree Classifier

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth= 2)
dt_model.fit(x_train_scaled, y_train)
dt_prediction = dt_model.predict(x_test_scaled)
dt_accuracy = (round(accuracy_score(dt_prediction, y_test), 4) * 100)
accuracy_list.append(dt_accuracy)

#Model 5 Naive Bayes

nb_model = GaussianNB()
nb_model.fit(x_train_scaled, y_train)
nb_prediction = nb_model.predict(x_test_scaled)
nb_accuracy = (round(accuracy_score(nb_prediction, y_test), 4) * 100)
accuracy_list.append(nb_accuracy)

#Model 6 Random Forest Classifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train_scaled, y_train)
y_pred_rf_model = rf_model.predict(x_test_scaled)
rf_accuracy = (round(accuracy_score(y_pred_rf_model, y_test), 4) * 100)
accuracy_list.append(rf_accuracy)

print("\nAccuracy of different models used:")
model_names = ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes", "Random Forest"]
for i in range(len(model_names)):
    print(f"{model_names[i]}: {accuracy_list[i]:.2f}%")
#Visualizing the accuracy of different models using a bar plot
plt.figure(figsize=(10,6))  
sns.barplot(x=model_names, y=accuracy_list, palette="viridis")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("Model Accuracy Comparison", fontsize=16)
plt.xticks(rotation=45)
for i, v in enumerate(accuracy_list):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)
plt.show()
from sklearn.metrics import classification_report
print("Random Forest classifier:\n", classification_report(y_test, y_pred_rf_model))
report = classification_report(y_test, y_pred_rf_model, output_dict=True)
report_df = pd.DataFrame(report).transpose()
# Plot classification report heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Classification Report - Random Forest Classifier")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()








