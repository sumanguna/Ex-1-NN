<H3>ENTER YOUR NAME : Vishal.s</H3>
<H3>ENTER YOUR REGISTER NO  : 212223240184 </H3>
<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```


## OUTPUT:
### Dataset:

![image](https://github.com/user-attachments/assets/2da8e209-4780-49e0-8f6f-e5052eabee39)

### X Values:

![image](https://github.com/user-attachments/assets/f4b1a0d2-5f49-48e0-bc17-4a1e791afd0a)

### Y Values:

![image](https://github.com/user-attachments/assets/6510c509-90dc-471c-87ab-447b09b2eeb3)

### Null Values:

![image](https://github.com/user-attachments/assets/d0f951ef-ecd2-4ba2-bfcf-822d5584e989)

### Duplicated Values:
![image](https://github.com/user-attachments/assets/b4c11e27-a28c-4e53-ab48-4d5fc4a641ec)

### Description:

![image](https://github.com/user-attachments/assets/be9b4326-36bc-4fc0-a8b1-f3d94156019b)

### Normalized Dataset:
![image](https://github.com/user-attachments/assets/7d6aef80-b1ae-4051-aeec-955a81afab74)






### Testing Data:
![image](https://github.com/user-attachments/assets/c5873f5c-ceb9-48b2-878b-c88de9325785)





## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



