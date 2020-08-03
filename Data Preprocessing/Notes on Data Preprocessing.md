Notes on Data Preprocessing

Hi, Welcome to the practical machine learning series.

The goal of this tutorial is to acquire knowledge of the key concepts required to get started with the preliminary steps when you work on your own data science/machine learning project.

I will be explaining the code in the jupyter notebook in a comprehensible manner so that whenever you need to start a ML/Data Science project you have an basic understanding on how to approach the problem statement.

We will look at the dataset we’re working with to better understand why we are doing all the steps in the notebook.

The dataset contains four columns named Country, Age, Salary, and Purchased with 10 entries each. It shows us the list of purchases made by an individual along with their information. 

You will see that there are nan/null values present in some of the columns which we will take care of in the notebook.

Now we will move onto the notebook and see how we can preprocess our dataset so that we can gain a meaningful insight from it. The tasks that we do for preprocessing are as follows -

Importing the dataset

Splitting columns into features and dependent variables

Taking care of missing values

Encoding categorical values

Labeling dependent variable

Splitting the dataset into training and test set.

Importing the dataset

We start by importing the libraries required for the dataset to be used in the jupyter notebook environment. The libraries we imported are numpy and pandas. To better understand their usage here is a short description of them.

Numpy- This library is used for working with arrays. We use this  library to perform array functions on the dataset when they are  converted into a dataframe. 

Usage: X = np.array(ct.fit_transform(X))

This line of code allows us to use the fit transform function on the array named X.

Pandas – It stands for "Python Data Analysis Library". This provides in-memory 2d table object called dataframe. It is like a spreadsheet with column names and row labels. This allows us to convert our dataset into a dataframe so that we could do further analysis/cleansing of the data and get the fields required to either create a simple model or plot them on a graph. 

Usage: dataset = pd.read_csv('data.csv')

This line of code allows us to convert the data.csv file into a pandas dataframe using the .read_csv function.

We have successfully imported the dataset using the pandas library explained above.

Splitting columns into features and dependent variables

We now extract the dependent and independent variables from the dataset.

IMP: Independent variables are the features of the dataset. The features are the columns you would use to predict the dependent variables.

IMP: Dependent variable is the column you are trying to predict.

NOTE: If you are looking for datasets online to start a project, you will find that the dependent variable is the last column of the dataset if the dataset is acquired from a good source. This makes it easier to split the dataset as we just need to split all the other columns (normally the features) from the last column.

Extracting independent variable (features)-  X = dataset.iloc[:, :-1].values

This line of code splits the indices by taking all the columns except the last column. The ‘-1’ specifies the last column.

Extracting dependent variable,  y = dataset.iloc[:, -1].values

This line of code splits the indices so that we only take the last column of the dataframe.

We have successfully split the columns into features and dependent variable.

Taking care of missing values

Now we will move onto the part on how to take care of missing data so that we can have a better train/test split for our model.

We use a library called scikit learn to use the SimpleImputer function.

Scikit-learn(sklearn) - The sklearn library contains a lot of  efficient tools for machine learning and statistical modeling including  classification and regression. It is used to build machine learning models.

This line of code imports the function from the library- 

Code:  from sklearn.impute import SimpleImputer

We create an instance of the function named 'imputer' and use the np.nan method to specify that we need to work with the null values in the dataframe. The strategy we use to take care of the nan values is by taking mean of the existing values.

Code: imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

The fit method would look at the column we require to apply the transformation. We are looking at the age and salary column of the dataframe. The range does not include upper bound which means when we say 1 : 3, it takes only the first two indices not the third.

Code: imputer.fit(X[:, 1:3]

The transform method applies the transformation we specified in the imputer function.

Code: X[:, 1:3] = imputer.transform(X[:, 1:3]

We have successfully taken care of the missing values.

Encoding categorical values

As we have categorical data for the countries in our dataframe. We have to encode them into numerical values so that the model can understand which value we are referring to rather than looking those values as strings.

In order to encode the independent variable, we require the use of these two functions- ColumnTransformer and OneHotEncoder.

The 2 lines of code imports both of those functions for us to use-

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

Now we create an instance of the ColumnTransformer named 'ct' and we use OneHotEncoder as the encoder for this function. The remainder method takes cares of the remaining dataset which we do not want to adjust. The 'passthrough' value means that the rest of the columns do no get   affected only the mentioned column '0' gets OHE (One Hot Encoding).  

Code: ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')

The fit_transform function applies the transformation on the column specified in the 'ct' variable.

Code: X = np.array(ct.fit_transform(X))

We have successfully encoded the categorical values.

Labeling dependent variable

To encode the dependent variable, we use LabelEncoder function as this column only has yes/no values.

from sklearn.preprocessing import LabelEncoder

We import the function from the library. Then we create an instance  of it named 'le'. Finally, we apply the transformation on the dependent variable column.

le = LabelEncoder()

y = le.fit_transform(y)

We have successfully labelled the dependent variable.

Splitting the dataset into training and test set

Now we are ready to split the dataset into training and test set as we have all the columns converted into numerical figures so that we can easily train a model.

We import the train_test_split function from the sklearn library.

from sklearn.model_selection import train_test_split

We create the testing and training variables for our dataset. For the method we need to specify the training set first then the test set, the test size is the split we want for our model. The 0.2 means 80% of the dataset is associated to the training and the rest 20% to the testing.  The random state variable is used to ensure when you're sharing the  notebook with anyone else they would have the same results as you when they train their model.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

We have successfully split the dataset into training and test set.

We have reached the end of our journey for the data preprocessing notebook. You would most likely require the knowledge of these topics when you are starting off with your own project. Couple of things to note here are, we had a decent looking dataset compared to some of the data that you are going to acquire manually. The columns were already split in a way so that it was easy to differentiate between independent  and dependent variable. Taking all of that into account we still performed all the basics steps you would do as a data scientist while working on a project.
