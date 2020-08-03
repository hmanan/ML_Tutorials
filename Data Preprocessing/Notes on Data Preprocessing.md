Notes on Data Preprocessing

I will be explaining the code in the jupyter notebook in a easy to understand manner so that whenever you need to start a ML/Data Science project you have an basic understanding on how to approach the problem statement.

We start by importing the libraries required for the dataset to be imported in the jupyter notebook environment. The libraries we imported are numpy and pandas. We will learn why we require those libraries and their usage for our notebook.

1. Numpy- This library is used for working with arrays. We use this library to perform array functions on the dataset when they are converted into a dataframe. Usage:

X = np.array(ct.fit\_transform(X))

This line of code allows us to use the fit transform function on the array named X.

2. Pandas – It stands for &quot;Python Data Analysis Library&quot;. This provides in-memory 2d table object called Dataframe. It is like a spreadsheet with column names and row labels. This allows us to convert our dataset into a dataframe so that we could do further analysis/cleansing of the data and get the fields required to either create a simple model or plot them on a graph. Usage:

dataset = pd.read\_csv(&#39;data.csv&#39;)

This line of code allows us to convert the data.csv file into a pandas dataframe using the .read\_csv function.

Then we imported the dataset using the pandas library explained above.

We then extracted the dependent and independent variables from the dataset.

Independent variables are the features of the dataset. The features are the columns you would use to predict the dependent variables.

Dependent variable is the column you are trying to predict.

NOTE: If you are looking for datasets online to start a project, you will find that the dependent variable is the last column of the dataset if the dataset is acquired from a good source. This makes it easier to split the dataset as we just need to split all the other columns (normally the features) from the last column.

# Independent variable (features)

X = dataset.iloc[:, :-1].values

This line of code splits the indices by taking all the columns except the last column. The -1 means the last column.

# Dependent variable

y = dataset.iloc[:, -1].values

Now we are splitting the indices so that we only take the last column of the dataframe.

Now we will move onto the part on how to take care of missing data as it is required before we begin splitting the dataset into training and testing so that we could get a better model in the end.

We use a library called scikit learn to use the SimpleImputer function.

Scikit-learn(sklearn) - The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification and regression. It is used to build machine learning models.

from sklearn.impute import SimpleImputer

This line of code imports the function from the library.

imputer = SimpleImputer(missing\_values=np.nan, strategy=&#39;mean&#39;)

We create an instance of the function named &#39;imputer&#39; and use the np.nan method to specify that we need to work with the null values in the dataframe. The strategy we use to take care of the nan values is by taking mean of the existing values.

imputer.fit(X[:, 1:3]

The fit method would look at the column we require to apply the transformation. We are looking at the age and salary column of the dataframe. The range does not include upper bound which means when we say 1:3 it takes only the first two indices not the third.

X[:, 1:3] = imputer.transform(X[:, 1:3]

The transform method applies the transformation we specified in the imputer function.

As we have categorical data for the countries in our dataframe. We have to encode them into numbers so that the model can understand which value we are referring to rather than looking those values as strings.

In order to encode the independent variable, we require the use of these two functions- ColumnTransformer and OneHotEncoder.

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

This code block imports both of those functions for us to use.

ct = ColumnTransformer(transformers=[(&#39;encoder&#39;, OneHotEncoder(), [0])],

remainder=&#39;passthrough&#39;)

Now we create an instance of the ColumnTransformer named &#39;ct&#39; and we use OneHotEncoder as the encoder for this function. The remainder method takes cares of the remaining dataset which we do not want to adjust. The &#39;passthrough&#39; value means that the rest of the columns do no get affected only the mentioned column &#39;0&#39; gets OHE.

X = np.array(ct.fit\_transform(X))

The fit\_transform function applies the transformation on the column specified in the &#39;ct&#39; variable.

To encode the dependent variable, we use LabelEnocder function as this column only has yes/no values.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit\_transform(y)

We import the function from the library. Then we create an instance of it named &#39;le&#39;. Finally, we apply the transformation on the dependent variable column.

Now we are ready to split the dataset into trainset and test set as we have all the columns converted into numerical figures so that we can easily train a model.

from sklearn.model\_selection import train\_test\_split

We import the train\_test\_split function from the sklearn library.

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size = 0.2

random\_state = 1

We create the testing and training variables for our dataset. For the method we need to specify the training set first then the test set, the test size is the split we want for our model. The 0.2 means 80% of the dataset is associated to the training and the rest 20% to the testing. The random state variable is used to ensure when you&#39;re sharing the notebook to anyone else, they would have the same results as you when they train their model.

We have reached the end of our journey for our data preprocessing notebook. You would almost require the knowledge of these topics mentioned when you are starting off your own project. Couple of things to note here are, we had a decent looking dataset compared to some of the data that you are going to acquire manually. The columns we already split in a way so that it was easy to differentiate between independent and dependent variable. Taking that into account we still performed all the basics steps you would do as a data scientist while working on a project.

Thank you!!
