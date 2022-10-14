import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
# import for knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def cleanData(df):
    #  taking care of the NA values
    # for Age, we just complete the few NA value by the mean of the NA
    df.Age = df.Age.fillna(df.Age.mean())
    # for the Cabin field, we have to many NA value, so we just drop the column
    df = df.drop(columns=["Cabin"])
    # for the Embarked column, only 2 value are missing, so we can just drop the line that contain it
    df = df.dropna()

    return df

def transformData(df):
    # first we turn the sex column into quantitative Variable
    df.Sex = df.Sex.replace("male", 1)
    df.Sex = df.Sex.replace("female", 0)

    # now, we creat a new column for each place of embarquement
    # for port in df.Embarked .unique():
        # val = df.Embarked  == port
        # df = df.assign(**{port : val.astype(int)})

    # for ticket in df.Ticket.unique():
        # val = df.Ticket== ticket
        # df = df.assign(**{ticket: val.astype(int)})


    #  now we remove the useless column in order to keep a usable dimention
    df = df.drop(columns=["Embarked", "Name", "Ticket", "PassengerId"])
    # print(df.info()) 

    #  now we normalise our data using min max normalisation to have all our data at the same scale
    df=(df-df.min())/(df.max()-df.min())

    return df




def buildKNNModel(df):
    # definding our variable to train
    y= df.Survived
    x= df.drop(columns=["Survived"])
    # preparation of model
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # creating an instance of our model
    knn = KNeighborsClassifier(n_neighbors = 2)
    # now, we train our model
    knn.fit(x_train,y_train)

    # we check the accuracy or our model
    print(knn.score(x_test,y_test))    
    # optimisation test
    KNN = KNeighborsClassifier()
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    # defining parameter range
    grid = GridSearchCV(KNN, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
    # fitting the model for grid search
    grid_search=grid.fit(x_train,y_train)


    print(grid_search.best_params_)

    # checking the model with the optimal parameter
    knn = KNeighborsClassifier(n_neighbors = 4)
    # now, we train our model
    knn.fit(x_train,y_train)

    # we check the accuracy or our model
    print(knn.score(x_test,y_test))
    print("Accuracy for our training dataset with tuning is : {:.2%}".format(grid_search.best_score_) )

    #  prediction
    y_predict=grid_search.predict(x_test)
    print(classification_report(y_test,y_predict))





def main():
    df =pd.read_csv("train.csv")
    cleanedDf= cleanData(df)
    transformedDf= transformData(cleanedDf)
    buildKNNModel(transformedDf)




if __name__ == '__main__':
    main()