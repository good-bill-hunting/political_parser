import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test

def vectorize_data(X_train, X_val, X_test, target_col):
    """
    Transforms data for modeling
    """
    #Creates object
    tfidf = TfidfVectorizer()
    
    #Uses object to change data
    X_train = tfidf.fit_transform(X_train[target_col])
    X_val = tfidf.transform(X_val[target_col])
    X_test = tfidf.transform(X_test[target_col])
    
    return X_train, X_val, X_test

    
def lr_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=77)

    #Fitting the model to the train dataset
    logit.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)        
        
        train_score = logit.score(X_train, y_train)
        val_score =  logit.score(X_val, y_val)
        method = 'Accuracy'

    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Logistic Regression classifier on training set:   {train_score:.4f}')
        print(f'{method} for Logistic Regression classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score

def rand_forest(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(bootstrap=True,
                                class_weight=None, 
                                criterion='gini',
                                min_samples_leaf=5,
                                n_estimators=250,
                                max_depth=6, 
                                random_state=77)
    
    #Fit the model to the train data
    rf.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)
        
        train_score = rf.score(X_train, y_train)
        val_score =  rf.score(X_val, y_val)
        method = 'Accuracy'
    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Random Forest classifier on training set:   {train_score:.4f}')
        print(f'{method} for Random Forest classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))

    return train_score, val_score

def dec_tree(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Decision Tree classifier on the training and validation test sets.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=6, random_state=77)
    
    #Train the model
    clf = clf.fit(X_train, y_train)
    
    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        
        train_score = clf.score(X_train, y_train)
        val_score =  clf.score(X_val, y_val)
        method = 'Accuracy'
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
        print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score


def final_test(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    This function takes in the target DataFrame, runs the data against the
    machine learning model selected for the final test and outputs some visuals.
    """
    
    #Eastablishes the standard to beat
    baseline = round(len(y_train[y_train == 'Ruby'])/len(y_train),4)
    
    #List for gathering metrics
    final_model_scores = []
    
    """ *** Builds and fits Random Forest Model *** """  
    
    #Creating the random forest object
    clf = DecisionTreeClassifier(max_depth=6, random_state=77)


    #Fit the model to the train data
    clf.fit(X_train, y_train)

    #Get the accuracy scores
    train_score = clf.score(X_train, y_train)
    val_score =  clf.score(X_val, y_val)
    test_score = clf.score(X_test, y_test)

    #Adds score to metrics list for comparison
    final_model_scores.append({'Model':'Decision Tree',
                              'Accuracy on Train': round(train_score,4), 
                              'Accuracy on Validate': round(val_score,4), 
                              'Accuracy on Test': round(test_score,4)})
    #Turn scores into a DataFrame
    final_model_scores = pd.DataFrame(data = final_model_scores)
    print(final_model_scores)
    
    #Create visuals to show the results
    fig, ax = plt.subplots(facecolor="gainsboro")

    plt.figure(figsize=(6,6))
    ax.set_title('Decision Tree results')
    ax.axhspan(0, baseline, facecolor='red', alpha=0.2)
    ax.axhspan(baseline, ymax=2, facecolor='palegreen', alpha=0.3)
    ax.axhline(baseline, label="Baseline", c='red', linestyle=':')

    ax.set_ylabel('Accuracy Score')    

    #x_pos = [0.5, 1, 1.5]
    width = 0.25

    bar1 = ax.bar(0.5, height=final_model_scores['Accuracy on Train'],width =width, color=('#4e5e33'), label='Train', edgecolor='dimgray')
    bar2 = ax.bar(1, height= final_model_scores['Accuracy on Validate'], width =width, color=('#8bc34b'), label='Validate', edgecolor='dimgray')
    bar3 = ax.bar(1.5, height=final_model_scores['Accuracy on Test'], width =width, color=('tomato'), label='Test', edgecolor='dimgray')

    # Need to have baseline input:
    ax.set_xticks([0.5, 1.0, 1.5], ['Training', 'Validation', 'Test']) 
    ax.set_ylim(bottom=0, top=1)
    #Zoom into the important area
    #plt.ylim(bottom=200000, top=400000)
    #ax.legend(loc='lower right', framealpha=.9, facecolor="whitesmoke", edgecolor='darkolivegreen')
    
    
def plot_model_scores(train_score0, val_score0, train_score1, val_score1, train_score2, val_score2, y_train):
    """
    Takes models scores and creats a plot
    """
    
    #Eastablishes the standard to beat
    baseline = round(len(y_train[y_train == 'Ruby'])/len(y_train),4)
    
    #Visual for train and validate data
    modeling_scores = pd.DataFrame({'Logistic Regression Train': train_score0,
                                    'Logistic Regression Validate': val_score0,
                                    'Decision Tree Train':train_score1,
                                    'Decision Tree Validate': val_score1,
                                    'Random Forest Train': train_score2,
                                    'Random Forest Validate':val_score2}, index = [0,1,2,3,4,5])
    
    model_colors = {'Logistic Regression Train': '#4e5e33',
                'Logistic Regression Validate': '#8bc34b',
                'Decision Tree Train':'#4e5e33',
                'Decision Tree Validate': '#8bc34b',
                'Random Forest Train': '#4e5e33',
                'Random Forest Validate':'#8bc34b'}
    
    baseline = round(len(y_train[y_train == 'Ruby'])/len(y_train),4)
    
    plt.subplots(facecolor="gainsboro")
    plt.axhspan(0, baseline, facecolor="red", alpha=0.2)
    plt.axhspan(baseline, ymax=2, facecolor="palegreen", alpha=0.3)
    plt.axhline(baseline, label="Baseline", c="red", linestyle=":")
    sns.barplot(modeling_scores, palette=model_colors)
    plt.ylim(bottom =0.3, top=1.0)
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy Score") 
    plt.title("Train and Validate Scores")