import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Tools to build machine learning models and reports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

#global variable
random_seed = 1969

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

def xgbooster(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the XGBoost classifier on the training and validation test sets.
    """
    #Changes alpha to numeric
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.fit_transform(y_val)
    #Creating the model
    xgb_model = xgb.XGBClassifier(max_depth=2,
              n_estimators= 100,
              n_jobs=-1,
              random_state = 1969)
    
    #Fitting the KNN model
    xgb_model.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = xgb_model.predict(X_train)
        y_pred_val = xgb_model.predict(X_val)
        
        train_score = xgb_model.score(X_train, y_train)
        val_score =  xgb_model.score(X_val, y_val)
        method = 'Accuracy'

    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = xgb_model.predict(X_train)
        y_pred_val = xgb_model.predict(X_val)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_val, y_pred_val, average='micro')
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = xgb_model.predict(X_train)
        y_pred_val = xgb_model.predict(X_val)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_val, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for XGBoost classifier on training set:   {train_score:.4f}')
        print(f'{method} for XGBoost classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))

    return train_score, val_score
    
def lr_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969,
                               max_iter=500,
                               solver='saga',
                               penalty='l1',
                               n_jobs=-1)

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

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_val, y_pred_val, average='micro')
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_val, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Logistic Regression classifier on training set:   {train_score:.4f}')
        print(f'{method} for Logistic Regression classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score

def rand_forest(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(class_weight="balanced_subsample", 
                                criterion="entropy",
                                min_samples_leaf=3,
                                n_estimators=100,
                                max_depth=6, 
                                random_state=1969)
    
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

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_val, y_pred_val, average='micro')
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_val, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Random Forest classifier on training set:   {train_score:.4f}')
        print(f'{method} for Random Forest classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))

    return train_score, val_score

def dec_tree(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Decision Tree classifier on the training and validation test sets.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=6, random_state=1969)
    
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

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_val, y_pred_val, average='micro')
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_val, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
        print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score

def knn_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the KNN classifier on the training and validation test sets.
    """
    #Creating the model
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

    #Fitting the KNN model
    knn.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        train_score = knn.score(X_train, y_train)
        val_score =  knn.score(X_val, y_val)
        y_pred_val = knn.predict(X_val)

        method = 'Accuracy'

    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_val)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_val, y_pred_val, average='micro')
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_val)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_val, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for KNN classifier on training set:   {train_score:.4f}')
        print(f'{method} for KNN classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))

    return train_score, val_score

    
def find_model_scores(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function takes in the target DataFrame, runs the data against four
    machine learning models and outputs some visuals.
    """

    #Eastablishes the standard to beat
    baseline = round(len(y_train[y_train == 'D'])/ len(y_train),4)
    
    #List for gathering metrics
    model_scores = []
    
    """ *** Builds and fits XGBoost Model *** """    
    train_score, val_score = xgbooster(X_train, y_train, X_val, y_val, metric=metric)    

    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'XGBoost',
                    'Accuracy on Train': round(train_score,4),
                    'Accuracy on Validate': round(val_score,4)})
    
    """ *** Builds and fits Decision Tree Model *** """
    
    
    train_score, val_score = dec_tree(X_train, y_train, X_val, y_val, metric=metric)

    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Decision Tree',
                    'Accuracy on Train': round(train_score,4),
                    'Accuracy on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits Random Forest Model *** """
   
    
    train_score, val_score = rand_forest(X_train, y_train, X_val, y_val, metric=metric)
    
    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Random Forest',
                    'Accuracy on Train': round(train_score,4),
                    'Accuracy on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits KNN Model *** """
    
    train_score, val_score = knn_mod(X_train, y_train, X_val, y_val, metric=metric)
    
    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'KNN',
                        'Accuracy on Train': round(train_score,4),
                        'Accuracy on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits Polynomial regression Model *** """

    
    train_score, val_score = lr_mod(X_train, y_train, X_val, y_val, metric=metric)

    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Logistic Regression',
                        'Accuracy on Train': round(train_score,4),
                        'Accuracy on Validate': round(val_score,4)})
    
    """ *** Later comparison section to display results *** """
    
    #Builds and displays results DataFrame
    model_scores = pd.DataFrame(model_scores)
    model_scores['Difference'] = round(model_scores['Accuracy on Train'] - model_scores['Accuracy on Validate'],2)    
    
    #Results were too close so had to look at the numbers
    if print_scores == True:
        print(model_scores)
    
    #Building variables for plotting
    score_min = min([model_scores['Accuracy on Train'].min(),
                    model_scores['Accuracy on Validate'].min(), baseline])
    score_max = max([model_scores['Accuracy on Train'].max(),
                    model_scores['Accuracy on Validate'].max(), baseline])

    lower_limit = score_min * 0.8
    upper_limit = score_max * 1.05


    x = np.arange(len(model_scores))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(facecolor="gainsboro")
    ax.axhspan(0, baseline, facecolor='red', alpha=0.2)
    ax.axhspan(baseline, upper_limit, facecolor='palegreen', alpha=0.3)
    rects1 = ax.bar(x - width/2, model_scores['Accuracy on Train'],
                    width, label='Training data', color='#4e5e33',
                    edgecolor='dimgray') #Codeup dark green
    rects2 = ax.bar(x + width/2, model_scores['Accuracy on Validate'],
                    width, label='Validation data', color='#8bc34b',
                    edgecolor='dimgray') #Codeup light green

    # Need to have baseline input:
    plt.axhline(baseline, label="Baseline Accuracy", c='red', linestyle=':')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel('Accuracy Score')
    ax.set_xlabel('Machine Learning Models')
    ax.set_title('Model Accuracy Scores')
    ax.set_xticks(x, model_scores['Model'])

    plt.ylim(bottom=lower_limit, top = upper_limit)

    ax.legend(loc='upper left', framealpha=.9, facecolor="whitesmoke",
              edgecolor='darkolivegreen')

    #ax.bar_label(rects1, padding=4)
    #ax.bar_label(rects2, padding=4)
    fig.tight_layout()
    #plt.savefig('best_model_all_features.png')
    plt.show()

def final_test(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    This function takes in the target DataFrame, runs the data against the
    machine learning model selected for the final test and outputs some visuals.
    """
    
    #Eastablishes the standard to beat
    baseline = round(len(y_train[y_train == 'D'])/ len(y_train),4)
    
    #List for gathering metrics
    final_model_scores = []
    
    """ *** Builds and fits Logistic Regression Model *** """  
    
    #Creating the random forest object
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969,
                               max_iter=500,
                               solver='saga',
                               penalty='l1',
                               n_jobs=-1)

    #Fit the model to the train data
    logit.fit(X_train, y_train)

    #Get the accuracy scores
    train_score = logit.score(X_train, y_train)
    val_score =  logit.score(X_val, y_val)
    test_score = logit.score(X_test, y_test)

    #Adds score to metrics list for comparison
    final_model_scores.append({'Model':'Logistic Regression',
                              'Accuracy on Train': round(train_score,4), 
                              'Accuracy on Validate': round(val_score,4), 
                              'Accuracy on Test': round(test_score,4)})
    #Turn scores into a DataFrame
    final_model_scores = pd.DataFrame(data = final_model_scores)
    print(final_model_scores)
    
    #Create visuals to show the results
    fig, ax = plt.subplots(facecolor="gainsboro")

    plt.figure(figsize=(6,6))
    ax.set_title('Logistic Regression results')
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
    