#Helpfunctions for Binary_Classification.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel






def AddBinaryString(df,ListOfStrings):
    #One-hot encoding for list of strings that occur in dfJobs.job_title_full
    for string in ListOfStrings:
        if string != '-':
            df['Contains_'+string] = df['job_title_full'].str.find(string)
            df['Contains_'+string] = np.where(df['Contains_'+string]>-1.,1,0)
    return(df)



def Scoring(clf, testX,testy, featureName, clTechnique,):
    ##Plots the ROC curve and calculates the AUC for a trained classifier and test data
    #Input: trained classifier clf, test feature testX, test target testy, 
    #        name of featuremix featureName, and classifying technique clTechnique
    # Output ROC curve
    
    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]

    preds   = clf.predict(testX)
    # make predictions for test data
    predictions  = [round(value) for value in preds]
    # evaluate predictions
    accuracy     = accuracy_score(testy, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    lr_probs     = clf.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs     = lr_probs[:, 1]
    ### calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    fpr, tpr, _ = roc_curve(testy, lr_probs)

    # plot the roc curve for the model
    fig,ax = plt.subplots(figsize = (4,3))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
    plt.plot(fpr, tpr, marker='.', label= clTechnique + ': ROC AUC=%.3f' % (lr_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/'+clTechnique + featureName +'.pdf')
    plt.show()
    plt.close()
    return(fpr,tpr)

#####################
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

def plot2(clf, X,y, featureName, clTechnique,):
    cv = StratifiedKFold(n_splits=6)
    classifier = clf

    tprs       = []
    aucs       = []
    mean_fpr   = np.linspace(0, 1, 100)

    X = np.asarray(X)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize = (6,3))
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.8, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
            label='Chance', alpha=.8)

    mean_tpr     = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = np.mean(abs(np.asarray(aucs)-0.5)+0.5)
    std_auc      = np.std(abs(np.asarray(aucs)-0.5)+0.5)
    ax.plot(mean_fpr, mean_tpr, color='C1',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.9)

    std_tpr    = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='0.5', alpha=.4,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=clTechnique)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('Results')
    plt.savefig('results/'+ clTechnique + featureName +'.pdf')
    plt.show()
    return(mean_auc,std_auc)

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
def plot3(clf, X,y, featureName, clTechnique, PlotSingleLines = False):
    cv = StratifiedKFold(n_splits=6)
    classifier = clf

    tprs       = []
    aucs       = []
    mean_fpr   = np.linspace(0, 1, 100)
    X = np.asarray(X)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize = (5,3.5))
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        if PlotSingleLines:
            viz = plot_roc_curve(classifier, X[test], y[test],
                                 name='fold {}'.format(i),
                                 alpha=0.8, lw=1, ax=ax)
        else:
            viz = plot_roc_curve(classifier, X[test], y[test], name = False,
                                          alpha=0.0, lw=1, ax=ax,label='_nolegend_')
        #### Correct MEAN for auc < 0.5!! Mirror along the 
        if viz.roc_auc > 0.5:
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        else:
            interp_tpr = np.interp(mean_fpr,  viz.tpr, viz.fpr)
        interp_tpr[0] = 0.0 
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=3, color='C0',
            label='No skill', alpha= 1)
    
            
    mean_tpr     = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = np.mean(abs(np.asarray(aucs)-0.5)+0.5)
    std_auc      = np.std(abs(np.asarray(aucs)-0.5)+0.5)
    ax.plot(mean_fpr, mean_tpr, color='C1',
            label='\n'+clTechnique +' ROC\n' +  r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=3, alpha=1)

    std_tpr    = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='0.5', alpha=.4,
                    label=r'$\pm$ 1 $\sigma $' )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title = clTechnique + ',   ' + featureName)
    ax.legend(loc='lower right', handlelength = 3)   #, bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('results/'+ clTechnique + featureName +'.pdf')
    plt.show()
    return(mean_auc,std_auc)
