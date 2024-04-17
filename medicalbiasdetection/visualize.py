# Data Manipulatoin
import pandas as pd
import numpy as np
import json

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# System
import os

# Modeling Metrics
from sklearn.metrics import (precision_recall_curve, precision_recall_fscore_support, roc_curve, roc_auc_score,
                             f1_score, auc, accuracy_score, confusion_matrix,fbeta_score,classification_report)


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          save_path=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        #normalized by cases
        group_percentages = ["{0:.2%}".format(value) for value in (cf.transpose()/np.sum(cf, axis = 1)).transpose().flatten()]
        
        #group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    stats_text = ""
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[0,0] / sum(cf[:,0])
            recall    = cf[0,0] / sum(cf[0,:])
            f1_score  = 2*precision*recall / (precision + recall)
            f2_score =  5 *((precision*recall)/((4*precision)+recall))
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nF2 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score,f2_score)
            print(stats_text)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    

    if save_path and stats_text:
        stats_file_path = os.path.splitext(save_path)[0] + '_stats.txt'
        with open(stats_file_path, 'w') as file:
            file.write(stats_text)
            
    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap((cf.transpose()/np.sum(cf, axis = 1)).transpose(),
                annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label') # + stats_text
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=400)
        
    
        
def plot_cfmatrix(df,y_true,y_pred,title=None, save_path=None):
    actual = df[y_true]
    predicted = df[y_pred]

    cf_matrix = confusion_matrix(actual,predicted, labels=[1,0])
    print('Confusion matrix : \n',cf_matrix)

    # outcome values order in sklearn
    tp = cf_matrix[0][0]
    fn = cf_matrix[0][1]
    fp = cf_matrix[1][0]
    tn = cf_matrix[1][1]
    print('Outcome values : ', "\nTP: ", tp, "\nFN: ", fn, "\nFP: ",fp, "\nTN: ",tn)

    matrix = classification_report(actual,predicted,labels=[1,0], digits = 4)
    print('Classification report : \n',matrix)

    labels = ["TP", "FN", "FP", "TN"]
    categories = ["sepsis", "no sepsis"]

    make_confusion_matrix(cf_matrix, 
                          group_names=labels,
                          categories=categories, title = title,
                          figsize=(8,6),
                          save_path=save_path)
    
def plot_roc_auc(df, y_true, y_prob, title=None, save_path=None):   
    
    y_true = df[y_true]
    y_scores = df[y_prob]

    fpr, tpr, thresholds = roc_curve(y_true,y_scores)
    roc_auc = auc(fpr,tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--', label='Baseline')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()
    
def remove_svg_elements(svg_data,id_list,outpath):
    # Parse the SVG data
    root = ET.fromstring(svg_data)
    for i in id_list:
        # print(i)
        # Find the specific <g> element to remove
        for g in root.findall(f".//{{http://www.w3.org/2000/svg}}g[@id='{i}']"):
            g.clear()

    # Update the SVG data string
    svg_data_modified = ET.tostring(root, encoding='unicode')

    # Save or further process the modified SVG data
    outpath +=".svg"
    with open(outpath, "w") as f:
        f.write(svg_data_modified)
       
    
def create_donut_plot(ax, names, sizes):
    # Validate inputs
    if not isinstance(ax, plt.Axes):
        raise TypeError("Expected plt.Axes for 'ax', got {}".format(type(ax)))
    if not isinstance(names, list) or not isinstance(sizes, list):
        names = list(names)
        sizes = list(sizes)
        # raise TypeError("Expected lists for 'names' and 'sizes'")
    if len(names) != len(sizes):
        raise ValueError("'names' and 'sizes' lists must have the same length")
    
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#9999FF','#FFD599']
    blues = ['#87CEFA','#83A6DA','#576E91','#1167F2','#5B9DFF','#99C2FF']
    
    # Create a circle at the center of the plot
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    
    # Custom wedges
    wedges, texts, autotexts = ax.pie(sizes, labels=names, autopct='%1.1f%%', 
                                  wedgeprops={'linewidth': 5, 'edgecolor': 'white'}, colors=blues)

    # Make percent texts more legible
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_weight('bold')
        
    ax.add_artist(my_circle)
    