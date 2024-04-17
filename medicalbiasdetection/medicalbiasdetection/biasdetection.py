
import os


# Data Manipulation
import pandas as pd
import numpy as np
import json
from six import StringIO
from matplotlib import pyplot as plt
import graphviz
import matplotlib.image as mpimg
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
import ast




def create_path(model,leaf_data,X_names,start_node, verbose=False):
    left_arr = model.tree_.children_left
    right_arr = model.tree_.children_right
    path = [start_node]
    parent = None
    signs = []
    sign = None
    while parent != 0:
        curr_val = path[-1]
        if curr_val in left_arr:
            parent = np.where(left_arr ==curr_val)[0][0]
            sign = '<'
        elif curr_val in right_arr:
            parent = np.where(right_arr==curr_val)[0][0]
            sign = '>'
        else:
            parent = 0
        path.append(parent)
        signs.append(sign)
    path = path[::-1]
    signs = signs[::-1]
    if verbose:
        print(path)
    feat_dict = {ind:col for ind,col in enumerate(X_names)}
    path_names = []
    for node in range(0,len(path)-1):
        path_names.append(feat_dict[model.tree_.feature[path[node]]])
    min_val = leaf_data['value'].min()
    if verbose:
        print(f"The minimized accuracy is {min_val}")
    thresh = model.tree_.threshold[path]
    thresh_rnd = model.tree_.threshold[path]
    if verbose:
        print(thresh)
    full_path = list(zip(path_names,signs,thresh_rnd))
    return full_path, path


def generate_tree_graph(model, X_names, y_name, outpath, min_node_path=None, num_nodes=None,fill_color='green', verbose=False, std_deviations=None, mark_all_paths=False):
    
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, 
                                feature_names=X_names,  
                                class_names=y_name,  
                                filled=True,
                                rounded=True,
                                # style = 'bold',
                                special_characters=True)
    
    # if there are standard deviations, update the dot data to include the standard deviation in each node
    if std_deviations:
        # get the dot data 
        dot_string = dot_data.getvalue()
        # Loop through each node and its standard deviation
        for node_id, std_dev in std_deviations.items():
            # skip decision nodes
            if std_dev:
                # Find the start of the label for this node
                label_start = dot_string.find(f'{node_id} [label=<') + len(f'{node_id} [label=<')
                if label_start > len(f'{node_id} [label=<'):  # if the node is found
                    # Insert the standard deviation info at the start of the label
                    dot_string = dot_string[:label_start] + f'stddev = {std_dev:.2f}<br/>' + dot_string[label_start:]
    
    graph = pydotplus.graph_from_dot_data(dot_string)     

    if min_node_path:
        for node in range(num_nodes+1):
            dest = graph.get_node(str(node))[0]
            dest.set_fillcolor('white')
        
        for node in min_node_path:
            dest = graph.get_node(str(node))[0]
            # dest.set_fillcolor(fill_color)
            dest.set_color('red')
            
    graph.set_size('"30,15!"')
    graph.write_png(outpath)
    
    
    if verbose:
        plt.figure(figsize=(5,5))
        img = mpimg.imread(outpath)
        imgplot = plt.imshow(img)
        plt.show()
        
def graph_all_paths(model, X_names, y_name, outpath, min_node_path=None, num_nodes=None,fill_color='green', verbose=False, std_deviations=None):
    
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, 
                                feature_names=X_names,  
                                class_names=y_name,  
                                filled=True,
                                rounded=True,
                                # style = 'bold',
                                special_characters=True)
    
    # if there are standard deviations, update the dot data to include the standard deviation in each node
    if std_deviations:
        # get the dot data 
        dot_string = dot_data.getvalue()
        # Loop through each node and its standard deviation
        for node_id, std_dev in std_deviations.items():
            # skip decision nodes
            if std_dev:
                # Find the start of the label for this node
                label_start = dot_string.find(f'{node_id} [label=<') + len(f'{node_id} [label=<')
                if label_start > len(f'{node_id} [label=<'):  # if the node is found
                    # Insert the standard deviation info at the start of the label
                    dot_string = dot_string[:label_start] + f'stddev = {std_dev:.2f}<br/>' + dot_string[label_start:]
    
    graph = pydotplus.graph_from_dot_data(dot_string)     

    # if min_node_path:
    for node in range(num_nodes+1):
        dest = graph.get_node(str(node))[0]
        dest.set_fillcolor('white')

    for path in min_node_path:
        for node in path:
            dest = graph.get_node(str(node))[0]
            # dest.set_fillcolor(fill_color)
            dest.set_color('red')
            
    graph.set_size('"30,15!"')
    graph.write_png(outpath)
    
    
    if verbose:
        plt.figure(figsize=(5,5))
        img = mpimg.imread(outpath)
        imgplot = plt.imshow(img)
        plt.show()
        
        
def node_std_devs(tree, X, y):
    """
    Calculate the standard deviation for each leaf node
    Params:
        tree: decision tree model
        X: input matrix
        y: response variable vector
    """
    # identify the leaf nodes for each record in X
    leaf_ids = tree.apply(X)
    # init the standard deviation array
    std_dev =  {i:None for i in range(tree.tree_.node_count)}
    # for each node in the decision tree
    for node_id in range(tree.tree_.node_count):
        # get the index of the samples belonging to the node
        samples_indexes = np.where(leaf_ids == node_id)
        # if there are samples in the node (indicating a leaf)
        if samples_indexes[0].size > 0:
            # get the response variable values for each sample in the node
            node_samples = y.iloc[samples_indexes]
            # calculate the standard deviation and update the 
            std_dev[node_id] = np.std(node_samples, axis=0)
    return std_dev

def apply_metric_threshold(df, metric='f2_score', n_std=1):
    target ='target'
    result = df.copy()
    df_avg = df[metric].mean()
    df_std = df[metric].std()
    threshold = df_avg - (df_std*n_std)
    mask = np.where(result[metric] < threshold,1,0)
    result[target] = mask
    return result, threshold

def generate_threshold_tree_graph(model, X_names, y_name, performance_df, metric, outpath, min_node_path=None, positive_nodes=None,
                        num_nodes=None,fill_color='green', verbose=False, std_deviations=None, mark_all_paths=False, reverse=False):

    dot_data = StringIO()
    
    export_graphviz(model, out_file=dot_data, 
                                feature_names=X_names,  
                                class_names=y_name,  
                                filled=True,
                                rounded=True,
                                # style = 'bold',
                                special_characters=True,
                               impurity=False, precision=2
                   )
    # get the dot data 
    dot_string = dot_data.getvalue()
    
    graph = pydotplus.graph_from_dot_data(dot_string)
    
    # get the total number of nodes
    n_nodes = model.tree_.node_count
    
    leaf_nodes = get_leafs(performance_df)
    leaf_data = get_leaf_data(performance_df,metric)
    
    if n_nodes > 1:
    
        for node_id in range(n_nodes):
            dot_string = update_node_thresh(dot_string, node_id) 
            if node_id in leaf_nodes:
                leaf_text = leaf_data[node_id]
                dot_string = update_leaf_node(dot_string, node_id, leaf_text)

        if reverse:
            dot_string = reverse_graph(dot_string)


        graph = pydotplus.graph_from_dot_data(dot_string)   

        # set the fill color of the ndoes
        for node_id in range(n_nodes):
            dest = graph.get_node(str(node_id))[0]
            dest.set_fillcolor('green')

        if positive_nodes:
            for node_id in positive_nodes:
                dest = graph.get_node(str(node_id))[0]
                dest.set_fillcolor('red')
            
    graph.set_size('"30,15!"')
    if outpath:
        graph.write_png(outpath)
    
    if verbose:
        plt.figure(figsize=(10,10))
        img = mpimg.imread(outpath)
        imgplot = plt.imshow(img)
        plt.show()
        
def get_leafs(df):
    leafs = df['node'].unique().tolist()
    return leafs

def get_leaf_data(df, metric):
    # print(df.head())
    # calculate leaf statistics
    mu = df.groupby('node')[metric].mean()
    
    sigma = df.groupby('node')[metric].std()
    results = pd.concat([mu,sigma],axis=1,keys=['mu','sigma'])
    
    #init leaf text dictionary
    leaf_text = {}
    
    # loop through each node and create dictionary of text entries
    for row in results.iterrows():
        node_id = row[0]
        mu = round(row[1]['mu'],2)
        sigma = round(row[1]['sigma'],2)
        text = f" mean(&mu;) = {mu}<br/>std dev(&sigma;) = {sigma}<br/>"
        leaf_text[node_id] = text
    
        # print(text)
    return leaf_text

def get_num_samples(node_string):
    samples_idx_start = node_string.find("samples = ") + len("samples = ")
    samples_idx_end = samples_idx_start + node_string[samples_idx_start:].find("<br/>")
    samples = int(node_string[samples_idx_start:samples_idx_end])
    return samples

def get_label_counts(node_string):
    labels_idx_start = node_string.find("value = ") + len("value = ")
    labels_idx_end = labels_idx_start + node_string[labels_idx_start:].find("<br/>")
    labels = ast.literal_eval(node_string[labels_idx_start:labels_idx_end])
    return labels[0], labels[1]

def get_threshold_text(node_string):
    # get the number of samples in the node
    num_samples = get_num_samples(node_string)
    # get the count of each label in the node
    label_0, label_1 = get_label_counts(node_string)
    # calculate percent above threshold
    above_thresh = int(round((label_0/num_samples)*100,0))
    below_thresh = int(round((label_1/num_samples)*100,0))
    threshold_txt = f" &ge; threshold = {label_0} ({above_thresh}%)<br/> &lt; threshold = {label_1} ({below_thresh}%)<br/>"
    return threshold_txt

def insert_node_text(node_string, text):
    # get the start index of the class label
    class_idx_start = node_string.find("class = ")
    updated_string = node_string[:class_idx_start] + text + node_string[class_idx_start:]
    return updated_string

def get_node_string(dot_string, node_id):
    # get node starting index
    node_idx_start = dot_string.find(f'{node_id} [label=<')
    # get node ending index
    node_idx_end = dot_string.find(f'{node_id+1} [label=<')
    # get full node substring
    node_string = dot_string[node_idx_start:node_idx_end]
    # find arrow index, if it exists
    arrow_idx = node_string.find(f'->')
 
    # update node ending index
    if arrow_idx >0:
        # update the end of the node index
        node_idx_end = node_idx_start + arrow_idx - 3
    
        # update the node string
        node_string = dot_string[node_idx_start:node_idx_end]
    return node_idx_start, node_idx_end, node_string

def update_node_thresh(dot_string, node_id):
    
    # get node string
    node_idx_start, node_idx_end, node_string = get_node_string(dot_string, node_id)

    # get the number of samples in the node
    num_samples = get_num_samples(node_string)
    
    # get the count of each label in the node
    label_0, label_1 = get_label_counts(node_string)
    
    # get threshold text
    threshold_text = get_threshold_text(node_string)
    # update node string 
    node_string = insert_node_text(node_string,threshold_text)
    # update dot string
    dot_string = dot_string[:node_idx_start] + node_string + dot_string[node_idx_end:]
    
    return dot_string

def update_leaf_node(dot_string, node_id, leaf_text):
    
    # get node string
    node_idx_start, node_idx_end, node_string = get_node_string(dot_string, node_id)

    # update node string 
    node_string = insert_node_text(node_string, leaf_text)
    
    # update dot string
    dot_string = dot_string[:node_idx_start] + node_string + dot_string[node_idx_end:]
    
    return dot_string


def reverse_graph(dot_string):
    # reverse signs in all branch nodes
    dot_string = dot_string.replace("&le;","&gt;")
    
    # get arrow with "False" string
    false_label_idx_start = dot_string.find('headlabel="False"') + len('headlabel=')
    false_label_idx_end = dot_string.find('headlabel="False"') + len('headlabel="False"')
    
    # replace "False" label with "True"
    dot_string = dot_string[:false_label_idx_start] + '"True"' + dot_string[false_label_idx_end:]
    
    #get arrow with "True" string, this will be the first arrow
    true_label_idx_start = dot_string.find('headlabel="True"') + len('headlabel=')
    true_label_idx_end = dot_string.find('headlabel="True"') + len('headlabel="True"')
    
    # replace first "True" label with "False"
    dot_string = dot_string[:true_label_idx_start] + '"False"' + dot_string[true_label_idx_end:]
    
    return dot_string

def report_removed_records(df, removed_text, rows, reason):
    """Report missing csn records for bias report
    Params:
        df (pd.DataFrame): data frame of patient performance metrics
        report_str (str): current missing csn report
        rows (np.array): index of filtered rows
        reason (str): reason for removal
    """
    if not removed_text:
        removed_text = ""
    
    if rows.shape[0] == 0:
        return removed_text
    
    if not reason:
        print("Enter valid reason")
        return
    
    n = df.shape[0]
    n_rows = rows.shape[0]
    missing = n - n_rows
    update_str = f"{missing} | {reason}\n"
    df_updated = df.loc[rows]
    return df_updated, removed_text + update_str
