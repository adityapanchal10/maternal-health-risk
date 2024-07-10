import pandas as pd
from sklearn.metrics import confusion_matrix    


# returns the probability of a given group (groupname) to be assigned to the positively predicted class (positive_class). 
def group_fairness(df,group,groupname,prediction,positive_class):
    value = round(len(df[(df[prediction] == positive_class) & (df[group] == groupname)]) / len(df[df[group] == groupname]),3)
    return value

# returns the probability of a given group (groupname) with a given condition (conditionname) to be assigned to the positively predicted class (positive_class). 
def conditional_statistical_parity(df,group,groupname,prediction,positive_class,condition,conditionname):
    value = round(len(df[(df[prediction] == positive_class) & (df[group] == groupname) & (df[condition] == conditionname)]) / len(df[(df[group] == groupname) & (df[condition] == conditionname)]), 3)
    return value


# returns the probability for subjects of a group (groupname) to truly belong to the positive class (Diabetes / 1). 
def predictive_parity(df,group,groupname,prediction,true_label):
    try:
        group_df = df.loc[df[group] == groupname]
        matrix = confusion_matrix(group_df[prediction], group_df[true_label], labels=[1, 0])
        PPV = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        return PPV
    
    except ValueError:
        print(f"Oops, something went wrong! Are {prediction} and {true_label} binary variables?")


# returns the probability for subjects of a group (groupname) to actually belong to the negative class even though they were predicted positive. 
def fp_error_rate_balance(df,group,groupname,prediction,true_label):
    try:
        group_df = df.loc[df[group] == groupname]
        matrix = confusion_matrix(group_df[prediction], group_df[true_label], labels=[1, 0])
        FPR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
        return FPR

    except ValueError:
        print(f"Oops, something went wrong! Are {prediction} and {true_label} binary variables?")
        