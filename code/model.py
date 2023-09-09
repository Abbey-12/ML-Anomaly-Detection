from json import load
import joblib
import pandas as pd
import warnings
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

#  load data
train_data = pd.read_csv("/app/data/Train.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])

test_data = pd.read_csv("/app/data/Test.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])
print("Size of Training data:",train_data.shape)
print("Size of Test  data:",test_data.shape)

# Unique Value Analysis
def CountUnique(d):
    for (i,j) in d.iteritems():
        # print(i," ",d[i].unique())
        if len(d[i].unique())==0 or len(d[i].unique())==1:
            d.drop([i],axis=1,inplace=True)
#print("Unique Values in Train Data ")

print("****************** Train Data ****************************")
CountUnique(train_data)
print("************* Test Data*********************************")
CountUnique(test_data)

print("**********************************************")
print(train_data.shape)

# Select only numerical columns
num_columns = train_data.select_dtypes(exclude = "object").columns
# print(num_columns)

# Drop Duplicates
train_data= train_data.drop_duplicates()
test_data=test_data.drop_duplicates()

# Test split
X_train=train_data.drop("attack",axis=1)
Y_train=train_data["attack"]
Y_train.columns=["attack"]
X_test=test_data.drop("attack",axis=1)
Y_test=test_data["attack"]
Y_test.columns=["attack"]

# Encoding Target Variable
def encodeTarget(data):
    
    data=pd.DataFrame(data)
    data["attack"]=data["attack"].apply(lambda x:0 if x=="normal" else 1)
    return data
Y_train=encodeTarget(Y_train)
Y_test=encodeTarget(Y_test)

sns.catplot(x="attack", kind="count", palette="ch:.25", data=Y_train)
plt.show

# Normalize data using a saved scaler
def NormaliseData(data, training=True, rs=None, num_columns=5):
    if training == True:
        sc=StandardScaler()
        rs = sc.fit(data.loc[:,num_columns])
        data.loc[:,num_columns] = rs.transform(data.loc[:,num_columns])
    if training == False and rs !=None:
        data.loc[:,num_columns] = rs.transform(data.loc[:,num_columns])
    return data, rs

X_train_ns, rs=NormaliseData(X_train,num_columns = num_columns)
X_test_ns, rs =NormaliseData(X_test, training=False, rs=rs, num_columns = num_columns)
X_train_ns.shape,X_test_ns.shape

# One hot Encoding on Non target Data
network_ohe_train=pd.get_dummies(X_train_ns,drop_first=True)
network_ohe_test=pd.get_dummies(X_test_ns,drop_first=True)
network_ohe_train.shape,network_ohe_test.shape

# feature selection
def featureSelect(data,test):
    model_dt = DecisionTreeClassifier(criterion = 'gini',random_state=500)
    model_dt.fit(data,test)
    
    print(data.columns[model_dt.feature_importances_ >= 0.001].unique())
    return model_dt   
X_train=featureSelect(network_ohe_train,Y_train)
X_test=network_ohe_test[network_ohe_train.columns[X_train.feature_importances_ >= 0.001]]
X_train=network_ohe_train[network_ohe_train.columns[X_train.feature_importances_ >= 0.001]]
X_train

def eval_model(Y_test,Y_predict):
    f1=f1_score(Y_test,Y_predict)
    prec=precision_score(Y_test,Y_predict)
    recall=recall_score(Y_test,Y_predict)
  # Calculate Accuracy
    acc = accuracy_score(Y_test,Y_predict)
    return f1,prec,recall, acc

## Creating Model List
model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),SVC(),
              AdaBoostClassifier(),XGBClassifier(),GradientBoostingClassifier()]

for model in model_list:
    print(model.__class__)
  # train the model
    model.fit(X_train,Y_train)
    Y_pred_train=model.predict(X_train)
    
  # test the model on X_test
    Y_predict = model.predict(X_test)
    
    ## Saving model
    model_name = type(model).__name__
    model_save="/app/models"+model_name+"StdSc.sav"
    joblib.dump(model,model_save)
    
    ## Evaluating Model
    f1,prec,recall, acc = eval_model(Y_test,Y_predict)
    
    #Print Results
    print("F1 Score :",f1)
    print("Precision : ",prec)
    print("Recall : ",recall)
    print("Accuracy : ",acc)
    
    plot_confusion_matrix(model, X_test, Y_test)  
    plt.show()
    print("***************************")
    print()

