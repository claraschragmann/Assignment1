# %% Imports
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from numpy import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score
from numpy import clip, column_stack, argmax

# %% Load dataset and create train-test sets
data = load_wine()
X = data.data
y = data.target
var_names = data.feature_names
var_names = [var_names[i][0:-5] for i in range(0, len(var_names))]
var_names = [var_names[i].title().replace(' ','') for i in range(0, len(var_names))]
var_names = [var_names[i].title().replace('/','_') for i in range(0, len(var_names))]

# is 0 where it is not Class 0
y_0_vs_all = copy(y)
y_0_vs_all[y_0_vs_all==0] = -1
y_0_vs_all[y_0_vs_all!=-1] = 0
y_0_vs_all[y_0_vs_all==-1] = 1

# is 0 where it is not Class 1
y_1_vs_all = copy(y)
y_1_vs_all[y_1_vs_all==1] = -1
y_1_vs_all[y_1_vs_all!=-1] = 0
y_1_vs_all[y_1_vs_all==-1] = 1

# is 0 where it is not Class 2
y_2_vs_all = copy(y)
y_2_vs_all[y_2_vs_all==2] = -1
y_2_vs_all[y_2_vs_all!=-1] = 0
y_2_vs_all[y_2_vs_all==-1] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_0_vs_all, _ = train_test_split(X, y_0_vs_all, test_size=0.2, random_state=42)
_, _, y_train_1_vs_all, _ = train_test_split(X, y_1_vs_all, test_size=0.2, random_state=42)
_, _, y_train_2_vs_all, _ = train_test_split(X, y_2_vs_all, test_size=0.2, random_state=42)


# %% Train 0 vs all

# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_0_vs_all, nr_clus=10) #clusters have to be estimated
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_0_vs_all, part_matrix)
conseq_params = ce.suglms()
#geht bis hier
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_0_vs_all = modbuilder.get_model()

# %% Train 1 vs all model

# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_1_vs_all, nr_clus=10)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_1_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_1_vs_all = modbuilder.get_model()


# %% Train 2 vs all model

# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_2_vs_all, nr_clus=10)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_2_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_2_vs_all = modbuilder.get_model()

# %% Get class probabilities predictions for each ova model

# Class probabilities predictions for 0 vs all
modtester = SugenoFISTester(model_0_vs_all, X_test, var_names)
y_pred_probs_0_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_0_vs_all = column_stack((1 - y_pred_probs_0_vs_all, y_pred_probs_0_vs_all))

# Class probabilities predictions for 1 vs all
modtester = SugenoFISTester(model_1_vs_all, X_test, var_names)
y_pred_probs_1_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_1_vs_all = column_stack((1 - y_pred_probs_1_vs_all, y_pred_probs_1_vs_all))

# Class probabilities predictions for 2 vs all
modtester = SugenoFISTester(model_2_vs_all, X_test, var_names)
y_pred_probs_2_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_2_vs_all = column_stack((1 - y_pred_probs_2_vs_all, y_pred_probs_2_vs_all))

# %% Aggregate class probabilities and get class predictions

y_pred_probs = column_stack((y_pred_probs_0_vs_all[:,1],y_pred_probs_0_vs_all[:,0],y_pred_probs_0_vs_all[:,0])) +\
               column_stack((y_pred_probs_1_vs_all[:,0],y_pred_probs_1_vs_all[:,1],y_pred_probs_1_vs_all[:,0])) +\
               column_stack((y_pred_probs_2_vs_all[:,0],y_pred_probs_2_vs_all[:,0],y_pred_probs_2_vs_all[:,1]))
y_pred_probs = y_pred_probs/y_pred_probs.sum(axis=1,keepdims=1)

y_pred = argmax(y_pred_probs,axis=1)

# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))