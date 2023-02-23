"""
RegressionTrustee
=================
"""
# importing required libraries
import graphviz
import time
import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from trustee import RegressionTrustee

# Loading data
#Specify path where the data is stored
rsir_experiences = '~/SharedFolders/SharedVM_RyuDRL/RL-links2/RL_paths trustee/23nodos/info_RL_paths_23n_30stps.csv'
df = pd.read_csv(rsir_experiences)
print(df.head())
print(df.shape[0])
print(df.shape[1])

# Cut dataset, only from episode 1000 that is the point of convergence of rsir paths:
df_cut = df[df['ep'] > 999]

print(df_cut.shape[0])
print(df_cut.shape[1])

# Instantiate encoder/scaler
scaler = MinMaxScaler()  #default interval [0,1]
ohe = OneHotEncoder(sparse_output=False)

X_df = df_cut[['ep', 'stp', 's', 'a', 'r', 's_', 'src', 'dst', 'src_', 'dst_']].copy()
print(X_df.head())

y_df = df_cut[['q_val']].copy()
print(y_df.head())

# Define which columns should be encoded vs scaled
# cols_to_encode_X = ['s', 's_', 'a', 'src', 'dst', 'src_', 'dst_']
# cols_to_encode_X = ['s', 'a']  
cols_to_encode_X = ['s', 's_', 'a'] 
# cols_to_scale_X = ['ep','stp','r']
cols_to_scale_X = ['r']
col_to_scale_y = ['q_val']

# Scale and Encode Separate Columns
encoded_cols_X = ohe.fit_transform(X_df[cols_to_encode_X])
scaled_cols_X = scaler.fit_transform(X_df[cols_to_scale_X])

feature_names = list(ohe.get_feature_names_out(cols_to_encode_X))
feature_names.insert(0,'r')


# Concatenate Processed Column to X
X = np.concatenate([scaled_cols_X, encoded_cols_X], axis=1)
y = scaler.fit_transform(y_df[col_to_scale_y]).ravel()

# Spliting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# creating an MLP regressor
clf = MLPRegressor(solver="adam", alpha=1e-5, hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# Evaluate model accuracy
print("Model R2-score:")
print(r2_score(y_test, y_pred))
print("Model RMSE:")
print(mean_squared_error(y_test, y_pred))

N_ITERS = [20,25,30,35] 
N_STAB = [1,2,3,5,7,9] 
SAMP_SIZE = [0.2,0.3,0.4] 

#specify the whole path (from /home) where the csv will be saved
with open('/home/dmcasasv/SharedFolders/SharedVM_RyuDRL/RL-links2/RL_paths trustee/23nodos/trustee-master/examples/trustee_search.csv','w') as csvfile:
	
	'''
	HADERS explanation of the csv saved at the end

	Parameter values passed to trustee.fit() function
	'N_ITERS' = num_iter. This variable is S in the paper 
	'N_STAB' = num_stability_iter. This variable is N in the paper
	'SAMP_SIZE' = samples_size. This variable is M in the paper

	Parameter retrieved from trustee report within previous settings
	'METrain_agreement' = Model explanation training agreement
	'MET_fidelity' = Model explanation training fidelity
	'ME_size' = Model Explanation size
	'KME_size' = Top-k Prunned Model explanation size
	'MEG_fidelity' = Model explanation global fidelity
	'KMEG_fidelity' = Top-k Model explanation global fidelity
	'MER2' = Model explanation R2-score
	'KMER2' = Top-k Model explanation R2-score:
	'Time_elapsed' = Total time trustee took for retrieving values within each setting
	
	'''
	header_names = ['N_ITERS', 'N_STAB', 'SAMP_SIZE', 'METrain_agreement', 'MET_fidelity', 'ME_size', 'KME_size', 'MEG_fidelity', 'KMEG_fidelity', 'MER2', 'KMER2', 'Time_elapsed']
	file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	file.writerow(header_names)
	for num_iter in N_ITERS:
		for num_stability_iter in N_STAB:
			for samples_size in SAMP_SIZE:
				print('-----',num_iter,num_stability_iter,samples_size)
				init = time.time()
				# Initialize Trustee and fit for classification models
				trustee = RegressionTrustee(expert=clf)
				trustee.fit(X_train, y_train, num_iter=num_iter, num_stability_iter=num_stability_iter, samples_size=samples_size, verbose=True)

				# Get the best explanation from Trustee
				dt, pruned_dt, agreement, reward = trustee.explain()
				print(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
				print(f"Model Explanation size: {dt.tree_.node_count}")
				print(f"Top-k Prunned Model explanation size: {pruned_dt.tree_.node_count}")

				# Use explanations to make predictions
				dt_y_pred = dt.predict(X_test)
				pruned_dt_y_pred = pruned_dt.predict(X_test)

				# Evaluate accuracy and fidelity of explanations
				print("Model explanation global fidelity:")
				MEG_fidelity = r2_score(y_pred, dt_y_pred)
				print(MEG_fidelity)
				print("Top-k Model explanation global fidelity:")
				KMEG_fidelity = r2_score(y_pred, pruned_dt_y_pred)
				print(KMEG_fidelity)

				print("Model explanation R2-score:")
				MER2 = r2_score(y_test, dt_y_pred)
				print(MER2)
				print("Top-k Model explanation R2-score:")
				KMER2 = r2_score(y_test, pruned_dt_y_pred)
				print(KMER2)
				end = time.time()-init
				print("Total time trustee: ", end)


				# Output decision tree to pdf
				dot_data = tree.export_graphviz(
					dt,
					feature_names=feature_names,
					filled=True,
					rounded=True,
					special_characters=True,
				)
				graph = graphviz.Source(dot_data)
				graph.render("dt_explanation"+"_N_ITERS_"+str(num_iter)+"_N_STAB_"+str(num_stability_iter)+"_SAMP_SIZE_"+str(samples_size))

				# Output pruned decision tree to pdf
				dot_data = tree.export_graphviz(
					pruned_dt,
					feature_names=feature_names,
					filled=True,
					rounded=True,
					special_characters=True,
				)
				graph = graphviz.Source(dot_data)
				graph.render("pruned_dt_explation"+"_N_ITERS_"+str(num_iter)+"_N_STAB_"+str(num_stability_iter)+"_SAMP_SIZE_"+str(samples_size))
			
				file.writerow([num_iter,num_stability_iter,samples_size,agreement,reward,dt.tree_.node_count,pruned_dt.tree_.node_count,MEG_fidelity,KMEG_fidelity,MER2,KMER2,end])
