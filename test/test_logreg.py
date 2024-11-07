"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from regression import logreg, utils

def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?

    # just so it passes all the time lol
    np.random.seed(42)

    # load in dataset 
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=['Penicillin V Potassium 500 MG', 
                  'Plain chest X-ray (procedure)', 
                  'Glucose',
                  'Low Density Lipoprotein Cholesterol',
                  'Creatinine', 
                  'AGE_DIAGNOSIS', 
                  'Total Cholesterol'],
        split_percent=0.8, split_state=42
    )
    
    # scale features training and validation sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # initialize logistic regression model
    log_model = logreg.LogisticRegression(num_feats=7, max_iter=1000, tol=0.01, learning_rate=0.05, batch_size=12)
    
    # train model and record training and validation losses
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # checks if len of loss history is not 0
    assert len(log_model.loss_history_train) !=0
    assert len(log_model.loss_history_val) != 0

    # checks if weights are in a reasonable range 
    assert np.all(np.abs(log_model.W) < 100)

    # checks if the final training loss is below a reasonable threshold to see if it converges (guesstimate at 5 which should be enough for convergence for batchsize 12)
    assert log_model.loss_history_train[-1] < 10
    
    # checks if the final validation loss is in a reasonable range (guestimating around the 300 or less range)
    assert log_model.loss_history_val[-1] < 300

    # checks if loss values decrease over time bc that means that the model is learning 
    assert log_model.loss_history_train[0] > log_model.loss_history_train[-1] #checks training loss
    assert log_model.loss_history_val[0] > log_model.loss_history_val[-1] # checks validation loss
    
def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training

    # so it can keep passing the tests w the same numbers
    np.random.seed(42)

    # load in dataset 
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=['Penicillin V Potassium 500 MG', 
                  'Plain chest X-ray (procedure)', 
                  'Glucose',
                  'Low Density Lipoprotein Cholesterol',
                  'Creatinine', 
                  'AGE_DIAGNOSIS', 
                  'Total Cholesterol'],
        split_percent=0.8, split_state=42
    )
    
    # scale features training and validation sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # initialize logistic regression model
    log_model = logreg.LogisticRegression(num_feats=7, max_iter=1000, tol=0.01, learning_rate=0.05, batch_size=12)
    
    # copy the initial weights to compare after training
    initial_weights = np.copy(log_model.W)

    # train model and record training and validation losses
    log_model.train_model(X_train, y_train, X_val, y_val)

    # check if the weights are different before training and after
    assert not np.allclose(initial_weights, log_model.W)

    # predict probabilities for the validation set
    predictions = log_model.make_prediction(X_val)

    # checks if predictions (prob sscores) are between 0-1  
    assert np.all(predictions >= 0) and np.all(predictions <= 1)

    # checks if prob scores have a are close to class balance 
    assert 0.40 <= np.mean(predictions) <= 0.60

    # convert probability predictions to binary labels using a threshold of 0.5
    predicted_labels = (predictions >= 0.5).astype(int)

    # calculate accuracy as proportion of correct predictions
    accuracy = np.mean(predicted_labels == y_val)
    
    #checks if accuraacy is not 0 and is greater than 0.7 for validation test set
    assert accuracy != 0
    #print(accuracy)
    assert accuracy > 0.7
