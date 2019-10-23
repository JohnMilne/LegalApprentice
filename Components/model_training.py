# The model_training function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/15/2019

# This function will train one model on one train/test/split dataset with the
# assumption that the nlp_transformer function was used to create the split.
# The model's history, weights, training and testing accuracies, learning rate
# curves, predictions and actual values are saved to a .csv file along with the
# model objects being saved to JSON and the model weights being saved to HDF5.

# The model fitting will take the current model compiled by the model_compiler
# function with the current train/test/split by the nlp_transformer and train
# a dense neural network using the passed variables as hyperparameters.
# The following is the data dictionary for the passed variables
#   labels - default is ['CitationSentence','EvidenceSentence',
#                        'FindingSentence','LegalRuleSentence',
#                        'ReasoningSentence','Sentence'], which is the labels
#       of the training data.  Only edit if the list actually changes.
#       Because the nlp_transformer returns this list, if used in a script
#       along with the nlp_transformer, labels = <nlp_transformer>[1] can be
#       passed.
#   dropout - default of 0.50, which is the decimal percentage of nodes
#       randomly 'dropped' during the current epoch of training.  This needs
#       to be the number used with the model_compiler, which, like labels above
#       is used previous to this has been returned by model_compiler and can be
#       passed as dropout = <model_compiler>[1].
#   reduction - default of 1, which is the reduction parameter of the
#       model_compiler and, as above, can be passed as redcuction = 
#       <model_compiler>[2].
#   scale - default of 1, which is the scale parameter of the model_compiler
#       and can be passed as scale = <model_compiler>[3].
#   max_words - default is 5000, which is the max_words parameter of the
#       nlp_transformer and can be passed as max_words = <nlp_transformer>[2].
#   ngrams - default of (1,1), which is the ngrams parameter of the
#       nlp_transformer and can be passed as ngram = <nlp_transformer>[4].
#   min_delta - default of 0.0001, which is the threshold for determining
#       Early Stopping.
#   patience - default of 5, which is the time to wait until Early Stopping.
#   exp_a - default of 0.001, which is coefficient in from of the ae^-bx
#       learning rate curve.
#   exp_b - default of 0.05, which is the exponent of the x variable in the
#       learning rate curve.
#   batch_size - default of 100, which is the amount of data seen in one batch
#       of training.  A batch_size of 1 is the equivalent of seeing one line
#       at a time vice the default of 100 lines at a time.
#   epochs - default of 30, which is the number of training epochs before the
#       training quits.  Anything larger gets beat by Early Stopping.
#   val_split - dafault is 0.10, which is the decimal percentage of the data
#       held out as testing data.

def model_training(labels      = ['CitationSentence','EvidenceSentence',
                                  'FindingSentence','LegalRuleSentence',
                                  'ReasoningSentence','Sentence'],
                   dropout     = 0.50,
                   reduction   = 1,
                   scale       = 1,
                   max_words   = 5000,
                   ngrams      = (1,1),
                   min_delta   = 0.0001,
                   patience    = 5,
                   exp_a       = 0.001,
                   exp_b       = 0.05,
                   batch_size  = 100,
                   epochs      = 30,
                   val_split   = 0.10):
    
    #Imports of import:
    from math import exp
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

    from keras import models
    import numpy as np
    import os
    import pandas as pd
    
    # Loading the saved model:
    model = models.load_model("./model_saves/model.h5")
    
    # Ingesting the data previously transformed by nlp_transformer().
    X_train = pd.read_pickle("./Pickles/Training/X_train.pkl")
    X_test  = pd.read_pickle("./Pickles/Testing/X_test.pkl")
    y_train = pd.read_pickle("./Pickles/Training/y_train.pkl")
    y_test  = pd.read_pickle("./Pickles/Testing/y_test.pkl")
    
    # EarlyStopping will be monitoring the accuracy and will stop the training
    # after <patience> epochs if the change in what is being monitored,
    # <'val_acc'>, only changes by <min_delta> during all of those epochs.
    early_stopping = EarlyStopping(monitor   = 'val_acc',
                                   min_delta = min_delta,
                                   patience  = patience,
                                   verbose   = 1)
    
    # LearningRateScheduler:
    
    # The first step for the LearningRateScheduler is to define the function it
    # will use - this is where the exp_* constants are used to change the 
    # behavior of the exponential decay of the learning rate.
    def learning_rate_function(epoch):
    
        # Keeping it simple with a smooth exponential decay.
        return float(exp_a*(exp(-exp_b*epoch)))
    
    # After making the learning rate scheduler function, just call it the same
    # way the other callbacks were called.
    learning_rate_reducer = LearningRateScheduler(learning_rate_function)
    
    # Fitting the model to the training data using the passed constants:
    new_model = model.fit(X_train,
                          y_train,
                          validation_split = val_split,
                          epochs           = epochs,
                          batch_size       = batch_size,
                          callbacks        = [early_stopping,
                                              learning_rate_reducer],
                          shuffle          = False,
                          verbose          = 1);
    
    # Saving the model.  This saves the weights and biases of each node as well
    # as the optimizer used when compiling the model:
    model.save(path = "./model_saves/model.h5")
    
    # Also storing the whole history of each model in a dataframe and exporting
    # to a .csv file for later imports and visualizations.
    history = pd.DataFrame(new_model.history.history)
    history.columns = ['Training Loss',
                             'Training Accuracy',
                             'Testing Loss',
                             'Testing Accuracy',
                             'Learning Rate']
    
    # These append the dropout rate, the max_words used, the n-grams used, an
    # the reduction and scale parameters of the model to a that dataframe...
    history['Dropout']   = np.ones(len(history['Testing Loss']))*dropout
    history['Max Words'] = np.ones(len(history['Testing Loss']))*max_words
    history['N-Grams']   = np.ones(len(history['Testing Loss']))*ngrams[1]
    history['Reduction'] = np.ones(len(history['Testing Loss']))*reduction
    history['Scale']     = np.ones(len(history['Testing Loss']))*scale
    
    # ...and then adds them to any existing histories file (after the below
    # check for an existing histories file).
    
    # Grabbing the current directory listing where histories will be stored:
    list_of_histories = os.listdir("./model_saves/")
    
    # Checking for the existence of a previous histories file:
    
    # If it exists, write the .csv file without a header and in append mode
    # (because the file alread exists and therefore already has its header)...
    if [s.lower for s in list_of_histories if 'model_histories.csv' in s]:
        history.to_csv(f"./model_saves/model_histories.csv",
                             mode   = 'a',
                             index  = False,
                             header = False)
        
    # ...otherwise, write the file with a header.
    else:
        history.to_csv(f"./model_saves/model_histories.csv",
                             index  = False,
                             header = True)

    # Predicting using the model on the testing data:
    predictions = new_model.predict_classes(X_test)
    
    # Mapping those numerical predictions to their labels...
    preds = pd.DataFrame([labels[pred] for pred in predictions],
                         columns = ['Predictions'])
    
    # ...and comparing them agains the real answers...
    preds['Actual'] = [labels[pred] for pred in y_test]
    
    # ...and adding all of these plus the other attributes of the model to the
    # dataframe...
    preds['Dropout']   = np.ones(len(preds['Predictions']))*dropout
    preds['Max Words'] = np.ones(len(preds['Predictions']))*max_words
    preds['N-Grams']   = np.ones(len(preds['Predictions']))*ngrams[1]
    preds['Reduction'] = np.ones(len(preds['Predictions']))*reduction
    preds['Scale']     = np.ones(len(preds['Predictions']))*scale
    
    # ...and then saving the dataframe to another .csv file - can't save to the
    # same file as the model histories because the size is all wrong.
    
    # Checking for the existence of a previous predictions file:
    
    # Grabbing the current directory listing where predictions will be stored:
    list_of_histories = os.listdir("./model_saves/")
    
    # Checking if the file name that will be saved here already exists.
    
    # If it exists, write the .csv file without a header and in append mode
    # (because the file alread exists and therefore already has its header)...
    if [s.lower for s in list_of_histories if 'preds_histories.csv' in s]:
        preds.to_csv("./model_saves/preds_histories.csv",
                     mode   = 'a',
                     index  = False,
                     header = False)
        
    # ...otherwise, write the file with a header.
    else:
        preds.to_csv("./model_saves/model_histories.csv",
                     index  = False,
                     header = True)
    
    # Now to return all of the hyperparameters of the training:    
    return min_delta, patience, exp_a, exp_b, batch_size, epochs, val_split