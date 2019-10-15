# This is the model training step of the Legal Apprentice workflow, developed
# by John Milne on 10/15/2019

# This function will train one model on one train/test/split dataset.  The
# model's history, weights, training and testing accuracies, learning rate
# curves, predictions and actual values are saved to a .csv file along with the
# model objects being saved to JSON and the model weights being saved to HDF5.

### Model Fitting/Training

# This model fitting and training is expecting to have the compiled model
# passed to it along with the pre-split training and testing data passed in
# along with any constants that differ from their defaults listd below:
#   model_label - default of cv, which is a label to append to the model's
#       filename.
#   labels - default is ['CitationSentence','EvidenceSentence',
#                        'FindingSentence','LegalRuleSentence',
#                        'ReasoningSentence','Sentence'], which is the labels
#       of the training data.  Only edit if the list actually changes.
#   dropout - default of 0.50, which is the decimal percentage of nodes
#       randomly 'dropped' during the current epoch of training.  This needs
#       to be the number used with the model_compiler() function call previous
#       to the use of this function call.
#   reduction - default of 1, which is the reduction parameter of the
#       model_compiler() and should be the number used there.
#   scale - default of 1, which is the scale parameter of the model_compiler()
#       and should be the number used there.
#   max_words - default is 5000, which is the max_words parameter of the
#       model_compiler() and should be the number used there.
#   ngrams - default of (1,1), which is the ngrams parameter of the
#       model_compiler() and should be the number used there.
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
def model_training(model,
                   X_train,
                   y_train_1_hot,
                   X_test,
                   y_test_1_hot,
                   model_label = 'cv',
                   labels      = ['CitationSentence','EvidenceSentence',
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
    
    import numpy as np
    import os
    import pandas as pd
    
    # EarlyStopping will be monitoring the accuracy and will stop the training
    # after <patience> epochs if the change in what is being monitored,
    # <'val_acc'>, only changes by <min_delta> during all of those epochs.
    early_stopping = EarlyStopping(monitor   = 'val_acc',
                                   min_delta = min_delta,
                                   patience  = patience,
                                   verbose   = 1)
    
    # LearningRateScheduler:
    
    # First step for the LearningRateScheduler is to define the function it
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
                          y_train_1_hot,
                          validation_split = val_split,
                          epochs           = epochs,
                          batch_size       = batch_size,
                          callbacks        = [early_stopping,
                                              learning_rate_reducer],
                          shuffle          = False,
                          verbose          = 1);
    
    # Calculating the model number based on the existing save files.  This
    # function will be writing two new save files per function call so this
    # calculation need to take that into account in order to make an accurate
    # model number iteration.  There also exist 4 static files as the baseline
    # number of files before any new files are written.
    new_model_number = int(((len(os.listdir('./model_saves/')) - 4)/4) + 1)
    
    # The actual saving to model to JSON part:
    model_json = new_model.to_json()
    with open(f"./model_saves/{model_label}model{new_model_number}.json", "w") as json_file:
        json_file.write(model_json)
    
    # Saving the model weights to HDF5.
    new_model.save_weights(f"./model_saves/{model_label}model{new_model_number}.h5")
    print(f"\nSaved {model_label}model{new_model_number} to 'C:\Documents\Python Scripts\model_saves'")
    
    # Also storing the whole history of each model in a dataframe and exporting
    # to a .csv file for later imports and visualizations.
    model_history = pd.DataFrame(model.history.history)
    model_history.columns = ['Training Loss','Training Accuracy','Testing Loss','Testing Accuracy','Learning Rate']
    
    # These append the model number, the dropout rate, the max_words used, the
    # n-grams used, and the reduction and scale parameters of the model to a
    # .csv file...
    model_history['Model #']   = np.ones(len(model_history['Testing Loss']))*new_model_number
    model_history['Dropout']   = np.ones(len(model_history['Testing Loss']))*dropout
    model_history['Max Words'] = np.ones(len(model_history['Testing Loss']))*max_words
    model_history['N-Grams']   = np.ones(len(model_history['Testing Loss']))*ngrams[1]
    model_history['Reduction'] = np.ones(len(model_history['Testing Loss']))*reduction
    model_history['Scale']     = np.ones(len(model_history['Testing Loss']))*scale
    
    # ... while this structure string is an attempt to capture the shape of
    # the model's layers here as well...
    structure = f'{max_words} input layer, reduction {reduction} on next layers with {scale} adjustment to third layer+'
    shape = []
    [shape.append(structure) for _ in range(len(model_history['Testing Loss']))]
    model_history['Shape'] = shape
    
    # ...and then adds them to any existing histories file.  The if statement
    # makes sure any new history files get created with a header while further
    # additions to the file do not repeat the header row addition.
    if new_model_number == 1:
        model_history.to_csv(f"./model_saves/model_histories.csv",
                             index  = False,
                             header = True)
    else:
        model_history.to_csv(f"./model_saves/model_histories.csv",
                             mode   = 'a',
                             index  = False,
                             header = False)
    print(f"Saved {model_label}model{new_model_number} to ~/model_saves/model_histories.csv\n")
    
    # Now to do our predictions store those in the model object as well:
    predictions_per_class = model.predict_classes(X_test)
    
    # Mapping those numerical predictions to their labels...
    model_predictions = pd.DataFrame([labels[prediction] for prediction in predictions_per_class],
                                     columns = ['Predictions'])
    
    # ...and comparing them agains the real answers...
    model_predictions['Actual'] = [labels[prediction] for prediction in y_test_1_hot]
    
    # ...and adding all of the other attributes of the model to a different
    # .csv file because the lengths of the sections are very different.
    model_predictions['Model #']   = np.ones(len(model_predictions['Predictions']))*new_model_number
    model_predictions['Dropout']   = np.ones(len(model_predictions['Predictions']))*dropout
    model_predictions['Max Words'] = np.ones(len(model_predictions['Predictions']))*max_words
    model_predictions['N-Grams']   = np.ones(len(model_predictions['Predictions']))*ngrams[1]
    model_predictions['Reduction'] = np.ones(len(model_predictions['Predictions']))*reduction
    model_predictions['Scale']     = np.ones(len(model_predictions['Predictions']))*scale
    
    # ...and then save those to file - can't save to the same file as the model histories because the size is all wrong.
    model_predictions.to_csv(f"./model_saves/prediction_histories.csv")
    print(f"Saved {model_label}model{new_model_number} to ~/model_saves/prediction_histories.csv")
    
    # Okay, that's all of the file writing in order to save the things stored in model.history.history before we overwrite
    # that with a new model!