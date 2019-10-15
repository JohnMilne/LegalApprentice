
    ### Model Fitting/Training

    # We have one model to train on two datasets - the CountVectorized data and the TF-IDF transformed data.  This is
    # accomplished by saving the model to a new variable for each fitting/training session.  Each model's history, weights
    # training and testing accuracies, learning rate curves, predictions and actual values are saved to file for later
    # retrieval to plot them all against each other and/or create confusion matrices.

    # Initializing the callback constants:
    min_delta = 0.00005
    patience  = 8

    # EarlyStopping will be monitoring the accuracy and will stop the training after <patience> epochs if the change in
    # what is being monitored only changes by <min_delta> during all of those epochs.
    early_stopping = EarlyStopping(monitor   = 'val_acc',
                                   min_delta = min_delta,
                                   patience  = patience,
                                   verbose   = 1)

    # LearningRateScheduler:

    # First step for the LearningRateScheduler is to define the function it will use:
    def learning_rate_function(epoch):

        # Keeping it simple with a smooth exponential decay.
        return float(0.001*(math.exp(-0.05*epoch)))

    # After making the learning rate scheduler function, just call it the same way the other callbacks were called.
    learning_rate_reducer = LearningRateScheduler(learning_rate_function)

    # Initializing the constants being used in the model compilation step.  Specifically, we are using a batch size of
    # 100, we are training for 100 epochs (or until early stopping happens...) and our validation split is 10% because we
    # have are low on data at the moment.
    batch_size = 100
    epochs     = 100
    val_split  = 0.10

    # Now we fit the model to the training data.  We'll do this twice, once for the CountVectorizered dataset and again
    # for the TF-IDF transformation on the dataset.

    # CountVectorized data first:
    print("\nTraining the CountVectorizer model\n")
    cv_model = model.fit(X_train_cv,
                         y_train_1_hot,
                         validation_split = val_split,
                         epochs           = epochs,
                         batch_size       = batch_size,
                         callbacks        = [early_stopping, learning_rate_reducer],
                         shuffle          = False,
                         verbose          = 1);

    # We need to calculate the model number based on the existing save files.  We will be writing 4 save files per
    # training session, so we need this calculation to make an accurate model number iteration.  I am also saving the four
    # history files in this directory, so the math below is the correct math to get the correct model number each time
    # (currently!).
    new_model_number = int(((len(os.listdir('./model_saves/')) - 4)/4) + 1) + 1

    # Now the actual saving to file part:
    # Serialize the cv_model to JSON.
    cv_model_json = model.to_json()
    with open(f"./model_saves/cv_model{new_model_number}.json", "w") as json_file:
        json_file.write(cv_model_json)

    # Serialize the cv_model weights to HDF5.
    model.save_weights(f"./model_saves/cv_model{new_model_number}.h5")
    print(f"\nSaved cv_model{new_model_number} with dropout {dropout} at 'C:\Documents\Python Scripts\model_saves'")
    
    # Saving the testing and training accuracies to a variable for printing to the screen at the end of the training
    # sessions.
    cv_train_accuracy = model.history.history['acc'][len(model.history.history['acc']) - 1]*100
    cv_test_accuracy  = model.history.history['val_acc'][len(model.history.history['val_acc']) - 1]*100

    # Also storing the whole history of each model in a dataframe and exporting to a .csv file for later imports and
    # visualizations.
    cv_model_history = pd.DataFrame(model.history.history)
    cv_model_history.columns = ['Training Loss','Training Accuracy','Testing Loss','Testing Accuracy','Learning Rate']

    # These append the model number, the dropout rate, the max_words used and the max n-grams used to the .csv so I can
    # plot these quantities on my plots if I decide I need the model parameters in later analyses...
    cv_model_history['Model #']   = np.ones(len(cv_model_history['Testing Loss']))*new_model_number
    cv_model_history['Dropout']   = np.ones(len(cv_model_history['Testing Loss']))*dropout
    cv_model_history['Max Words'] = np.ones(len(cv_model_history['Testing Loss']))*max_words
    cv_model_history['N-Grams']   = np.ones(len(cv_model_history['Testing Loss']))*ngrams[1]
    cv_model_history['Reduction'] = np.ones(len(cv_model_history['Testing Loss']))*reduction
    cv_model_history['Scale']     = np.ones(len(cv_model_history['Testing Loss']))*scale
    
    # ...and I wanted to append a description of the shape of the model as well; thus, the structure variable will be a
    # string that will be a (hopefully) brief description of the models' layer structures...
    structure = f'{max_words} input layer, reduction {reduction} on next layers with {scale} adjustment to third layer+'
    shape = []
    [shape.append(structure) for _ in range(len(cv_model_history['Testing Loss']))]
    cv_model_history['Shape'] = shape

    # ...and then adds them to the existing histories file.  The if statement makes sure any new history files get
    # created with a header while further additions to the file do not repeat the header row addition, which breaks
    # things!
    if new_model_number == 1:
        cv_model_history.to_csv(f"./model_saves/cv_model_histories.csv",
                                index  = False,
                                header = True)
    else:
        cv_model_history.to_csv(f"./model_saves/cv_model_histories.csv",
                                mode   = 'a',
                                index  = False,
                                header = False)
    print(f"Saved cv_model{new_model_number} to ~/model_saves/cv_model_histories.csv\n")
    
    # We also need to do our predictions before the next fit and store those as well:
    cv_predictions_per_class = model.predict_classes(X_test_cv)
    
    # Now to map those numerical predictions to their labels...
    cv_model_predictions = pd.DataFrame([labels[prediction] for prediction in cv_predictions_per_class],
                                        columns = ['Predictions'])
    
    # ...and compare them agains the real answers...
    cv_model_predictions['Actual']    = y_test
    
    # ...and add all of the other attributes of the model I may want to know about afterwards...
    cv_model_predictions['Model #']   = np.ones(len(cv_model_predictions['Predictions']))*new_model_number
    cv_model_predictions['Dropout']   = np.ones(len(cv_model_predictions['Predictions']))*dropout
    cv_model_predictions['Max Words'] = np.ones(len(cv_model_predictions['Predictions']))*max_words
    cv_model_predictions['N-Grams']   = np.ones(len(cv_model_predictions['Predictions']))*ngrams[1]
    cv_model_predictions['Reduction'] = np.ones(len(cv_model_predictions['Predictions']))*reduction
    cv_model_predictions['Scale']     = np.ones(len(cv_model_predictions['Predictions']))*scale
    
    # ...and then save those to file - can't save to the same file as the model histories because the size is all wrong.
    cv_model_predictions.to_csv(f"./model_saves/cv_prediction_histories.csv")
    
    # Okay, that's all of the file writing in order to save the things stored in model.history.history before we overwrite
    # that with a new model!
    
    # Now for the TF-IDF transformed data:
    print("Training the TF-IDF model\n")
    tfidf_model = model.fit(X_train_tf,
                            y_train_1_hot,
                            validation_split = val_split,
                            epochs           = epochs,
                            batch_size       = batch_size,
                            callbacks        = [early_stopping, learning_rate_reducer],
                            shuffle          = False,
                            verbose          = 1);

    # Writing this model to file as well.
    # Serialize the tf_model to JSON.
    tf_model_json = model.to_json()
    with open(f"./model_saves/tf_model{new_model_number}.json", "w") as json_file:
        json_file.write(tf_model_json)

    # Serialize the tf_model weights to HDF5.
    model.save_weights(f"./model_saves/tf_model{new_model_number}.h5")
    print(f"\nSaved tf_model{new_model_number} with dropout {dropout} at 'C:\Documents\Python Scripts\model_saves'")

    # Again saving the training and testing accuracies to variable for printing to screen...
    tf_train_accuracy = model.history.history['acc'][len(model.history.history['acc']) - 1]*100
    tf_test_accuracy  = model.history.history['val_acc'][len(model.history.history['val_acc']) - 1]*100

    # ...and saving the model histories to a .csv for later comparison (same situation as before)
    tf_model_history = pd.DataFrame(model.history.history)
    tf_model_history.columns = ['Training Loss','Training Accuracy','Testing Loss','Testing Accuracy','Learning Rate']
    tf_model_history['Model #']   = np.ones(len(tf_model_history['Testing Loss']))*new_model_number
    tf_model_history['Dropout']   = np.ones(len(tf_model_history['Testing Loss']))*dropout
    tf_model_history['Max Words'] = np.ones(len(tf_model_history['Testing Loss']))*max_words
    tf_model_history['N-Grams']   = np.ones(len(tf_model_history['Testing Loss']))*ngrams[1]
    tf_model_history['Reduction'] = np.ones(len(tf_model_history['Testing Loss']))*reduction
    tf_model_history['Scale']     = np.ones(len(tf_model_history['Testing Loss']))*scale
    shape = []
    [shape.append(structure) for _ in range(len(tf_model_history['Testing Loss']))]
    tf_model_history['Shape'] = shape
    if new_model_number == 1:
        tf_model_history.to_csv(f"./model_saves/tf_model_histories.csv",
                                index  = False,
                                header = True)
    else:
        tf_model_history.to_csv(f"./model_saves/tf_model_histories.csv",
                                mode = 'a',
                                index  = False,
                                header = False)
    print(f"Saved tf_model{new_model_number} to ~/model_saves/tf_model_histories.csv")

    # Same as above with the predictions being written to file as well.
    tf_predictions_per_class   = model.predict_classes(X_test_cv)
    tf_model_predictions = pd.DataFrame([labels[prediction] for prediction in tf_predictions_per_class],
                                        columns = ['Predictions'])
    tf_model_predictions['Actual']    = y_test
    tf_model_predictions['Model #']   = np.ones(len(tf_model_predictions['Predictions']))*new_model_number
    tf_model_predictions['Dropout']   = np.ones(len(tf_model_predictions['Predictions']))*dropout
    tf_model_predictions['Max Words'] = np.ones(len(tf_model_predictions['Predictions']))*max_words
    tf_model_predictions['N-Grams']   = np.ones(len(tf_model_predictions['Predictions']))*ngrams[1]
    tf_model_predictions['Reduction'] = np.ones(len(tf_model_predictions['Predictions']))*reduction
    tf_model_predictions['Scale']     = np.ones(len(tf_model_predictions['Predictions']))*scale
    tf_model_predictions.to_csv(f"./model_saves/tf_prediction_histories.csv")
    
    # Ta Da! That trained up a bunch of CountVectorizer and TF-IDF models with <shape> nodes and varying dropout based
    # on the for-loop characteristics.