# The model_predicting function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/15/2019

# This function assumes that a working model has been saved to a local
# /model_saves/ folder in JSON format and that the weights of that model have
# also been saved to that folder in HDF5 format.  It also assumese that the
# training data has been previously created and the results of the training
# will be located in the directory /Pickles.  It loads the model and the data
# then predicts using said model against said data; thus, the expectation is
# that the data is well-formed testing data named (X_test) with answers in
# (y_test).  The function then returns the models predictions paired with the
# answers in y_test to a list of lists as well as writes the model's object
# to disk.

def model_predictor():
    
    # Imports of import
    from keras.models import load_model
    import json
    import pandas as pd
    
    # Loading the model to predict with:
    model = load_model("./model_saves/mode.json)
    
    # Loading the saved weights from the HDF5 file into the model:
    loaded_model.load_weights(f"model.h5")
    
    ### Predict against the input data
    
    # Decompose the data:
    (X_test, y_test) = (data[0],data[1])
    
    # Compile the model again:
    loaded_model.compile(loss      = 'categorical_crossentropy',
                         optimizer = 'adam',
                         metric    = ['accuracy'])
    
    # Predict against the testing data using this set of weights on the model:  
    predictions = loaded_model.predict_classes(X_test)
    
    # Check the accuracy of the predictions:
    # Create the labels list first...
    labels = ['CitationSentence','EvidenceSentence','FindingSentence',
              'LegalRuleSentence','ReasoningSentence','Sentence']
    
    # ...and create the revised predictions list using the labels:
    predictions = pd.DataFrame([labels[pred] for pred in predictions],
                               columns = ['Predictions'])
    
    # Check against the answers:
    
    # First we have to find out what the format of the y_test data is in
    # because it could be a 1-hot-encoded list of lists with length 6 list
    # embedded lists, or a single integer entry that corresponds to the labels
    # list, or an actual label string.  Thus, checking to see if it's a single
    # list of a list of lists will do the job of separating the possibilities.
    answers = []
    
    # Using try/except on the second shape entry will reveal if the y_test
    # shape is a 1-hot-encoded-length-6-list entry or a single integer entry
    # corresponding to a given label, or an actual label.
    try:
        y_test.shape[1]
        
        # If y_test.shape[1] is true, then it exists; therefore, it's a list of
        # lists.  That 1-hot-encoded list needs to be transformed back into an
        # answer using the correct label. The logic of the list comprehension
        # is take the first length-6 list in y_test (pred), go through the list
        # and see if the entry equals 1, if so, find the index of that entry
        # and grab the label associated with that index in the labels list.
        answers = [label[pred.idx(x)] for pred in y_test for x in pred if x == 1]

    # If y_test.shape[1] is false, then it's either an integer or a string in
    # y_test[0]
    except:
        
        # If it's a string entry, then it's a direct answer, append that to the
        # answers list.
        if type(y_test[0]) == 'str':
            answers.append(y_test)
            
        # If it's an integer, then it's the index of the label list that holds
        # the answer, so grab that answer and add it to the answers list.
        elif type(y_test[0]) == 'int':
            answers = [labels[prediction] for prediction in y_test]
        
        # If none of the above work, the data is not in any expected format,
        # so call that out
        else:
            print('Answers list not in a useful format.')
            
    # Now that the answers are such that they align with the way the
    # predictions will be output, the predictions can be checked against the
    # answers to determine how well the predictor worked.
    
    # Making a new list of lists where each entry is a list with the answer as
    # the first entry and the prediction as the second entry
    avp = list(zip(answers,predictions))
    
    # Checking if the two entries in each list are equivalent and appending
    # those answers to the new list check.
    check = []
    check = [check.append(1) if x[0] == x[1] else check.append(0) for x in avp]
    
    # Because the check for each prediction has been performed, simple math
    # on that list can produce the accuracy statistic.
    print(f"The total accuracy for this model is {check.sum()/len(check):2.3}")
    
    # Returning the model's object such that the model.history information is
    # retrievable later for use in future calculations
    return loaded_model, avp
    
    
            
    
    