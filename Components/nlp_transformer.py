# The nlp_transformer function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/15/2019

# This function takes as an input the data from the Legal Apprentice workflow
# that is assumed to have been saved to "~/Pickles/" as the currently known
# pickle file name of 50Cases.pkl.

# Another assumption is that the data will be a dataframe that has at least
# two columns which are labeled as Sentences and RhetoricalRoles respectively.

# This function will do a train/test/split on the data, then transform the
# data with the Keras' Tokenizer transformer using one of the available
# <modes> (default is count) of transformation and pickle those
# split-transformed datasets (X_train, X_test, y_train, y_test) into 4 pickle
# files holding all 4 transformed training and testing datasets.

# This is an NLP transformer process.  The passed variables are the
# hyperparameters of the NLP transformer.  The max_words variable determines
# the maximum number of words to keep within the transformer, the ngrams tuple
# gives the minimum and maximum (respectively) of the number of consecutive
# words to pay attention to and the mode refers to the type of NLP transformer
# being used.  The current set of modes available in Tokenizer are 'binary',
# 'count', 'freq' and 'tf-idf'.

# The function will return the list of unique labels if such are needed
# further along in the workflow.

def nlp_transformer(max_words = 5000,
                    mode      = 'count',
                    ngrams    = (1,3)):
    
    # Necessary imports:
    from sklearn.model_selection  import train_test_split
    from sklearn.preprocessing    import LabelEncoder
    from sklearn.utils.multiclass import unique_labels
    from keras.preprocessing.text import Tokenizer
    from keras.utils              import to_categorical
    
    import pandas as pd
    import pickle
    
    # Ingesting the data from "~/Pickles/50cases.pkl"
    df = pd.read_pickle("./Pickles/50Cases.pkl")
    
    # Using train_test_split to do the sorting into training and testing
    # datasets.  The random_state flag allows for reproducability across
    # implementations, only using a 15% testing split due to a low amount of
    # data currently and there is the need to set the shuffle flag to false to
    # accomplish that reproducability.
    X_train, X_test, y_train, y_test = train_test_split(df.Sentences,
                                                        df.RhetoricalRoles,
                                                        random_state = 42,
                                                        test_size    = 0.15,
                                                        shuffle      = False)
    
    # Instantiating the Tokenizer object with the passed max_words variable.
    tokens = Tokenizer(num_words = max_words)
    
    ### The actual fit/transform on the training data.
    
    # Step #1 is to use the fit_on_text to transform the tokenizer using the
    # training data.
    tokens.fit_on_texts(X_train)
    
    # Step #2 is to use the text_to_matrix method on both the training and
    # testing data, passing mode as the NLP transform type.
    X_train_tokens = tokens.texts_to_matrix(X_train,
                                            mode = mode) 
    X_test_tokens  = tokens.texts_to_matrix(X_test,
                                            mode = mode)
    
    # Turning the labels on the training data into one-hot-encoded vectors that
    # the neural network will understand.
    
    # First step is to use Sci-Kit Learn's labelEncoder to turn the text labels
    # into integers:
    
    # Initializing the LabelEncoder:
    encoder = LabelEncoder()
    
    # The LabelEncoder is fit to the y_train labels...
    encoder.fit(y_train)
    
    # ...and then used to transform the y_* series of labels.
    y_train_encode = encoder.transform(y_train)
    y_test_encode  = encoder.transform(y_test)
    
    # The second step is to one-hot-encode those integer-based vectors using
    # Sci-Kit Learn's to_categorical function.
    y_train_1_hot = to_categorical(y_train_encode)
    y_test_1_hot  = to_categorical(y_test_encode)
    
    # Now that the train/test/split is complete, pickling the transformed
    # datasets into the respective /Training and /Testing directories:
    
    # Using try/except in order to not error out when the file already exists
    try:
        with open('./Pickles/Training/X_train.pkl','xb') as f:
            pickle.dump(X_train_tokens, f)
    except FileExistsError:
        print("the file X_train already exists - rename or move and retry.")

    try:
        with open('./Pickles/Testing/X_test.pkl','xb') as f:
            pickle.dump(X_test_tokens, f)
    except FileExistsError:
        print("The file X_test already exists - rename or move and retry.")

    try:
        with open('./Pickles/Training/y_train.pkl','xb') as f:
            pickle.dump(y_train_1_hot, f)
    except FileExistsError:
        print("The file y_train already exists - rename or move and retry.")

    try:
        with open('./Pickles/Testing/y_test.pkl','xb') as f:
            pickle.dump(y_test_1_hot, f)
    except FileExistsError:
        print("The file y_test already exists - rename or move and retry.")
        
    # Creating the unique labels list as the return item:
    labels = unique_labels(y_test)
    
    return labels

nlp_transformer(mode = 'count')