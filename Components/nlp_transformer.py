# The nlp_transformer function, written by John Milne 10/15/2019

# This function takes the data from the Legal Apprentice workflow as an input.
# One of the assumptions is that the data will be a dataframe that has at least
# two columns which are labeled as Sentences and RhetoricalRoles respectively.

# This function will transform the data using both CountVectorizer and TF-IDF
# transformers and will return a large number of objects at the conclusion of
# those transformations.  The information about those return objects is in the
# comments above the return statement.

def nlp_transformer(df):
    
    # Necessary imports:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.multiclass import unique_labels
    from tensorflow.keras.utils import to_categorical
    
    # Using train_test_split to do the sorting into training and testing
    # datasets.  The random_state flag allows for reproducability across
    # implementations, only using a 10% testing split due to a low amount of
    # data and need to set the shuffle flag to false so the order that the
    # predictions exist in the df.Sentences dataframe will be the order that
    # the list of answers in df.RhetoricalRoles will occur.
    X_train, X_test, y_train, y_test = train_test_split(df.Sentences,
                                                        df.RhetoricalRoles,
                                                        random_state = 42,
                                                        test_size    = 0.1,
                                                        shuffle      = False)
    
    # Ensuring the labeling is unique:
    labels = unique_labels(y_test)
    
    ### Preprocessing the data using NLP methods
    
    # We use both a word count vectorizing method and a TF-IDF ((t)erm
    # (f)requency - (i)nverse (d)ocument (f)requency) method for turning words
    # into vectors.  We are specifically using both methods to compare against
    # each other performance-wise.
    
    ### Count Vectorizer first!
    
    # Initializing constants for both the CountVectorizer and the TF-IDF
    # transformers for ease with tweaking the transformers during modeling.
    # The max_words constant is both the maximum words the transformer will
    # hold after transforming the data as well as the basis for the number of
    # nodes in the layers of the neural network during later parts of the
    # workflow.  The ngrams constant is the range of n-grams that the
    # transformers will use.  Typical is just an n-gram of 1 which for both
    # transformers look at each word individually against all of the other
    # words in the corpus.  N-grams larger than 1 indicate that the transformer
    # will use that many words that occur next to each other together to make a
    # phrase and add all of those comparisons to the features list.  Because
    # that list necessarily gets quite large, the max_words constant becomes
    # a necessity to keep the features list to a reasonable size.
    max_words = 10000
    ngrams    = (1,3)
    
    # Instantiating the CountVectorizer object that will hold all of the
    # information about the transformations done to the dataset by the
    # CountVectorizer - the reason the whole object is returned by the function
    count_vec = CountVectorizer(ngram_range  = ngrams,
                                max_features = max_words)
    
    # The actual fit/transform on the training data...
    X_train_cv = count_vec.fit_transform(X_train)
    
    # ...but only transforming the testing data.  Otherwise, the testing data
    # gets included with the training data!
    X_test_cv  = count_vec.transform(X_test)
    
    # Turning the labels on the training data into one-hot-encoded vectors that
    # the neural network will understand.
    
    # First step is to use Sci-Kit Learn's labelEncoder to turn the text labels
    # into integers:
    
    # Initializing the LabelEncoder:
    encoder = LabelEncoder()
    
    # The LabelEncoder is fit to the y_train labels...
    encoder.fit(y_train)
    
    # ...and then used to transform the y_* series of labels.  Again, no
    # fitting on the test data.
    y_train_encode = encoder.transform(y_train)
    y_test_encode  = encoder.transform(y_test)
    
    # The second step is to one-hot-encode those integer-based vectors using
    # Sci-Kit Learn's to_categorical function.
    y_train_1_hot = to_categorical(y_train_encode)
    y_test_1_hot  = to_categorical(y_test_encode)
    
    ### Now for TF-IDF!
    
    # Using the same constants for TF-IDF that were used for CountVectorizer;
    # thus, no need for new constants to be initialized here!
    tfidf_vec = TfidfVectorizer(ngram_range  = ngrams,
                                max_features = max_words)
    
    # Doing the fit_transform on the data using TF-IDF in the same way that
    # CountVectorizer was used to transform the data above:
    X_train_tf = tfidf_vec.fit_transform(X_train)
    X_test_tf  = tfidf_vec.transform(X_test)
    
    # The labels have already been transformed using the
    # LabelEncoder/to_categorical process above, so no need to repeat that
    # process - we're done with the data transformations!
    
    # Now for the long list of return objects this function gives:
    #   X_train_cv - the training data transformed by the CountVectorizer
    #   X_test_cv  - the testing data transformed by the CountVectorizer
    #   X_train_tf - the training data transformed by TD-IDF
    #   X_test_cv  - the testing data transformed by TF-IDF
    #   y_train_1_hot - the labels for the training data 1-hot-encoded
    #   y_test_1_hot  - the labels for the testing data 1-hot-encoded
    #   labels    - the list of unique labels amongst the entries of *_1_hot
    #   max_words - the number of words the model has been limited to
    #   ngrams    - the number of n-grams the transformers went out to
    #   count_vec - the CountVectorizer object and its attributes
    #   tfidf_vec - the TF-IDF Vectorizer object and its attributes
    return (X_train_cv, X_test_cv, X_train_tf, X_test_tf, y_train_1_hot,
            y_test_1_hot, labels, max_words, ngrams, count_vec, tfidf_vec)