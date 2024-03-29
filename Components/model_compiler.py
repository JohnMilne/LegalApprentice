# The model_compiler function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/15/2019

# This compiles the neural network model layers.  This compiler does not do any
# training on training data or any predicting on test data.  It just creates
# the model using the following passed parameters:
#   max_words - default of 5000
#   dropout   - default of 0.50
#   labels    - default of ['CitationSentence','EvidenceSentence',
#                           'FindingSentence','LegalRuleSentence',
#                           'ReasoningSentence','Sentence'],
#   reduction - default of 1
#   scale     - default of 1

# The meaning is those constants is as follows:
#   max_words is the maximum number of stored features used by the model.  This
#       is also the number passed by the nlp_transformer function that should
#       have been used to create the data; so, assuming this is correct, then
#       this can be passed as max_words = <nlp_transformer>[2]
#   dropout is the regularization rate of the model - use the default unless
#       specifically testing new models.
#   labels is the list of labels that are the answers for which type of
#       sentence the model is going to label each sentence.  Labels is also
#       the output of the nlp_transformer function, which if called previously
#       can be passed as labels = <nlp_transformer>[1]
#   reduce is a scaler used within the model to scale the layers' nodes
#       compared to the input layer - use default unless testing again.
#   scale is another scaler for the model - again use default unless testing.

### Note: When the production version of this is created, these should become
#   hard-coded constants associated with the best performing model within the
#   framework of the function rather than as passed variables as this function
#   is currently built.  They are currently only set up as passed variables for
#   ease of use when testing out different models.

def model_compiler(max_words = 5000,
                   dropout   = 0.50,
                   labels    = ['CitationSentence','EvidenceSentence',
                                'FindingSentence','LegalRuleSentence',
                                'ReasoningSentence','Sentence'],
                   reduce    = 1,
                   scale     = 1):
    
    # Imports of import:
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    
    # The first step of the model creation is to instantiate the model:
    model = Sequential();

    # Adding the layers:

    # The first layer is the input layer.  It seems desirable to have the
    # number of input nodes equal to the size of the input data.  The input
    # data is sized by the max_words constant initialized at the beginning of
    # the nlp_transformer function prior to this function's use.  The default
    # for max_words is 5000, which is an extremely large input layer when going
    # with this method for creating said layer.  The activation type of 'relu'
    # is the current data science best practice for activating nodes in a
    # standard dense neural network.
    model.add(Dense(max_words,
                    input_shape = (max_words,),
                    activation  = 'relu'
                    name        = f'Input Layer: ({max_words},)'))

    # All subsequent layers between the input layer and the final output layer
    # are the hidden layers.  Deep Learning's pedantic meaning is that for deep
    # learning to be happening, at least one hidden layer must exist.
    # Experience has shown that multiple hidden layers add to the magic - and
    # the word magic means it is much more art than science on how the shape
    # of the input layer versus the deep layers and the output layer affect
    # the actual training of the model.  But, more deep layers typically brings
    # about better performance against the accuracy metric commonly used with
    # these standard dense neural networks.  Thus, this model will use multiple
    # deep layers and part of the tweaking of the model during testing will be
    # to change the size of the deep layers, so constants will be used
    # associated with calculatable attributes from the data rather than
    # hard-coding the number of nodes per layer.  The reduction constant allows
    # for scaling of the first and fourth hidden layers by <reduction>.  The
    # scale constant allows for further scaling of the second and third hidden
    # layers.  Thus, if reduction and scale are both unity, then the hidden
    # layers are the same size as the input layer and the shape is square.  If
    # reduction is less than unity, then the hidden layers grow in size and if
    # scale is greater than unity, those secondary hidden layers also grow in
    # size.  This allows for the aforementioned square shape, a pear shape or
    # a saddle shape in terms of the shape of the whole set of hidden layers
    # when using the above constants.
    model.add(Dense(int(max_words/reduce),
                    activation = 'relu'
                    name       = f'Hidden Layer 1: {max_words/reduce}'))

    # Overfitting refers to the condition where a model trains well against the
    # training data, but doesn't actually understand the data well enough to
    # perform well on new (testing) data.  This is referred to as variance,
    # while the actual errors involved are referred to as bias. There is always
    # a bias/variance trade-off which is exactly analogous to the Heisenberg
    # Uncertainty Principle in physics.  Because of this, using dropout right
    # from the start when building a neural network is typical because neural
    # networks tend towards overfitting.  Dropout refers to the decimal
    # percentage of the nodes in the previous layer that are randomly turned
    # off during the current epoch of training while the neural network
    # advances through its training epochs.  This necessarily introduces bias
    # into the neural network, which trades off of the extra variance and
    # thereby increases the performance of the neural network on testing data
    # through the bias/variance trade-off (magic!).  This is referred to as
    # regularization and is another parameter that can be tweaked to improve
    # the neural network's overall performance.

    # TL;DR: dropout is a decimal percentage and increases performance on
    # testing data, which is the performance metric in use with this model.
    model.add(Dropout(dropout,
                      name = f'Dropout 1 with dropout = {dropout}'))

    # This is the second real hidden layer because the dropout layer is really
    # a process vice a layer. The number of nodes are scaled by both the
    # <scale> constant and the <reduction> constant, whereas, the previous
    # hidden layer had only been scaled by the <reduction> constant.  If the
    # <scale> constant equals 1, then the shape is square.  If the <scale>
    # constant is positive, then the shape is pyramidal.  Lastly, if the
    # <scale> constant is negative then we have a saddle shape.
    model.add(Dense(int(max_words*scale/reduce),
                    activation = 'relu'
                    name       = f'Hidden Layer 2: {max_words*scale/reduce}'))

    # Every hidden layer will have dropout applied to it.  The normal use of
    # this is to preclude a particular node from becoming overactivated by an
    # outlier in the data and becoming the driver of the output concerning that
    # outlier.
    model.add(Dropout(dropout,
                      name = f'Dropout 2 with dropout = {dropout}'))

    # This is layer number 4 or hidden layer number 3.  Same as the previous
    # layer.
    model.add(Dense(int(max_words*scale/reduction),
                    activation = 'relu',
                    name       = f'Hidden Layer 3: {max_words*scale/reduce}'))
    model.add(Dropout(dropout,
                      name = f'Dropout 3 with dropout = {dropout}'))

    # The last hidden layer is the same as the first hidden layer, which allows
    # for square, saddle and pear shapes as structures for the hidden layers.
    model.add(Dense(int(max_words/reduction),
                    activation = 'relu',
                    name       = f'Hidden Layer 4: {max_words*scale/reduce}'))
    model.add(Dropout(dropout,
                      name = f'Dropout 4 with dropout = {dropout}'))

    # The Output Layer.  The output layer has two properties that distinguish
    # it from the other layers.  The first distinction is that the number of
    # nodes is dictated by the number of classes the neural network is trying
    # to classify.  The second distinction is that the best practice in data
    # science is to use softmax for the activation of the output layer vice
    # the relu activation of the previous layers.
    model.add(Dense(len(labels),
                    activation = 'softmax',
                    name       = f'Output layer with {len(labels)} nodes'))

    # That's it for the building of the neural network's layers.  The next step
    # compiles them into the model object using the parameters given:
    # categorical_crossentropy is the loss type for any neural network
    # classification that is not a simple binary classification; 'adam' is the
    # current best practice for the optimizers in neural networks - other
    # options are SGD, RMSprop, Adagrad, Adadelta...  We are also most
    # concerned about the accuracy (vice the recall or the specificity) of our
    # model; thus, the metric being trained against is accuracy.
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics   = ['accuracy'])
    
    # Saving the model.  This saves the weights and biases of each node as well
    # as the optimizer used when compiling the model:
    model.save(path = "./model_saves/model.h5")
    
    # This gargantuan model will produce a very interesting summary; thus, the
    # need to print out its summary as part of the output of the function:
    model.summary() 
    
    # Creating the return variable:
    model_created_and_saved = 'Done'
    
    # And now to return the hyperparameters of the model:
    return model_created_and_saved, dropout, reduction, scale