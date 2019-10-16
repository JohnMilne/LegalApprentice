# This is the model visualization step of the Legal Apprentice workflow,
# developed by John Milne on 10/16/2019

# This function will take the latest saved model(s) in the model_save folder,
# the dataset to predict against and the labels list for each classification;
# it will printout a set of metrics for that model's training and testing
# scores.  The function will then return the model for later consumption.
# The data should be in the form (X_test, y_test) and the labels should be an
# alphabetized list of the labeled classes.

def model_visualizations(model,
                         data,
                         labels):
    
    # Imports of import
    from sklearn.metrics import classification_report
    from sklearn.metrics import multilabel_confusion_matrix
    
    # Grab the predictions from the model.
    predictions = model.predict_classes(X_test)

    # We can print out a classification report for our predictions to see the
    # breakdown for each class:
    print("\nClassification Report for the the model:\n")
    print(classification_report(y_test,
                                predictions,
                                target_names = labels))
    
    # We can also print out a table of the actual versus predicted for each
    # class:
    print("\nCross Tabulation for all of the Sentence Types\n")
    print(pd.crosstab(y_test,
                      labels[],
                      rownames = ['Actual'],
                      colnames = ['Predicted'],
                      margins  = True).to_string())
    
    # Load previously saved histories for visualization.
    models = pd.read_csv("./model_saves/model_histories.csv",
                            index_col = False)
    
    # Create the beginning and ending indices for the models in the history
    # file for use when grabbing entries from that space.
    beginning = int(models['Model #'].unique()[0])
    ending    = int(models['Model #'].unique()[-1])
    
    # Closing any possible open plots.
    plt.close();
    
    # Creating the subplot axes for training and testing accuracies as well as
    # showing the learning rate curve.
    fig, (ax1, ax2, ax3) = plt.subplots(3,
                                        1,
                                        sharex  = True,
                                        sharey  = False,
                                        figsize = (12,12));
        
    # For-loop to range through the model numbers and add them to the plot. The
    # +1 after the ending variable is to deal with the 0-indexing in python.
    
    # This for-loop produces the multiplot for the training accuracies...
    for i in range(beginning,
                   ending + 1):
        ax1.plot(np.arange(1,(1 + len(models['Model #'][models['Model #'] == i]))),
                 models['Training Accuracy'][models['Model #'] == i],
                 linewidth = 3,
                 label     = f'Model{int(i)}');
    ax1.set_title(f'Models {beginning}-{ending} Training Accuracies');
    ax1.set_xlabel('Epochs');
    ax1.set_ylabel('Accuracy (%)');
    ax1.legend();
    
    # ...and this for-loop produces the multiplot for the testing accuracies...
    for j in range(beginning,
                   ending + 1):
        ax2.plot(np.arange(1,(1 + len(models['Model #'][models['Model #'] == i]))),
                 models['Testing Accuracy'][models['Model #'] == i],
                 linewidth = 3,
                 label     = f'Model{int(i)}');
    ax2.set_title(f'Models {beginning}-{ending} Testing Accuracies');
    ax2.set_xlabel('Epochs');
    ax2.set_ylabel('Accuracy (%)');
    ax2.legend();
    
    # ...and this for-loop produces the multiplot of the learning rate curves.    
    for i in range(beginning,
                   ending + 1):
        ax3.plot(np.arange(1,(1 + len(models['Model #'][models['Model #'] == i]))),
                 cv_models['Learning Rate'][models['Model #'] == i],
                 linewidth = 3,
                 label     = f'Model{int(i)}');
    ax3.set_title(f'Models {beginning}-{ending} Testing Accuracies');
    ax3.set_xlabel('Epochs');
    ax3.set_ylabel('Accuracy (%)');
    ax3.legend();
    
    # Now to return the model:
    return model