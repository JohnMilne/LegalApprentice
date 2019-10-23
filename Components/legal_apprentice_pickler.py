# This pickle creator function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/17/2019

# This function takes the JSON-formatted data in /Data that the Legal
# Apprentice data starts as, throws it into a dataframe and then pickles that
# to the /Pickle directory.

# The assumption here is that the data stored at "~/.Data/" hasn't previously
# been pickled and the running of this function is to create the pickle file
# of the JSON-formatted NEW data, which is then moved to the /Pickle directory;
# thus, the "./Data/" directory only contains that data which needs to be
# pickled and nothing else and any data in the /Pickle directory has either
# been moved or renamed so as not to overwrite any previous work (that should
# not be overwritten).

def legal_apprentice_pickler():
    
    # Imports of import.
    import json
    import os
    import pandas as pd  
        
    # Creating the dataframe into which all of the files will be stored:
    df = pd.DataFrame()
    
    # Getting the list of files in <data_path>:
    data_path = './Data/'
    list_of_files = os.listdir(data_path)
    
    # Using a for-loop to iterate over the filenames...
    for filename in list_of_files:
        
        # ... and opening the given filename...
        file = open(data_path + filename)
        
        # ...using the json file loader to translate the json data...
        data = json.load(file)
        
        # ...and creating new lists for the texts of the sentences...
        df_sents = []
        df_rhets = []
        
        # ...and adding the sentences to those new lists...
        for sent in data['sentences']:
            
            # ...creating the 'Sentences'...
            df_sents.append(sent['text'])
            
            # ...and the 'RhetoricalRoles' columns...
            df_rhets.append(sent['rhetRole'][0])
            
    # ...and adding those to the previously instantiated dataframe:
    df['Sentences']       = df_sents
    df['RhetoricalRoles'] = df_rhets
                
    # Pickling the dataframe:
    df.to_pickle("./Pickles/50Cases.pkl")
    
    # Now to pass the fact that this has completed as the return statement:
    pickled = 'Done'
    
    return pickled

legal_apprentice_pickler()