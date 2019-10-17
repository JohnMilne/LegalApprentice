# This pickle creator function was developed for the Legal Apprentice workflow
# by John Milne, 10/17/2019

# This data file creator takes the JSON-formatted data in /Data that the Legal
# Apprentice data starts as, throws it into a dataframe and then pickles that
# to the /Pickle directory.

# The assumption here is that the data stored at "~/.Data/" hasn't previously
# been pickled and the running of this function is to create the pickle file
# of the JSON-formatted NEW data, which is then moved to the /Pickle directory;
# thus, the "./Data/" directory only contains that  data which needs to be
# pickled and nothing else and any data in the /Pickle directory has either
# been moved or renamed so as not to overwrite any previous work (that should
# not be overwritten).

def legal_apprentice_pickler():
    
    # Imports of import.
    import json
    import os
    import pandas as pd
    
    # Getting the list of files in <data_path>:
    list_of_files = os.listdir(os.get_cwd() + './Data/')
    
    # Loading each file into a dataframe - adopted from Stephen Strong's code:
    
    # Using a for-loop to iterate over the filenames
    for filename in list_of_files:
        
        # Opening the given filename...
        file = open(filename)
        
        # ...using the json file loader to translate the json data...
        data = json.load(file)
        
        # ...parsing the json for the relevant pieces to add in...
        sentences = []
        rhetroles = []
        for sent in data['sentences']:
            
            # ... creating the 'Sentences'...
            sentences.append(sent['text'])
            
            # ...and the 'RhetoricalRoles' columns
            rhetroles.append(sent['rhetRole'][0])
            
    # Adding everything in the lists above to the dataframe to be pickled.
    df = pd.DataFrame({'Sentences'     : sentences,
                       'RhetoricalRole': rhetroles})
    
    # Pickling the dataframe:
    df.to_pickle("./PickledData/50Cases.pkl")
    
    # Now to pass the fact tht this has completed as the return statement:
    pickled = 'done'
    
    return pickled
    