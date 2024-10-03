import pickle

# write a function to save date in pickle format
def save_data_as_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

# write a function to load data from pickle format
def load_data_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

