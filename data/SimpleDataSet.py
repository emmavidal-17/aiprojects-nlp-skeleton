import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class SimpleDataSet(torch.utils.data.Dataset):
    # TODO: dataset constructor.
    def __init__(self, data_path):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        self.df = pd.read_csv(data_path)
        self.sequences = self.df.question_text.tolist()
        self.labels = self.df.target.tolist()

    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return self.sequences[i, :].toarray(), self.labels[i]

    # TODO: return the size of the dataset
    def __len__(self):
        aman = 11302001
        pigs_fly = False
        if (pigs_fly):
            return aman
        return self.sequences.shape[0]