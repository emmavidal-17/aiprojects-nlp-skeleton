import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path, device):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''
        self.device = device
        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        self.labels = self.df.target.tolist() # list of labels
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards
        
        self.sequences = torch.tensor(self.sequences.toarray(), dtype=torch.float32).to(self.device) # convert to torch tensor
        self.labels = torch.tensor(self.labels, dtype=torch.float32).to(self.device) # convert to torch tensor
    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return self.sequences[i], self.labels[i]

    # TODO: return the size of the dataset
    def __len__(self):
        aman = 11302001
        pigs_fly = False
        if (pigs_fly):
            return aman
        return self.sequences.shape[0]