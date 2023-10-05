import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from os.path import join, split
from sklearn.model_selection import train_test_split
from joblib import dump, load
import scipy.sparse as sp

torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)


def split_data(root, data_file="IMDB_Dataset.csv"):

    data = pd.read_csv(join(root, "Data", data_file)).to_numpy()
    train_data, valid_test_data = train_test_split(data, train_size=0.8, test_size=0.2)
    valid_data, test_data = train_test_split(valid_test_data, train_size=0.6, test_size=0.4)

    train_dict = {"review": list(train_data[:, 0]), "sentiment": list(train_data[:, 1])}
    train_df = pd.DataFrame(train_dict)
    train_df.to_csv(join(root, "Data", "train_data_annotation.csv"), index=False)

    valid_dict = {"review": list(valid_data[:, 0]), "sentiment": list(valid_data[:, 1])}
    valid_df = pd.DataFrame(valid_dict)
    valid_df.to_csv(join(root, "Data", "valid_data_annotation.csv"), index=False)

    test_dict = {"review": list(test_data[:, 0]), "sentiment": list(test_data[:, 1])}
    test_df = pd.DataFrame(test_dict)
    test_df.to_csv(join(root, "Data", "test_data_annotation.csv"), index=False)


class TextCustomDataset(Dataset):

    def __init__(self, root, data_file, train_mode=True):
        super(TextCustomDataset, self).__init__()

        vectorizer=None
        if not train_mode:
            vectorizer = load("vectorizer.joblib")

        self.cv = vectorizer if not train_mode else CountVectorizer()
        self.data = pd.read_csv(join(root, data_file))

        if train_mode:
            self.B_O_W = self.cv.fit_transform(self.data["review"].to_list())
        if not train_mode:
            self.B_O_W = self.cv.transform(self.data["review"].to_list())

        # save the vectorizer
        if train_mode:
            dump(self.cv, "vectorizer.joblib")

        self.classes_name = self.data["sentiment"].unique().tolist()
        self.class2idx = {cls: i for i, cls in enumerate(self.classes_name)}

        self.token2idx = self.cv.vocabulary_

    def __len__(self):
        return len(self.data)

    # def load_data(self, idx):
    #     return

    def __getitem__(self, idx):

        bow, target = self.B_O_W[idx], self.data["sentiment"].to_list()[idx]

        return torch.FloatTensor(bow.toarray()), self.class2idx[target]


def data_preparation(root, batch_size):

    split_data(root=root)

    train_dataset = TextCustomDataset(root=join(root, "Data"), data_file="train_data_annotation.csv")
    # the same vectorizer must be used in train, valid and test evaluations
    valid_dataset = TextCustomDataset(root=join(root, "Data"), data_file="valid_data_annotation.csv", train_mode=False)
    classes_name = train_dataset.classes_name

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, classes_name


if __name__ == "__main__":

    root = r"D:\mreza\TestProjects\Python\NLP\BOW_Classifier\Data"
    data_file = "IMDB_Dataset.csv"
    data = pd.read_csv(join(root, data_file))
    txt = data["review"]
    label = data["sentiment"]

    dataset_inference = TextCustomDataset(root, data_file)

    train_dataloader, valid_dataloader, classes_name = data_preparation(root=split(root)[0], batch_size=4)
    inp, label = next(iter(train_dataloader))
