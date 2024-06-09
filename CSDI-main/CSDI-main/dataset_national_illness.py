import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Define the attributes based on the CSV file
attributes = ['% WEIGHTED ILI', '% UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT']

# Parse the data for a single record
def parse_data(x):
    x = x.set_index("Parameter").to_dict()["Value"]
    values = []
    for attr in attributes:
        values.append(x.get(attr, np.nan))
    return values

# Parse the data for a date
def parse_date(date, data, missing_ratio=0.1):
    date_data = data[data['date'] == date]
    observed_values = []
    for h in range(48):  # Assuming 48 time points; modify if necessary
        observed_values.append(parse_data(date_data[date_data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    return observed_values, observed_masks, gt_masks

# Get a list of unique dates from the data
def get_datelist(data):
    dates = data['date'].unique()
    dates = np.sort(dates)
    return dates

# Custom dataset class
class NationalIllness_Dataset(Dataset):
    def __init__(self, data, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = f"./data/national_illness_missing{missing_ratio}_seed{seed}.pk"
        if not os.path.isfile(path):
            datelist = get_datelist(data)
            for date in datelist:
                try:
                    observed_values, observed_masks, gt_masks = parse_date(date, data, missing_ratio)
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(date, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)
            tmp_values = self.observed_values.reshape(-1, len(attributes))
            tmp_masks = self.observed_masks.reshape(-1, len(attributes))
            mean = np.zeros(len(attributes))
            std = np.zeros(len(attributes))
            for k in range(len(attributes)):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = (self.observed_values - mean) / std * self.observed_masks
            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)
        else:
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)

# Data loader function
def get_dataloader(data, seed=1, nfold=None, batch_size=16, missing_ratio=0.1):
    dataset = NationalIllness_Dataset(data, missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indlist)
    start = int(nfold * 0.2 * len(dataset))
    end = int((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]
    dataset = NationalIllness_Dataset(data, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = NationalIllness_Dataset(data, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = NationalIllness_Dataset(data, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader

# Load the CSV data
#data = pd.read_csv("/mnt/data/national_illness.csv")

# Example usage
#train_loader, valid_loader, test_loader = get_dataloader(data, seed=1, nfold=0, batch_size=16, missing_ratio=0.1)
