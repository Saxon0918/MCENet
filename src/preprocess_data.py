import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef
import sys

sys.path.append("..")


class ADNI_Dataset:
    def __init__(self, random_seed=1, mode='train', group=0, split_ratio=(0.7, 0.1, 0.2)):
        self.mode = mode
        self.random_seed = random_seed
        mri = pd.read_csv('data/MRI.csv')  # MRI
        av45 = pd.read_csv('data/AV45.csv')  # AV45-PET
        fdg = pd.read_csv('data/FDG.csv')  # FDG-PET
        gene = pd.read_csv('data/SNP.csv')  # SNP Gene data
        alldata = pd.read_csv('data/all_data.csv')  # All data
        same_id = av45.merge(mri, on='IID').merge(gene, on='IID').merge(fdg, on='IID').merge(alldata, on='IID')[['IID']]
        same_data = alldata.merge(same_id, on='IID')
        if group == 0:
            # ------------AD vs. CN------------
            subset = same_data[same_data['diagnosis'].isin(['AD', 'CN']) | same_data['diagnosis'].isna()]
            subset['label'] = subset['diagnosis'].map({'AD': 1, 'CN': 0, np.nan: -1}).fillna(-1).astype(int)
        elif group == 1:
            # ------------AD vs. MCI------------
            subset = same_data[same_data['diagnosis'].isin(['AD', 'MCI']) | same_data['diagnosis'].isna()]
            subset['label'] = subset['diagnosis'].map({'AD': 1, 'MCI': 0, np.nan: -1}).fillna(-1).astype(int)
        elif group == 2:
            # ------------MCI vs. CN------------
            subset = same_data[same_data['diagnosis'].isin(['MCI', 'CN']) | same_data['diagnosis'].isna()]
            subset['label'] = subset['diagnosis'].map({'MCI': 1, 'CN': 0, np.nan: -1}).fillna(-1).astype(int)
        else:
            print("unknown groups")

        same_mri = mri[mri['IID'].isin(subset['IID'])]
        same_av45 = av45[av45['IID'].isin(subset['IID'])]
        same_fdg = fdg[fdg['IID'].isin(subset['IID'])]
        same_gene = gene[gene['IID'].isin(subset['IID'])]
        full_data = same_mri.merge(same_av45, on='IID', suffixes=('_mri', '_av45'))
        full_data = full_data.merge(same_fdg, on='IID', suffixes=('', '_fdg'))
        full_data = full_data.merge(same_gene, on='IID', suffixes=('', '_gene'))
        full_data = full_data.merge(subset[['IID', 'label']], on='IID')
        same_mri = full_data.iloc[:, 1:141]
        same_av45 = full_data.iloc[:, 141:281]
        same_fdg = full_data.iloc[:, 281:421]
        same_gene = full_data.iloc[:, 421:521]
        labels = full_data.iloc[:, 521].to_numpy().reshape(-1, 1)
        mri = same_mri.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        av45 = same_av45.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        fdg = same_fdg.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        gene = same_gene.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()

        mri_train, mri_temp, av45_train, av45_temp, fdg_train, fdg_temp, gene_train, gene_temp, y_train, y_temp = \
            train_test_split(mri, av45, fdg, gene, labels, test_size=(1 - split_ratio[0]),
                             random_state=self.random_seed)
        val_size = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        mri_val, mri_test, av45_val, av45_test, fdg_val, fdg_test, gene_val, gene_test, y_val, y_test = \
            train_test_split(mri_temp, av45_temp, fdg_temp, gene_temp, y_temp, test_size=(1 - val_size),
                             random_state=self.random_seed)

        if self.mode == 'train':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_train, av45_train, fdg_train, gene_train, y_train
        elif self.mode == 'val':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_val, av45_val, fdg_val, gene_val, y_val
        elif self.mode == 'test':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_test, av45_test, fdg_test, gene_test, y_test
        else:
            raise ValueError("invalid data")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'mri': torch.tensor(self.mri[idx], dtype=torch.float32),
            'av45': torch.tensor(self.av45[idx], dtype=torch.float32),
            'fdg': torch.tensor(self.fdg[idx], dtype=torch.float32),
            'gene': torch.tensor(self.gene[idx], dtype=torch.float32),
            'label': torch.tensor(self.label[idx], dtype=torch.float32)
        }


class ADNI_three_Dataset:
    def __init__(self, random_seed=1, mode='train', group=0, split_ratio=(0.7, 0.1, 0.2)):
        self.mode = mode
        self.random_seed = random_seed
        mri = pd.read_csv('data/MRI.csv')  # MRI
        av45 = pd.read_csv('data/AV45.csv')  # AV45-PET
        fdg = pd.read_csv('data/FDG.csv')  # FDG-PET
        gene = pd.read_csv('data/SNP.csv')  # SNP Gene data
        alldata = pd.read_csv('data/all_data.csv')  # All data
        same_id = av45.merge(mri, on='IID').merge(gene, on='IID').merge(fdg, on='IID').merge(alldata, on='IID')[['IID']]
        same_data = alldata.merge(same_id, on='IID')

        # ------------three class: AD vs. MCI vs. CN------------
        subset = same_data[same_data['diagnosis'].isin(['AD', 'MCI', 'CN']) | same_data['diagnosis'].isna()]
        subset['label'] = subset['diagnosis'].map({'CN': 0, 'AD': 1, 'MCI': 2, np.nan: -1}).fillna(-1).astype(int)

        same_mri = mri[mri['IID'].isin(subset['IID'])]
        same_av45 = av45[av45['IID'].isin(subset['IID'])]
        same_fdg = fdg[fdg['IID'].isin(subset['IID'])]
        same_gene = gene[gene['IID'].isin(subset['IID'])]

        full_data = same_mri.merge(same_av45, on='IID', suffixes=('_mri', '_av45'))
        full_data = full_data.merge(same_fdg, on='IID', suffixes=('', '_fdg'))
        full_data = full_data.merge(same_gene, on='IID', suffixes=('', '_gene'))
        full_data = full_data.merge(subset[['IID', 'label']], on='IID')

        same_mri = full_data.iloc[:, 1:141]
        same_av45 = full_data.iloc[:, 141:281]
        same_fdg = full_data.iloc[:, 281:421]
        same_gene = full_data.iloc[:, 421:521]

        labels = full_data.iloc[:, 521].to_numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        onehot_labels = encoder.fit_transform(labels)
        labels = onehot_labels

        mri = same_mri.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        av45 = same_av45.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        fdg = same_fdg.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()
        gene = same_gene.apply(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()

        mri_train, mri_temp, av45_train, av45_temp, fdg_train, fdg_temp, gene_train, gene_temp, y_train, y_temp = \
            train_test_split(mri, av45, fdg, gene, labels, test_size=(1 - split_ratio[0]),
                             random_state=self.random_seed)
        val_size = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        mri_val, mri_test, av45_val, av45_test, fdg_val, fdg_test, gene_val, gene_test, y_val, y_test = \
            train_test_split(mri_temp, av45_temp, fdg_temp, gene_temp, y_temp, test_size=(1 - val_size),
                             random_state=self.random_seed)

        if self.mode == 'train':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_train, av45_train, fdg_train, gene_train, y_train
        elif self.mode == 'val':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_val, av45_val, fdg_val, gene_val, y_val
        elif self.mode == 'test':
            self.mri, self.av45, self.fdg, self.gene, self.label = mri_test, av45_test, fdg_test, gene_test, y_test
        else:
            raise ValueError("invalid data")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'mri': torch.tensor(self.mri[idx], dtype=torch.float32),
            'av45': torch.tensor(self.av45[idx], dtype=torch.float32),
            'fdg': torch.tensor(self.fdg[idx], dtype=torch.float32),
            'gene': torch.tensor(self.gene[idx], dtype=torch.float32),
            'label': torch.tensor(self.label[idx], dtype=torch.float32)
        }
