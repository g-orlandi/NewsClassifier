import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from .config import *
from .utils import load_data
class CSRDataset(Dataset):
    def __init__(self, X_csr: sp.csr_matrix, y: np.ndarray):
        assert sp.isspmatrix_csr(X_csr)
        self.X = X_csr
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]                 # 1 x D CSR
        y = self.y[idx]
        return row, y

def csr_collate(batch, device="cpu"):
    # batch: list of (1xD csr, label)
    rows, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    # costruisci COO per una matrice (B x D)
    data = []
    col_idx = []
    row_idx = []

    for i, r in enumerate(rows):
        r = r.tocoo()
        data.append(torch.tensor(r.data, dtype=torch.float32))
        col_idx.append(torch.tensor(r.col, dtype=torch.int64))
        row_idx.append(torch.full((r.nnz,), i, dtype=torch.int64))

    data = torch.cat(data)
    col_idx = torch.cat(col_idx)
    row_idx = torch.cat(row_idx)

    indices = torch.stack([row_idx, col_idx], dim=0)  # (2, nnz)
    B = len(rows)
    D = rows[0].shape[1]

    X = torch.sparse_coo_tensor(indices, data, size=(B, D), device=device)
    X = X.coalesce()
    return X, labels


import torch.nn as nn
import torch.nn.functional as F

class SparseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float=0.5):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(hidden, in_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden, num_classes)

    def forward(self, X_sparse):
        h = torch.sparse.mm(X_sparse, self.W1.t()) + self.b1   # dense (B,H)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.fc_out(h)
