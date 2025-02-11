import torch
from torch.utils.data import ConcatDataset, DataLoader
from utils import LSTMDataset, EarlyStopping 
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import TOPVIEWRODENTS_CONFIG
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time
from math import ceil
from torchmetrics import Precision, Recall, AveragePrecision, Accuracy


BATCH_SIZE = 512
EPOCHS = 50
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds = ConcatDataset([LSTMDataset('lstm_dataset', 'train', cl) for cl in ds_config['classes']])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_ds = ConcatDataset([LSTMDataset('lstm_dataset', 'val', cl) for cl in ds_config['classes']])
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
train_steps = ceil(len(train_ds) / BATCH_SIZE)  # number of train and val steps
val_steps = ceil(len(val_ds) / BATCH_SIZE)

print(f"Total samples in train dataset: {len(train_ds)}")
print(f"Total samples in val dataset: {len(val_ds)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
model = LSTMClassifier(542, 128, len(ds_config['classes'])).to(device)
#model = MultiLayerBiLSTMClassifier(2078, 256, 2, len(ds_config['classes'])).to(device)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimizer, gamma=0.9)
train_loss = nn.CrossEntropyLoss()
precision = Precision(num_classes=len(ds_config['classes']), task='multiclass').to(device)
recall = Recall(num_classes=len(ds_config['classes']), task='multiclass').to(device)
ap = AveragePrecision(num_classes=len(ds_config['classes']), task='multiclass').to(device)
acc = Accuracy(num_classes=len(ds_config['classes']), task='multiclass').to(device)

es = EarlyStopping(5)
best_val_ap = 0


for epoch in range(EPOCHS):
    model.train()  # set model to training mode
    total_loss = 0  # init total losses and metrics
    total_precision = 0
    total_recall = 0
    total_ap = 0
    total_acc = 0
    with tqdm(train_dl, unit='batch') as tepoch:
        for X_train, y_train in tepoch:
            free, total = torch.cuda.mem_get_info(0)
            mem_used_MB = int((total - free) / 1024 ** 2)
            tepoch.set_description(f"GPU usage {mem_used_MB} MB EPOCH {epoch + 1}/{EPOCHS} TRAINING")
            X_train, y_train = X_train.to(device), y_train.to(device)  # get data
            y_pred = model(X_train)  # get predictions
            
            loss = train_loss(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()  # back propogation
            optimizer.step()  # optimizer's step
            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
            time.sleep(0.1)
    total_loss = total_loss / train_steps
    print(f"Train loss: {total_loss:.4f}\n")
    scheduler.step()  # apply lr decay
    with torch.no_grad():
        model.eval()
        with tqdm(val_dl, unit='batch') as vepoch:
            for X_val, y_val in vepoch:
                vepoch.set_description(f'EPOCH {epoch + 1}/{EPOCHS} VALIDATING')
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                y_pred = model(X_val)  # get predictions
                total_precision += precision(y_pred, y_val)
                total_recall += recall(y_pred, y_val)
                total_ap += ap(y_pred, y_val)
                total_acc += acc(y_pred, y_val)
                time.sleep(0.1)

    print(f"""Val Precision: {total_precision / val_steps:.4f} Recall: {total_recall / val_steps:.4f}
Accuracy {total_acc / val_steps:.4f} mAP: {total_ap / val_steps:.4f}\n""")
    if total_ap > best_val_ap:  # save best weights
        best_val_ap = total_ap
        torch.save(model.state_dict(), "best_lstm.pt")
    if es.step(total_ap):  # check early stopping
        print(f'Activating early stopping callback at epoch {epoch}')
        break
    

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassPrecision, MulticlassRecall

ma = MulticlassAccuracy(num_classes=len(ds_config['classes']), average=None).to(device)
m_ap = MulticlassAUPRC(num_classes=len(ds_config['classes']), average=None).to(device)
mp = MulticlassPrecision(num_classes=len(ds_config['classes']), average=None).to(device)
mr = MulticlassRecall(num_classes=len(ds_config['classes']), average=None).to(device)

model.load_state_dict(torch.load('best_lstm.pt'))

with torch.no_grad():
    model.eval()
    with tqdm(val_dl, unit='batch') as vepoch:
        for X_val, y_val in vepoch:
            vepoch.set_description("Final eval")
            X_val, y_val = X_val.to(device), y_val.to(device)  # get data
            y_pred = model(X_val)  # get predictions
            ma.update(y_pred, y_val)
            m_ap.update(y_pred, y_val)
            mp.update(y_pred, y_val)
            mr.update(y_pred, y_val)
            time.sleep(0.1)

acc = ma.compute()
ap = m_ap.compute()
prec = mp.compute()
rec = mr.compute()
[print(f"Accuracy for class {ds_config['classes'][i]} {acc[i]}") for i in range(len(ds_config['classes']))]
print(f"Mean accuracy {acc.mean()}")
[print(f"Average Precision for class {ds_config['classes'][i]} {ap[i]}") for i in range(len(ds_config['classes']))]
print(f"Mean Average Precision {ap.mean()}")
[print(f"Precision for class {ds_config['classes'][i]} {prec[i]}") for i in range(len(ds_config['classes']))]
print(f"Mean precision {prec.mean()}")
[print(f"Recall for class {ds_config['classes'][i]} {rec[i]}") for i in range(len(ds_config['classes']))]
print(f"Mean recall {rec.mean()}")