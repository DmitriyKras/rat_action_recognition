import cv2
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time
from math import ceil
from torchmetrics import Precision, Recall, AveragePrecision, Accuracy
from utils import build_conv3d_dataset, EarlyStopping
from configs import TOPVIEWRODENTS_CONFIG
from models import resnet3d
from torchsummary import summary


BATCH_SIZE = 8
EPOCHS = 10
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds, val_ds = build_conv3d_dataset(ds_config, 
                                        input_shape=(256, 256), 
                                        w_size=32, 
                                        overlap=0, 
                                        offset=10)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
train_steps = ceil(len(train_ds) / BATCH_SIZE)  # number of train and val steps
val_steps = ceil(len(val_ds) / BATCH_SIZE)

print(f"Total samples in train dataset: {len(train_ds)}")
print(f"Total samples in val dataset: {len(val_ds)}")


# for b in train_dl:
#     print(b[0].size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
model = resnet3d('resnet18', n_classes=len(ds_config['classes'])).to(device)
#summary(model, (3, 32, 256, 256))

optimizer = Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimizer, gamma=0.9)
train_loss = nn.CrossEntropyLoss()
precision = Precision(num_classes=len(ds_config['classes']), task='multiclass', average='macro').to(device)
recall = Recall(num_classes=len(ds_config['classes']), task='multiclass', average='macro').to(device)
ap = AveragePrecision(num_classes=len(ds_config['classes']), task='multiclass', average='macro').to(device)
acc = Accuracy(num_classes=len(ds_config['classes']), task='multiclass', average='micro').to(device)
es = EarlyStopping(5)
best_val_ap = 0


for epoch in range(EPOCHS):
    model.train()  # set model to training mode
    total_loss = 0  # init total losses and metrics
    precision.reset()
    recall.reset()
    ap.reset()
    acc.reset()
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
                precision.update(y_pred, y_val)
                recall.update(y_pred, y_val)
                ap.update(y_pred, y_val)
                acc.update(y_pred, y_val)
                time.sleep(0.1)

    total_ap = ap.compute()
    print(f"""Val Precision: {precision.compute():.4f} Recall: {recall.compute():.4f}
Accuracy {acc.compute():.4f} mAP: {total_ap:.4f}\n""")
    if total_ap > best_val_ap:  # save best weights
        best_val_ap = total_ap
        torch.save(model.state_dict(), "best_3dconv.pt")
    if es.step(total_ap):  # check early stopping
        print(f'Activating early stopping callback at epoch {epoch}')
        break

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassPrecision, MulticlassRecall

ma = MulticlassAccuracy(num_classes=len(ds_config['classes']), average=None).to(device)
m_ap = MulticlassAUPRC(num_classes=len(ds_config['classes']), average=None).to(device)
mp = MulticlassPrecision(num_classes=len(ds_config['classes']), average=None).to(device)
mr = MulticlassRecall(num_classes=len(ds_config['classes']), average=None).to(device)

model.load_state_dict(torch.load('best_3dconv.pt'))

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