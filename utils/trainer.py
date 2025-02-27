import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time
from math import ceil
from torchmetrics import Precision, Recall, AveragePrecision, Accuracy
from .callbacks import EarlyStopping 
from typing import Tuple, Dict
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassPrecision, MulticlassRecall


class ClassificationTrainer:
    def __init__(self, config: Dict, model: torch.nn.Module, 
                 datasets: Tuple[Dataset, Dataset], name: str = 'lstm') -> None:
        # Unpack attributes
        self.config = config
        self.train_ds, self.val_ds = datasets
        self.name = name
        self.n_classes = len(config['classes'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
        self.model = model.to(self.device)

    def __prepare_data(self, batch: int) -> None:
        self.train_dl = DataLoader(self.train_ds, batch_size=batch, shuffle=True, num_workers=4)
        self.val_dl = DataLoader(self.val_ds, batch_size=batch * 2, shuffle=True, num_workers=2)

    def __prepare_for_train(self, lr: float) -> None:
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.train_loss = nn.CrossEntropyLoss()
        self.precision = Precision(num_classes=self.n_classes, task='multiclass', average='macro').to(self.device)
        self.recall = Recall(num_classes=self.n_classes, task='multiclass', average='macro').to(self.device)
        self.ap = AveragePrecision(num_classes=self.n_classes, task='multiclass', average='macro').to(self.device)
        self.acc = Accuracy(num_classes=self.n_classes, task='multiclass', average='micro').to(self.device)
        self.es = EarlyStopping(5)

    def save_weights(self, path: str = './') -> None:
        torch.save(self.model.state_dict(), f"{path}/best_{self.name}.pt")

    def validate(self, path: str = './') -> None:
        ma = MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device)
        m_ap = MulticlassAUPRC(num_classes=self.n_classes, average=None).to(self.device)
        mp = MulticlassPrecision(num_classes=self.n_classes, average=None).to(self.device)
        mr = MulticlassRecall(num_classes=self.n_classes, average=None).to(self.device)
        self.model.load_state_dict(torch.load(f"{path}/best_{self.name}.pt"))

        with torch.no_grad():
            self.model.eval()
            with tqdm(self.val_dl, unit='batch') as vepoch:
                for X_val, y_val in vepoch:
                    vepoch.set_description("Final eval")
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)  # get data
                    y_pred = self.model(X_val)  # get predictions
                    ma.update(y_pred, y_val)
                    m_ap.update(y_pred, y_val)
                    mp.update(y_pred, y_val)
                    mr.update(y_pred, y_val)
                    time.sleep(0.1)

        acc = ma.compute()
        ap = m_ap.compute()
        prec = mp.compute()
        rec = mr.compute()
        [print(f"Accuracy for class {self.config['classes'][i]} {acc[i]}") for i in range(len(self.config['classes']))]
        print(f"Mean accuracy {acc.mean()}")
        [print(f"Average Precision for class {self.config['classes'][i]} {ap[i]}") for i in range(len(self.config['classes']))]
        print(f"Mean Average Precision {ap.mean()}")
        [print(f"Precision for class {self.config['classes'][i]} {prec[i]}") for i in range(len(self.config['classes']))]
        print(f"Mean precision {prec.mean()}")
        [print(f"Recall for class {self.config['classes'][i]} {rec[i]}") for i in range(len(self.config['classes']))]
        print(f"Mean recall {rec.mean()}")

    def train(self, batch: int, epochs: int, lr: float = 10e-4) -> None:
        self.__prepare_data(batch)
        self.__prepare_for_train(lr)

        train_steps = ceil(len(self.train_ds) / batch)  # number of train steps

        print(f"Total samples in train dataset: {len(self.train_ds)}")
        print(f"Total samples in val dataset: {len(self.val_ds)}")

        best_val_ap = 0

        for epoch in range(epochs):
            self.model.train()  # set model to training mode
            total_loss = 0  # init total losses and metrics
            self.precision.reset()
            self.recall.reset()
            self.ap.reset()
            self.acc.reset()
            with tqdm(self.train_dl, unit='batch') as tepoch:
                for X_train, y_train in tepoch:
                    free, total = torch.cuda.mem_get_info(0)
                    mem_used_MB = int((total - free) / 1024 ** 2)
                    tepoch.set_description(f"GPU usage {mem_used_MB} MB EPOCH {epoch + 1}/{epochs} TRAINING")
                    X_train, y_train = X_train.to(self.device), y_train.to(self.device)  # get data
                    y_pred = self.model(X_train)  # get predictions
                    
                    loss = self.train_loss(y_pred, y_train)
                    self.optimizer.zero_grad()
                    loss.backward()  # back propogation
                    self.optimizer.step()  # optimizer's step
                    total_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    time.sleep(0.1)
            total_loss = total_loss / train_steps
            print(f"Train loss: {total_loss:.4f}\n")
            self.scheduler.step()  # apply lr decay
            with torch.no_grad():
                self.model.eval()
                with tqdm(self.val_dl, unit='batch') as vepoch:
                    for X_val, y_val in vepoch:
                        vepoch.set_description(f'EPOCH {epoch + 1}/{epochs} VALIDATING')
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)  # get data
                        y_pred = self.model(X_val)  # get predictions
                        self.precision.update(y_pred, y_val)
                        self.recall.update(y_pred, y_val)
                        self.ap.update(y_pred, y_val)
                        self.acc.update(y_pred, y_val)
                        time.sleep(0.1)

            total_ap = self.ap.compute()
            print(f"""Val Precision: {self.precision.compute():.4f} Recall: {self.recall.compute():.4f}
Accuracy {self.acc.compute():.4f} mAP: {total_ap:.4f}\n""")
            if total_ap > best_val_ap:  # save best weights
                best_val_ap = total_ap
                self.save_weights()
            if self.es.step(total_ap):  # check early stopping
                print(f'Activating early stopping callback at epoch {epoch}')
                break
        self.validate()
