import torch
from torch.utils.data import DataLoader
import time
from typing import Tuple, Dict
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from .dataloader import TwoStreamDataset



class TwoStreamValidator:
    def __init__(self, config: Dict, models: Dict[str, torch.nn.Module], 
                 dataset: TwoStreamDataset) -> None:
        self.config = config
        self.val_ds = dataset
        self.n_classes = len(config['classes'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
        self.models = models
        for model in self.models.values():
            model.to(self.device)
            model.eval()

    def __prepare_data(self, batch: int) -> None:
        self.val_dl = DataLoader(self.val_ds, batch_size=batch * 2, shuffle=False, num_workers=8)

    def validate(self, batch: int) -> None:
        self.__prepare_data(batch)
        metrics = {key: [MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device),
                         MulticlassAUPRC(num_classes=self.n_classes, average=None).to(self.device),
                         MulticlassPrecision(num_classes=self.n_classes, average=None).to(self.device),
                         MulticlassRecall(num_classes=self.n_classes, average=None).to(self.device)]
                         for key in ('rgb', 'flow', 'two_stream')}

        with torch.no_grad():
            with tqdm(self.val_dl, unit='batch') as vepoch:
                for X_val, y_val in vepoch:
                    vepoch.set_description("Final eval")
                    y_val = y_val.to(self.device)  # get data
                    X_rgb = X_val[0].to(self.device)
                    X_flow = X_val[1].to(self.device)
                    # RGB model
                    y_pred = self.models['rgb'](X_rgb)
                    for m in metrics['rgb']:
                        m.update(y_pred, y_val)
                    # Flow model
                    y_pred = self.models['flow'](X_flow)
                    for m in metrics['flow']:
                        m.update(y_pred, y_val)
                    # Two Stream model
                    y_pred = self.models['two_stream']((X_rgb, X_flow))
                    for m in metrics['two_stream']:
                        m.update(y_pred, y_val)
                    time.sleep(0.1)

        for key in metrics.keys():
            acc = metrics[key][0].compute()
            ap = metrics[key][1].compute()
            prec = metrics[key][2].compute()
            rec = metrics[key][3].compute()
            [print(f"Accuracy for model {key} for class {self.config['classes'][i]} {acc[i]}") for i in range(len(self.config['classes']))]
            print(f"Mean accuracy for model {key} {acc.mean()}")
            [print(f"Average Precision for model {key} for class {self.config['classes'][i]} {ap[i]}") for i in range(len(self.config['classes']))]
            print(f"Mean Average Precision for model {key} {ap.mean()}")
            [print(f"Precision for model {key} for class {self.config['classes'][i]} {prec[i]}") for i in range(len(self.config['classes']))]
            print(f"Mean precision for model {key} {prec.mean()}")
            [print(f"Recall for model {key} for class {self.config['classes'][i]} {rec[i]}") for i in range(len(self.config['classes']))]
            print(f"Mean recall for model {key} {rec.mean()}\n")
