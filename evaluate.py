import torch
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy, MulticlassConfusionMatrix

def evaluate(best_model, test_dataloader, device, num_classes=4):
    metrics = {"f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
               "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
               "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
               "recall_micro": MulticlassRecall(num_classes=num_classes, average="micro"),
               "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
               "precision_micro": MulticlassPrecision(num_classes=num_classes, average="micro"),
               "accuracy": MulticlassAccuracy(num_classes=num_classes),
               "conf_mat": MulticlassConfusionMatrix(num_classes=num_classes)
               }

    best_model.eval()
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_hat = best_model(X.float())
    
        for metric in metrics:
            metrics[metric].update(y_hat, y)
        
    for metric in metrics:
        if metric == "conf_mat":
            metrics[metric] = metrics[metric].compute().long()
        else:
            metrics[metric] = metrics[metric].compute()

    return metrics