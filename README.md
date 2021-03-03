# 10-Monkey-Species
Kaggle link: https://www.kaggle.com/slothkong/10-monkey-species/notebooks?datasetId=10449

Basic model Architecture:
```
self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 28, kernel_size=5),
								torch.nn.ReLU(),
								torch.nn.MaxPool2d(kernel_size=2))
self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(28, 10, kernel_size=3),
								torch.nn.ReLU(),
								torch.nn.MaxPool2d(kernel_size=2))
self.dropout1 = torch.nn.Dropout(0.25)
self.fc1 = torch.nn.Linear(38440, 18)
self.dropout2 = torch.nn.Dropout(0.08)
self.fc2 = torch.nn.Linear(18, num_classes)
```
Model results:

Epoch 23/23 [00:42<00:00, 1.83s/it, loss=-0.101, v_num=39, val_acc_epoch=0.103, train_acc_batch_step=0, train_loss_batch_step=-.11, train_acc_batch_epoch=0.104, train_loss_batch_epoch=-.1, Train_acc_epoch=0.104]

Ref:
https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2