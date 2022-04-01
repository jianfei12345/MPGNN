from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GraphConv, GatedGraphConv, GATv2Conv, SuperGATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, BatchNorm
import torchmetrics
import torch
import deepchem as dc
from torch_geometric.loader import DataLoader
import torch.nn as nn
import numpy as np
import pandas as  pd
import plotly_express as px
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

#convert GraphDataset to pygDatalist.
def GraphDataset_to_pygDatalist(dataset):
  temp = []
  for i in range(len(dataset.X)):
    a = dataset.X[i].to_pyg_graph()
    a.y = torch.from_numpy(dataset.y[i,])
    a.w = torch.from_numpy(dataset.w[i,])
    temp.append(a)
  return temp




def vis_training(training):
    a = pd.melt(training, id_vars='Epoch', value_vars=['train', 'test', 'valid'], var_name='dataset',
                value_name='AUC or MSE')
    fig = px.scatter(a, x='Epoch', y='AUC or MSE', color='dataset', range_y=[0, 1])
    fig.show(renderer='browser')





#random sampling to make classes balanced and then convert to pygDatalist
def dataset_balanced(dataset):
    label = dataset.y.flatten()
    number_0 = np.unique(label, return_counts=True)[1][0]
    number_1 = np.unique(label, return_counts=True)[1][1]
    index_0 = np.where(label == 0)[0]
    index_1 = np.where(label == 1)[0]
    if (number_0 > number_1):
        index_sampled = np.random.choice(number_0, number_1, replace=False)
        index_sampled  = index_0[index_sampled]
        index = np.concatenate((index_sampled, index_1), axis=0)
    else:
        index_sampled = np.random.choice(number_1, number_0, replace=False)
        index_sampled = index_1[index_sampled]
        index = np.concatenate((index_sampled, index_0), axis=0)

    dataset_X = dataset.X[index]
    dataset_y = dataset.y[index]
    dataset_w = dataset.w[index]

    temp = []
    for i in range(len(dataset_X)):
        a = dataset_X[i].to_pyg_graph()
        a.y = torch.from_numpy(dataset_y[i]).long()
        a.w = torch.from_numpy(dataset_w[i])
        temp.append(a)
    return temp






#random sampling to make classes balanced,输入一维numpy,label, 返回索引
def balanced_index(label):
    number_0 = np.unique(label, return_counts=True)[1][0]
    number_1 = np.unique(label, return_counts=True)[1][1]
    index_0 = np.where(label == 0)[0]
    index_1 = np.where(label == 1)[0]
    if (number_0 > number_1):
        index_sampled = np.random.choice(number_0, number_1, replace=False)
        index_sampled  = index_0[index_sampled]
        index = np.concatenate((index_sampled, index_1), axis=0)
    else:
        index_sampled = np.random.choice(number_1, number_0, replace=False)
        index_sampled = index_1[index_sampled]
        index = np.concatenate((index_sampled, index_0), axis=0)
    return index




class Graph_Class(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*12, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, edge_index)
    x = self.norm1(x)
    x = torch.relu(x)


    # 2. Readout layer
    x3 = global_mean_pool(x, batch)
    x4 = global_max_pool(x, batch)
    x5 = global_add_pool(x, batch)
    x=torch.cat((x5, x4, x3), dim=1)

    # 3. Apply a final classifier
    x = F.dropout(x, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x



class Graph_Class_ablation1(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class_ablation1, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*4, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, edge_index)
    x = self.norm1(x)
    x = torch.relu(x)


    # 2. Readout layer
    x3 = global_mean_pool(x, batch)

    # 3. Apply a final classifier
    x = F.dropout(x3, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x



class Graph_Class_ablation2(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class_ablation2, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*4, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, edge_index)
    x = self.norm1(x)
    x = torch.relu(x)


    # 2. Readout layer
    x3 = global_max_pool(x, batch)

    # 3. Apply a final classifier
    x = F.dropout(x3, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x



class Graph_Class_ablation3(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class_ablation3, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*4, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, edge_index)
    x = self.norm1(x)
    x = torch.relu(x)


    # 2. Readout layer
    x3 = global_add_pool(x, batch)

    # 3. Apply a final classifier
    x = F.dropout(x3, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x



#single GCN
class Graph_Class_ablation4(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class_ablation4, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*3, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = self.norm2(x)
    x = torch.relu(x)

    # 2. Readout layer
    x3 = global_mean_pool(x, batch)
    x4 = global_max_pool(x, batch)
    x5 = global_add_pool(x, batch)
    x=torch.cat((x5, x4, x3), dim=1)

    # 3. Apply a final classifier
    x = F.dropout(x, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x



#single GAT
class Graph_Class_ablation5(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class_ablation5, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.lin1 = Linear(hidden_size*12, output_size)


  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.gat1(x, edge_index)
    x = self.norm1(x)
    x = torch.relu(x)


    # 2. Readout layer
    x3 = global_mean_pool(x, batch)
    x4 = global_max_pool(x, batch)
    x5 = global_add_pool(x, batch)
    x=torch.cat((x5, x4, x3), dim=1)

    # 3. Apply a final classifier
    x = F.dropout(x, p=0.6, training=self.training)
    x = torch.sigmoid(self.lin1(x))
    return x




# evaluation for  binary classification
def model_evaluation(pred, target):
    evaluation = {}
    #define evaluation metrics
    metric_acc = torchmetrics.Accuracy().to(device) #输入为预测标签或概率矩阵
    metric_roc = torchmetrics.ROC(pos_label=1).to(device)  #设置target正标签,然后以不同阈值分别对pred分类正负标签，得到不同迅销矩阵，计算tpr,fpr
    metric_auroc = torchmetrics.AUROC(pos_label=1).to(device) #二分类:pos_label=1,正标签预测概率。多分类:num_class,概率矩阵
    metric_pr = torchmetrics.BinnedPrecisionRecallCurve(num_classes=1,thresholds=500).to(device) #paired PR values with common threshold(binned)
    metric_AUC = torchmetrics.AUC(reorder=True).to(device)
    metric_F1 = torchmetrics.F1(ignore_index=0).to(device)
    metric_pre = torchmetrics.Precision(ignore_index=0).to(device)
    metric_Recall = torchmetrics.Recall(ignore_index=0).to(device) #precision和recall计算的是正标签，没有设置正标签，所以每类都会计算一下
    metric_ConfusionMatrix = torchmetrics.ConfusionMatrix(num_classes=2).to(device) #列名预测值，行名标签值
    metric_averageprecision = torchmetrics.BinnedAveragePrecision(num_classes=1, thresholds=500).to(device)
    metrics_MCC = torchmetrics.MatthewsCorrcoef(num_classes=2).to(device)

    # model evaluation
    acc = metric_acc(pred, target)
    auroc = metric_auroc(pred[:,1], target)
    precision = metric_pre(pred, target)
    F1 = metric_F1(pred, target)
    Recall = metric_Recall(pred, target)
    roc = metric_roc(pred[:,1], target)
    pr = metric_pr(pred[:,1], target)
    auprc = metric_averageprecision(pred[: ,1], target)
    confusionMatrix = metric_ConfusionMatrix(pred, target)
    MCC = metrics_MCC(pred[:,1], target)

    evaluation['accuracy'] = acc
    evaluation['auroc'] = auroc
    evaluation['precision'] = precision
    evaluation['F1'] = F1
    evaluation['Recall'] = Recall
    evaluation['roc'] = roc
    evaluation['pr'] = pr
    evaluation['auprc'] = auprc
    evaluation['confusionMatrix'] = confusionMatrix
    evaluation['MCC'] = MCC

    return evaluation





def evaluation_regression(pred, label):
    result = {}
    R2_Metrics = torchmetrics.R2Score().to(device)
    result['r2'] = R2_Metrics(pred, label)
    MAE_Metrics = torchmetrics.MeanAbsoluteError().to(device)
    result['mae'] = MAE_Metrics(pred, label)
    MSE_Metrics = torchmetrics.MeanSquaredError().to(device)
    result['mse'] = MSE_Metrics(pred, label)
    result['rmse'] = MSE_Metrics(pred, label).to(device)**(0.5)
    MAPE_Metrics = torchmetrics.MeanAbsolutePercentageError().to(device)
    result['mape'] = MAPE_Metrics(pred, label)
    EV_Metrics = torchmetrics.ExplainedVariance().to(device)
    result['ev'] = EV_Metrics(pred, label)
    Pearson_Metrics = torchmetrics.regression.PearsonCorrCoef().to(device)
    result['pearson'] = Pearson_Metrics(pred, label)
    return result





def train(model, mode, train_loader, test_loader, valid_loader, process=True, epoch=100, learning_rate=0.001):
    train_record = pd.DataFrame(np.zeros((epoch, 4)), columns=['Epoch','train', 'test', 'valid'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    test_losses = []
    if (mode == 'classification'):
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()
    if (process == True):
        print(mode, '\tAUC or MSE:')
    #early_stopping = EarlyStopping(patience=50, verbose=True)
    for epoch in range(epoch):
        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)   #batch: 这个batch大图里面，每个节点对应的图，用于池化层聚合节点信息，属于同一batch小图进行池化
            loss = criterion(out.view(-1), data.y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

       # model.eval()
       # for data in test_loader:
       #     data = data.to(device)
       #     out = model(data.x, data.edge_index, data.batch)
       #     loss = criterion(out.view(-1), data.y.float())
       #     test_losses.append(loss.item())
       # test_loss = np.average(test_losses)
        #early_stopping(test_loss, model)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break

        if (process == True):
            if (mode == 'classification'):
                keys = 'auroc'
            else:
                keys = 'mse'
            a = test(model, train_loader, mode=mode)[keys].item()
            b = test(model, test_loader, mode=mode)[keys].item()
            c = test(model, valid_loader, mode=mode)[keys].item()
            train_record.iloc[epoch, 0] =epoch+1
            train_record.iloc[epoch, 1] = a
            train_record.iloc[epoch, 2] = b
            train_record.iloc[epoch, 3] = c
            print(f'Epoch:{epoch:03d}\t train={a:.4f}, test={b:.4f}, valid={c:.4f}')
    #model.load_state_dict(torch.load('checkpoint.pt'))
    return train_record.round(4)


      #  print('epoch:',str(epoch),'train_auroc:', test(model, loader)['auroc'],'test_auroc:', test(model, test_loader)['auroc'])


def test(model, loader, mode):
  model.eval()
  pred = []
  label = []
  for data in loader:# Iterate in batches over the training/test dataset.
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred.append(out.view(-1))
    label.append(data.y)
  pred = torch.cat(pred, dim=0)
  label = torch.cat(label, dim=0)
  if (mode == 'regression'):
      criterion = torch.nn.MSELoss()
      result = evaluation_regression(pred, label)
  else:
      pred = torch.cat((1 - torch.unsqueeze(pred, 1), torch.unsqueeze(pred, 1)), 1)
      result = model_evaluation(pred, label)
  #print('test:',result)
  return result # Derive ratio of correct predictions.




def run_onetime(Dataset, model_name, mode, learning_rate=0.001, splitter='stratified', batch_size=64, hidden_size = 45, epoch = 50, process=True):
  # 载入数据集
  tasks, datasets, transformers = eval('dc.molnet.load_'+Dataset)(featurizer=dc.feat.MolGraphConvFeaturizer(),splitter=splitter)
  train_dataset, valid_dataset, test_dataset = datasets

  # 数据集转换
  if (mode == 'classification'):
      train_loader = DataLoader(dataset_balanced(train_dataset), batch_size=batch_size,shuffle=True)
      test_loader = DataLoader(dataset_balanced(test_dataset), batch_size=batch_size,shuffle=False)
      valid_loader = DataLoader(dataset_balanced(valid_dataset), batch_size=batch_size,shuffle=False)
  else:
      train_loader = DataLoader(GraphDataset_to_pygDatalist(train_dataset), batch_size=batch_size, shuffle=True)
      test_loader = DataLoader(GraphDataset_to_pygDatalist(test_dataset), batch_size=batch_size, shuffle=True)
      valid_loader = DataLoader(GraphDataset_to_pygDatalist(valid_dataset), batch_size=batch_size, shuffle=True)

  # 初始化模型
  input_size = datasets[0].X[0].node_features.shape[1]
  n_tasks = len(tasks)
  model = model_name(input_size=input_size, hidden_size=hidden_size, output_size=n_tasks).to(device)

  # 训练模型
  train_record= train(model, mode, train_loader=train_loader, test_loader=test_loader, valid_loader=valid_loader, process=process, epoch=epoch, learning_rate=learning_rate)

  # 模型评估
  train_result =test(model, train_loader, mode)
  test_result = test(model, test_loader, mode)
  valid_result = test(model, valid_loader, mode)

  return dict(train_result=train_result, test_result=test_result, valid_result=valid_result, train_record=train_record)





def run_n_times(Dataset, model,mode,n=10, learning_rate=0.001, splitter='stratified', batch_size=64, hidden_size = 45, epoch = 50, process=False):
    if (mode == 'regression'):
        test_result = pd.DataFrame(np.zeros((n, 7)), columns=['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'EV', 'Pearson'])
        valid_result = pd.DataFrame(np.zeros((n, 7)), columns=['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'EV', 'Pearson'])
        for i in range(n):
            result = run_onetime(Dataset, model, mode='regression', learning_rate=learning_rate, splitter=splitter, batch_size=batch_size, hidden_size=hidden_size, epoch=epoch, process=process)
            test_result.iloc[i, 0] = result['test_result']['r2'].item()
            test_result.iloc[i, 1] = result['test_result']['mae'].item()
            test_result.iloc[i, 2] = result['test_result']['mse'].item()
            test_result.iloc[i, 3] = result['test_result']['rmse'].item()
            test_result.iloc[i, 4] = result['test_result']['mape'].item()
            test_result.iloc[i, 5] = result['test_result']['ev'].item()
            test_result.iloc[i, 6] = result['test_result']['pearson'].item()
            valid_result.iloc[i, 0] = result['valid_result']['r2'].item()
            valid_result.iloc[i, 1] = result['valid_result']['mae'].item()
            valid_result.iloc[i, 2] = result['valid_result']['mse'].item()
            valid_result.iloc[i, 3] = result['valid_result']['rmse'].item()
            valid_result.iloc[i, 4] = result['valid_result']['mape'].item()
            valid_result.iloc[i, 5] = result['valid_result']['ev'].item()
            valid_result.iloc[i, 6] = result['valid_result']['pearson'].item()

        summary_test = pd.concat([test_result.mean(axis=0).round(3).astype('str'), test_result.std(axis=0).round(3).astype('str')], axis=1)
        summary_test['test'] = summary_test[0] + '+' +summary_test[1]
        summary_valid = pd.concat([valid_result.mean(axis=0).round(3).astype('str'), valid_result.std(axis=0).round(3).astype('str')], axis=1)
        summary_valid['valid'] = summary_valid[0] + '+' +summary_valid[1]
        summary = pd.concat([summary_test['test'], summary_valid['valid']], axis=1)
        return summary
    else:
        test_result = pd.DataFrame(np.zeros((n, 7)), columns=['Accuracy', 'AUC', 'Precision', 'F1', 'Recall', 'AUPR', 'Mcc'])
        valid_result = pd.DataFrame(np.zeros((n, 7)), columns=['Accuracy', 'AUC', 'Precision', 'F1', 'Recall', 'AUPR', 'Mcc'])
        for i in range(n):
            result = run_onetime(Dataset, model, mode='classification', splitter=splitter, batch_size=batch_size,
                                 hidden_size=hidden_size, epoch=epoch, process=process)
            test_result.iloc[i, 0] = result['test_result']['accuracy'].item()
            test_result.iloc[i, 1] = result['test_result']['auroc'].item()
            test_result.iloc[i, 2] = result['test_result']['precision'].item()
            test_result.iloc[i, 3] = result['test_result']['F1'].item()
            test_result.iloc[i, 4] = result['test_result']['Recall'].item()
            test_result.iloc[i, 5] = result['test_result']['auprc'].item()
            test_result.iloc[i, 6] = result['test_result']['MCC'].item()
            valid_result.iloc[i, 0] = result['valid_result']['accuracy'].item()
            valid_result.iloc[i, 1] = result['valid_result']['auroc'].item()
            valid_result.iloc[i, 2] = result['valid_result']['precision'].item()
            valid_result.iloc[i, 3] = result['valid_result']['F1'].item()
            valid_result.iloc[i, 4] = result['valid_result']['Recall'].item()
            valid_result.iloc[i, 5] = result['valid_result']['auprc'].item()
            valid_result.iloc[i, 6] = result['valid_result']['MCC'].item()

        summary_test = pd.concat(
            [test_result.mean(axis=0).round(3).astype('str'), test_result.std(axis=0).round(3).astype('str')], axis=1)
        summary_test['test'] = summary_test[0] + '+' + summary_test[1]
        summary_valid = pd.concat(
            [valid_result.mean(axis=0).round(3).astype('str'), valid_result.std(axis=0).round(3).astype('str')], axis=1)
        summary_valid['valid'] = summary_valid[0] + '+' + summary_valid[1]
        summary = pd.concat([summary_test['test'], summary_valid['valid']], axis=1)
        return summary



#dataset: a list in which each element is  a 元组(dataset, mode)
def MP_run(dataset):
    result = {}
    len(dataset)
    for i in range(len(dataset)):
        result[dataset[i][0] + '_' + dataset[i][1] + '_' + dataset[i][2]] = run_n_times(Dataset=dataset[i][0], model=Graph_Class, mode=dataset[i][1], n=10, splitter=dataset[i][2], epoch=50)
    return result








class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





def multitasks_to_singletask(dataset):
    sss = []
    for j in range(dataset.y.shape[1]):
        temp = []
        for i in range(len(dataset.X)):
            a = dataset.X[i].to_pyg_graph()
            a.y = torch.tensor(dataset.y[i, j].astype(np.int64))
            a.w = torch.tensor(dataset.w[i, j])
            temp.append(a)
        sss.append(temp)
    return sss





def multi_task(Dataset, model_name, mode='classification', splitter='stratified', batch_size=64, hidden_size = 45, epoch = 50, process=False):
    tasks, dataset, transform = eval('dc.molnet.load_'+Dataset)(featurizer=dc.feat.MolGraphConvFeaturizer(),splitter=splitter)
    train_dataset, test_dataset, valid_dataset = dataset
    train_dataset = multitasks_to_singletask(train_dataset)
    test_dataset = multitasks_to_singletask(test_dataset)
    valid_dataset = multitasks_to_singletask(valid_dataset)
    result = []
    for i in range(len(tasks)):
        train_loader = DataLoader(train_dataset[i], batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(test_dataset[i], batch_size=batch_size,shuffle=True)
        valid_loader = DataLoader(valid_dataset[i], batch_size=batch_size,shuffle=True)

        input_size = dataset[0].X[0].node_features.shape[1]
        n_tasks = 1
        model = model_name(input_size=input_size, hidden_size=hidden_size, output_size=n_tasks).to(device)

        # 训练模型
        train_record = train(model, mode, train_loader=train_loader, test_loader=test_loader, valid_loader=valid_loader,
                             process=process, epoch=epoch, learning_rate=0.0005)

        # 模型评估
        train_result = test(model, train_loader, mode)
        test_result = test(model, test_loader, mode)
        valid_result = test(model, valid_loader, mode)

        result.append(dict(train_result=train_result, test_result=test_result, valid_result=valid_result, train_record=train_record))
    return result



#需要改进：多标签任务不要拆分成多个单标签任务，因为不同标签之间存在关系，应该采用增加输出层神经元节点的方式。
def multi_task_n_times(n, Dataset, model_name, mode='classification', splitter='stratified', batch_size=64, hidden_size = 45, epoch = 50, process=False):
    result = []
    for i in range(n):
        tt = multi_task(Dataset, model_name, mode, splitter, batch_size, hidden_size, epoch, process)
        result.append(tt)
    temp = pd.DataFrame(np.zeros((n, 7)))
    temp1 = pd.DataFrame(np.zeros((n, 7)))
    summary_test = pd.DataFrame(np.zeros((7, len(result[0]))))
    summary_valid = pd.DataFrame(np.zeros((7, len(result[0]))))
    for k in range(len(result[0])):
        for j in range(n):
            temp.iloc[j, 0] = result[j][k]['test_result']['accuracy'].item()
            temp.iloc[j, 1] = result[j][k]['test_result']['auroc'].item()
            temp.iloc[j, 2] = result[j][k]['test_result']['precision'].item()
            temp.iloc[j, 3] = result[j][k]['test_result']['F1'].item()
            temp.iloc[j, 4] = result[j][k]['test_result']['Recall'].item()
            temp.iloc[j, 5] = result[j][k]['test_result']['auprc'].item()
            temp.iloc[j, 6] = result[j][k]['test_result']['MCC'].item()
            temp1.iloc[j, 0] = result[j][k]['valid_result']['accuracy'].item()
            temp1.iloc[j, 1] = result[j][k]['valid_result']['auroc'].item()
            temp1.iloc[j, 2] = result[j][k]['valid_result']['precision'].item()
            temp1.iloc[j, 3] = result[j][k]['valid_result']['F1'].item()
            temp1.iloc[j, 4] = result[j][k]['valid_result']['Recall'].item()
            temp1.iloc[j, 5] = result[j][k]['valid_result']['auprc'].item()
            temp1.iloc[j, 6] = result[j][k]['valid_result']['MCC'].item()
        temp_test = pd.concat([temp.mean(axis=0).round(3).astype('str'), temp.std(axis=0).round(3).astype('str')], axis=1)
        summary_test.iloc[:, k] = temp_test[0] + '+' + temp_test[1]
        temp_valid = pd.concat([temp1.mean(axis=0).round(3).astype('str'), temp1.std(axis=0).round(3).astype('str')],axis=1)
        summary_valid.iloc[:, k] = temp_valid[0] + '+' + temp_valid[1]
    summary_test.index = ['Accuracy', 'AUC', 'Precision', 'F1', 'Recall', 'AUPR', 'Mcc']
    summary_valid.index = ['Accuracy', 'AUC', 'Precision', 'F1', 'Recall', 'AUPR', 'Mcc']
    return dict(result=result, summary_test=summary_test, summary_valid=summary_valid)








