import gc
import random
from itertools import permutations
from time import sleep

import numpy
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import model
import os.path as osp
import os
import numpy as np


class AdaptiveBatchSampler(IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        labels_batch = []
        for data in self.dataset:
            if isinstance(data, Data):
                batch.append(data)
            elif isinstance(data, list):
                labels_batch.append(data)
                if batch:
                    yield batch, labels_batch
                    batch, labels_batch = [], []
            if len(batch) >= self.batch_size:
                yield batch, labels_batch
                batch, labels_batch = [], []
        if batch:
            yield batch, labels_batch

    def __len__(self):
        return len(self.dataset)


def custom_collate_fn(batch):
    data_list, labels_list = batch
    batch_data = Batch.from_data_list(data_list)
    return batch_data, labels_list


def start_training(dataset):
    random.shuffle(dataset)
    # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    test_dataset = dataset[train_size:]
    # 去掉train_set的外层list
    train_dataset = [item for sublist in train_dataset for item in sublist]
    # 去掉test_set的外层list
    test_dataset = [item for sublist in test_dataset for item in sublist]
    train_sampler = AdaptiveBatchSampler(train_dataset, batch_size=len(train_dataset))
    test_sampler = AdaptiveBatchSampler(test_dataset, batch_size=len(test_dataset))
    train_loader = DataLoader(train_sampler, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_sampler, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    model_ = model.UTango(h_size=128, max_context=2, drop_out_rate=0.5, gcn_layers=3)
    train(epochs=1, trainLoader=train_loader, testLoader=test_loader, model=model_, learning_rate=1e-4)


def train_on_single_dataset(dataset, i):
    if i == 0:
        model_ = model.UTango(h_size=128, max_context=2, drop_out_rate=0.5, gcn_layers=3)
        previous_loss = torch.tensor(0.0, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model_.parameters(), lr=1e-4)
    else:
        checkpoint = torch.load(f"model/model_{i - 1}.pt")
        model_ = model.UTango(h_size=128, max_context=2, drop_out_rate=0.5, gcn_layers=3)
        model_.load_state_dict(checkpoint['model_state_dict'])
        previous_loss = checkpoint['loss']
        optimizer = torch.optim.Adam(model_.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_on_single(epochs=i, _data=dataset, model=model_,
                    learning_rate=1e-4, previous_loss=previous_loss, optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(), batch=i)


def evaluate_metrics(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        acc = 0
        for _data, labels in tqdm(test_loader):
            if isinstance(_data, list):
                _data = [item.to(device) for item in _data]
            else:
                _data = _data.to(device)
            out = model(_data)
            y_ = labels[0]
            temp_acc = 0
            for i in range(len(out)):
                y_i = numpy.array([data[0] for data in y_[i]], dtype=np.int64)
                if len(out[i]) != len(y_i):
                    out[i] = out[i][:len(y_i)]
                loop_set = loop_calculation(out[i], y_i)
                max_acc = -999
                for pos_ in loop_set:
                    tmp_acc = accuracy_score(pos_, y_i)
                    if tmp_acc > max_acc:
                        max_acc = tmp_acc
                temp_acc += max_acc
            temp_acc = temp_acc / len(out)
            acc += temp_acc
        acc = acc / len(test_loader)
        sleep(0.1)
        print("Average Accuracy: ", acc)


def loop_calculation(input_1, input_2):
    out_ = []
    input_set = set(input_1)
    label_set = set(input_2)
    pairs = loop_check(tuple(sorted(label_set)), tuple(sorted(input_set)))
    for pair in pairs:
        tem_input = list(input_1)
        changed = np.zeros(len(tem_input))
        for pair_info in pair:
            original_label = pair_info[0]
            replace_label = pair_info[1]
            for i in range(len(tem_input)):
                if tem_input[i] == original_label and changed[i] == 0:
                    tem_input[i] = replace_label
                    changed[i] = 1
        for i in range(len(changed)):
            if changed[i] == 0:
                tem_input[i] = 0
        out_.append(tem_input)
    return out_


def loop_check(label_tuple, input_tuple):
    label_list = list(label_tuple)
    input_list = list(input_tuple)

    if len(label_list) == 1 and len(input_list) == 1:
        return [[[input_list[0], label_list[0]]]]

    set_pairs = []
    seen = set()
    for perm in permutations(input_list, len(label_list)):
        pair = tuple(sorted(zip(perm, label_list)))
        if pair not in seen:
            seen.add(pair)
            set_pairs.append([list(p) for p in zip(perm, label_list)])

    return set_pairs


def data_reformat(input_data, label):
    max_ = 0
    for label_ in label:
        if label_ > max_:
            max_ = label_
    max_ = max_ + 1
    output_d = []
    for data_ in input_data:
        new_data = []
        for i in range(max_):
            if data_ == i + 1:
                new_data.append(1)
            else:
                new_data.append(0)
        output_d.append(new_data)
    return output_d


def train(epochs, trainLoader, testLoader, model, learning_rate):
    if not os.path.exists('model'):
        os.makedirs('model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    try:
        for e in range(epochs):
            model.train()  # 将模型设置为训练模式
            print(f"Epoch {e + 1}/{epochs}")
            for index, (_data, labels) in enumerate(tqdm(trainLoader, leave=False)):
                if isinstance(_data, list):
                    _data = [item.to(device) for item in _data]
                else:
                    _data = _data.to(device)
                optimizer.zero_grad()
                out = model(_data)
                y_ = labels[0]
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for i in range(len(out)):
                    y_i = numpy.array([data[0] for data in y_[i]], dtype=np.int64)
                    if len(out[i]) != len(y_i):
                        out[i] = out[i][:len(y_i)]
                    loop_set = loop_calculation(out[i], y_i)
                    min_loss = min(criterion(torch.tensor(data_reformat(ls, y_i), dtype=torch.float, device=device),
                                             torch.tensor(y_i, dtype=torch.long, device=device))
                                   for ls in loop_set)
                    total_loss = total_loss + min_loss
                total_loss.backward()
                optimizer.step()
                if index % 20 == 0:
                    print(f'epoch: {e + 1}, batch: {index + 1}, loss: {total_loss.item()}')
            torch.save(model.state_dict(), os.path.join('model', f"model_{e + 1}.pt"))
        evaluate_metrics(model=model, test_loader=testLoader)
    except KeyboardInterrupt:
        evaluate_metrics(model=model, test_loader=testLoader)


def demo_work(dataset):
    model_ = torch.load("model.pt")
    test_dataset = dataset
    sleep(0.1)
    evaluate_metrics(model=model_, test_loader=test_dataset)
    print("Among the demo dataset, the results are shown above")


def train_on_single(epochs, _data, model, learning_rate,
                    previous_loss, optimizer, criterion, batch):
    if not os.path.exists('model'):
        os.makedirs('model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()  # 将模型设置为训练模式
    _data = [item.to(device) if isinstance(item, Data) else item for item in _data]
    total_loss = previous_loss
    out = model(_data[:-1])
    y_ = _data[-1]
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for (i, j) in zip(out, y_):
        i = torch.tensor(i, dtype=torch.float, device=device)
        j = torch.tensor(j, dtype=torch.float, device=device)
        print("now", i, j)
        loss += criterion(i, j)
    total_loss = total_loss + loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    torch.save({
        'epoch': batch,
        'model_state_dict': model.state_dict(),
        'loss': total_loss,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('model', f"model_{batch}.pt"))


def demo_work(dataset):
    model_ = torch.load("model.pt")
    test_dataset = dataset
    sleep(0.1)
    evaluate_metrics(model=model_, test_loader=test_dataset)
    print("Among the demo dataset, the results are shown above")


if __name__ == '__main__':
    dataset = []
    # for i in range(1):
    #     # Change based on your dataset size.
    #     data = torch.load(osp.join(os.getcwd() + "/data/", 'data_{}.pt'.format(i)))
    #     dataset.append(data)
    # 从data目录下读取所有数据点数
    data_size = len(os.listdir(os.path.join(os.getcwd() + "/data/")))
    for i in range(data_size):
        dataset = torch.load(os.path.join(os.getcwd() + "/data/", f'data_{i}.pt'))
        train_on_single_dataset(dataset, i)
    # start_training(dataset)
