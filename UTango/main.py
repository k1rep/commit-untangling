import random
from itertools import permutations
from time import sleep

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm
import model
import os.path as osp
import os
import copy
import numpy as np


def start_training(dataset):
    random.shuffle(dataset)
    # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    test_dataset = dataset[train_size:]
    model_ = model.UTango(h_size=128, max_context=5, drop_out_rate=0.5, gcn_layers=3)
    train(epochs=1, trainLoader=train_dataset, testLoader=test_dataset, model=model_, learning_rate=1e-4)


def evaluate_metrics(model, test_loader):
    model.eval()
    with torch.no_grad():
        acc = 0
        for data in tqdm(test_loader):
            out = model(data[:-1])
            temp_acc = 0
            for i in range(len(out)):
                loop_set = loop_calculation(out[i], data[-1][i])
                max_acc = -999
                for pos_ in loop_set:
                    tmp_acc = accuracy_score(pos_, data[-1][i])
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
        tem_input = copy.deepcopy(input_1)
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
    for perm in permutations(input_list, len(label_list)):
        pair = list(zip(perm, label_list))
        set_pairs.append(pair)

    # 保持唯一性
    unique_set_pairs = []
    seen = set()
    for pair in set_pairs:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in seen:
            seen.add(sorted_pair)
            unique_set_pairs.append([list(p) for p in pair])

    return unique_set_pairs


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
            for index, _data in enumerate(tqdm(trainLoader, leave=False)):
                data_set = [
                    data.to(device) if isinstance(data, Data) else Data(
                        x=torch.tensor(data.x, dtype=torch.int8, device=device),
                        edge_index=torch.tensor(data.edge_index, dtype=torch.int8, device=device),
                        y=torch.tensor(data.y, dtype=torch.int8, device=device)
                    )
                    for data in _data[:-1]
                ]
                optimizer.zero_grad()
                out = model(data_set)
                y_ = _data[-1]
                if isinstance(y_, list):
                    y_ = [item.to(device) if isinstance(item, torch.Tensor) else torch.tensor(item, device=device) for
                          item in y_]
                else:
                    y_ = y_.to(device)

                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for i in range(len(out)):
                    y_i = y_[i] if isinstance(y_[i], torch.Tensor) else torch.tensor(y_[i], device=device)
                    loop_set = loop_calculation(out[i], y_i.cpu().numpy())
                    min_loss = None
                    for data_setting in loop_set:
                        temp_loss = criterion(
                            torch.tensor(data_reformat(data_setting, y_i.cpu().numpy()), dtype=torch.float,
                                         device=device),
                            y_i
                        )
                        if min_loss is None or temp_loss < min_loss:
                            min_loss = temp_loss
                    if min_loss is not None:
                        total_loss = total_loss + min_loss
                total_loss.backward()
                optimizer.step()
                if index % 20 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, index + 1, total_loss.item()))
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


if __name__ == '__main__':
    dataset = []
    for i in range(30):
        # Change based on your dataset size.
        data = torch.load(osp.join(os.getcwd() + "/data/", 'data_{}.pt'.format(i)))
        dataset.append(data)
    start_training(dataset)
