import torch
from torch import nn
import torch.optim as optim
import os
import pandas as pd
from antennahandler import AntennaHandler
from audiohandler import AudioHandler
from typing import Optional

num_inputs = len(AntennaHandler.COLUMNS)
num_outputs = len(AudioHandler.COLUMNS) - 2


class Model:
    def __init__(self, uni_model_path: str):
        self.uni_model_path = uni_model_path
        self.models = dict()

    def create_uni_model(self, epc: Optional = None):
        if epc is None:
            if os.path.exists(self.uni_model_path):
                uni_model = torch.load(self.uni_model_path)
            else:
                uni_model = UniModel()
                torch.save(uni_model, self.uni_model_path)
        else:
            if epc in self.models:
                uni_model = self.models[epc]
            else:
                uni_model = torch.load(self.uni_model_path)
                self.models[epc] = uni_model
        return uni_model

    # def train(self, file_path: str):
    #     with open(file_path, 'r', encoding='utf-8') as input_file:
    #         columns = input_file.readline()[:-1]
    #         columns = list(columns.split(','))
    #         # columns will have all antenna_handler columns and then all the labels
    #         inputs_len = len(AntennaHandler.COLUMNS)
    #         for line in input_file:
    #             line = line[:-1]
    #             line = list(line.split(','))
    #             data = line[:inputs_len]
    #             labels = line[inputs_len:]
    #             print("a")
    #             epc = data[0]
    #             self.get_uni_model(epc)

    def train(self, file_path: str, learning_rate: float = 0.001):
        uni_model = self.create_uni_model()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(uni_model.parameters(), lr=learning_rate)
        loss = torch.zeros((1, 1))

        full_data = pd.read_csv(filepath_or_buffer=file_path, header=0, dtype=int, encoding="UTF-8")
        full_data = full_data.sort_values(['EPC', 'Time'])
        for epc, df in full_data.groupby(full_data["EPC"]):
            uni_model.clear_hidden_layer()
            data = torch.from_numpy(df[AntennaHandler.COLUMNS].values).float()
            labels = torch.from_numpy(df[AudioHandler.COLUMNS[2:]].values).float()
            id_tensor = torch.unsqueeze(data, 1)
            output = uni_model.forward(input=id_tensor).float()
            temp_loss = loss_func(torch.flatten(output), torch.flatten(labels))
            print(f"loss: {temp_loss}")
            loss += temp_loss
        print(f"total loss: {loss}")
        loss.backward()
        optimizer.step()
        uni_model.zero_grad()
        torch.save(uni_model, self.uni_model_path)


class UniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=num_inputs, hidden_size=num_inputs)
        self.hidden_layer = torch.zeros(size=(1, 1, num_inputs)).float()
        self.linear = nn.Linear(in_features=num_inputs, out_features=num_outputs)

    def forward(self, input):
        # output1, hidden_layer = self.rnn(input, self.hidden_layer)
        # self.hidden_layer = hidden_layer
        # return self.linear(output1)

        ls = []
        for uno in torch.split(input, 1):
            output, hidden_layer = self.rnn(uno, self.hidden_layer)
            self.hidden_layer = hidden_layer
            ls.append(output)
        output2 = torch.cat(ls, dim=0)
        return self.linear(output2)

    def clear_hidden_layer(self):
        self.hidden_layer = torch.zeros(size=(1, 1, num_inputs)).float()
