import serial
import torch
from torch import nn
import torch.optim as optim
import os
import pandas as pd
from typing import Optional
from input_listening.__antennahandler import SERIAL_COLUMNS, AntennaHandler
from input_listening.__audiohandler import AUDIO_COLUMNS

data_features = SERIAL_COLUMNS
data_labels = AUDIO_COLUMNS[2:]
num_inputs = len(data_features)
num_outputs = len(data_labels)


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

    def train(self, path: str, learning_rate: float = 0.001):
        # TODO: make it that train can get a dir and not only a file, use keyboardthread.MERGE_PATH iuf needed
        uni_model = self.create_uni_model()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(uni_model.parameters(), lr=learning_rate)
        loss = torch.zeros((1, 1))

        full_data = pd.read_csv(filepath_or_buffer=path, header=0, dtype=int, encoding="UTF-8")
        full_data = full_data.sort_values(['EPC', 'Time'])
        for epc, df in full_data.groupby(full_data["EPC"]):
            uni_model.clear_hidden_layer()
            data = torch.from_numpy(df[data_features].values).float()
            labels = torch.from_numpy(df[data_labels].values).float()
            id_tensor = torch.unsqueeze(data, 1)
            output = uni_model.forward(input_values=id_tensor).float()
            temp_loss = loss_func(torch.flatten(output), torch.flatten(labels))
            print(f"loss: {temp_loss}")
            loss += temp_loss
        print(f"total loss: {loss}")
        loss.backward()
        optimizer.step()
        uni_model.zero_grad()

        torch.save(uni_model, self.uni_model_path)

    def predict_stream(self, port: str, output_file: Optional[str] = None):
        ser = serial.Serial(port=port, baudrate=AntennaHandler.BAUDRATE)
        header = data_features + data_labels

        def line_handler():
            try:
                raw = ser.read_until()
                inputs = AntennaHandler.parse_raw(raw)
            except ValueError:
                return None
            epc = inputs[0]
            uni_model = self.create_uni_model(epc)
            input_tensor = torch.Tensor([int(val) for val in inputs])
            input_tensor = input_tensor[None, None, ...]
            prediction = uni_model.forward(input_tensor)
            output_tensor = torch.cat([input_tensor, prediction], dim=2)
            output = torch.squeeze(output_tensor).tolist()
            output = str(output)[1:-1]
            return output

        if output_file is not None:
            if not os.path.exists(output_file):
                with open(fr"{output_file}", 'x') as file:
                    file.write(",".join(header) + "\n")
            while True:
                with open(fr"{output_file}", 'a') as file:
                    line_output = line_handler()
                    file.write(line_output)
        else:
            while True:
                line_output = line_handler()
                print(line_output)


class UniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=num_inputs, hidden_size=num_inputs)
        self.hidden_layer = torch.zeros(size=(1, 1, num_inputs)).float()
        self.linear = nn.Linear(in_features=num_inputs, out_features=num_outputs)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # output1, hidden_layer = self.rnn(input, self.hidden_layer)
        # self.hidden_layer = hidden_layer
        # return self.linear(output1)

        ls = []
        for uno in torch.split(input_values, 1):
            output, hidden_layer = self.rnn(uno, self.hidden_layer)
            self.hidden_layer = hidden_layer
            ls.append(output)
        output2 = torch.cat(ls, dim=0)
        return self.linear(output2)

    def clear_hidden_layer(self):
        self.hidden_layer = torch.zeros(size=(1, 1, num_inputs)).float()
