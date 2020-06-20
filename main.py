import os
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

ARQUIVO_REDE = './state_dict.pt'
TRAIN_WINDOW = 30 # janela de treinamento

# carrega os dados completos
pbr = pd.read_csv('PBR.csv')
data_series = pbr['Close']
# seleciona só o número de passageiros por mês
all_data = data_series.values.astype(float)

# dividir os dados em treino e teste

train_data = all_data[:-TRAIN_WINDOW]
test_data  = all_data[-TRAIN_WINDOW:]

# Normalização dos dados (deixar entre -1 e 1)
scaler = MinMaxScaler(feature_range=(-1, 1)) # cria o scaler
train_data_1_col = train_data.reshape(-1, 1)  # deixo os dados em uma única coluna
train_data_normalized = scaler.fit_transform(train_data_1_col) # normalizo

# Criar um tensor do PyTorch
# ATENÇÃO AO .view(-1): 
#   o scaler só trabalha com os dados um por linha
#   mas as RNN trabalham com sequências, por isso, 
#   a gente vai colocar tudo numa linha de novo
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

def cria_sequencias_de_treinamento(dados, janela = TRAIN_WINDOW):
  duplas = []
  tamanho_dados = len(dados)
  for i in range(tamanho_dados - janela):
    entrada = dados[i : i + janela] # para cada sequência de 'janela' elementos
    label   = dados[i + janela : i + janela + 1] # prevê o próximo valor
    duplas.append((entrada, label))
  return duplas

train_inout_seq = cria_sequencias_de_treinamento(train_data_normalized, TRAIN_WINDOW)

# Observação sobre o construtor LSTM:
#   batch_first = False --> entrada/saída: (seq_len, batch, input_size)
#   batch_first = True  --> entrada/saída: (batch, seq_len, input_size)
class PrevisaoTemporal(nn.Module):
  def __init__(self, input_size = 1, hidden_layer_size = 100, output_size = 1):
    super(PrevisaoTemporal, self).__init__()
    
    self.hidden_layer_size = hidden_layer_size
    self.gru = nn.GRU(input_size, hidden_layer_size)
    self.linear = nn.Linear(hidden_layer_size, output_size)
    self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size))

  def forward(self, input_seq):
    # precisamos deixar a entrada no formato (seq_len, batch, input_size)
    input_seq = input_seq.view(len(input_seq), 1, -1)
    gru_out, self.hidden_cell = self.gru(input_seq, self.hidden_cell)
    predictions = self.linear(gru_out.view(len(input_seq), -1))
    return predictions[-1]

# Criando o otimizador
model = PrevisaoTemporal()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def treina(epochs = 100):
  # Inicia o estado interno zerado
  model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size))

  min_loss   = 10000.0
  total_loss = 0

  for i in range(epochs):
    total_loss = 0
    for seq, labels in train_inout_seq:
      optimizer.zero_grad()

      # Reutiliza o estado interno do passo anterior
      model.hidden_cell = (model.hidden_cell.detach())

      y_pred = model(seq)

      single_loss = loss_function(y_pred, labels)
      single_loss.backward()
      optimizer.step()
      total_loss += single_loss.item()

    if total_loss < min_loss:
      print(f'MIN LOSS! SAVING ... epoch: {i:3} loss: {total_loss:10.8f}')
      torch.save(model.state_dict(), ARQUIVO_REDE)
      min_loss = total_loss

    if i % 25 == 0:
      print(f'epoch: {i:3} loss: {total_loss:10.8f}')

def faz_previsao(qtd_previsoes = TRAIN_WINDOW):
  # pega os últimos valores do conjunto de treinamento
  test_inputs = train_data_normalized[-TRAIN_WINDOW:].tolist()

  # carrega o melhor modelo salvo
  model.load_state_dict(torch.load(ARQUIVO_REDE))
  model.eval() # coloca a rede no modo de avaliação

  # Nós não modificamos o estado oculto porque nossa rede é statefull
  for i in range(qtd_previsoes):
    seq = torch.FloatTensor(test_inputs[-TRAIN_WINDOW:]) # usamos como entrada os 12 últimos valores
    with torch.no_grad(): # desligamos os gradientes da rede (necessário para o gerenciamento do estado escondido)
      test_inputs.append(model(seq).item()) # adiciona a saída ao final do conjunto de entrada  

  # A saída precisa ser escalada de volta
  actual_predictions = scaler.inverse_transform(np.array(test_inputs[qtd_previsoes:]).reshape(-1, 1))
  return actual_predictions

if __name__ == '__main__':
  print('Verificando se a rede já está treinada ...')
  if not os.path.exists(ARQUIVO_REDE):
    print('Não foi encontrado um arquivo da rede. Treinando ...')
    treina()
  else:
    print('Já existe uma rede treinada.')
  
  previsoes = faz_previsao(30)

  # Vamos calcular o eixo X das previsões
  x_inicio = len(data_series) - len(previsoes)
  x_fim    = len(data_series)
  x = list(range(x_inicio, x_fim, 1))

  plt.figure(figsize=(15, 5)) # aumenta o tamanho do plot
  plt.title('PBR Fechamento por dia')
  plt.ylabel('Valor das ações')
  plt.grid(True)
  plt.autoscale(axis='x', tight=True)
  plt.plot(data_series)
  plt.plot(x, previsoes)
  plt.show()
