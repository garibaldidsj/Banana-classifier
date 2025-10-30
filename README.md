# *Sistema de Detecção e Classificação de Bananas com Arduino e Visão Computacional*

Este projeto integra visão computacional (usando TensorFlow Lite e OpenCV) com um sistema físico controlado por Arduino.
O sistema detecta a presença de uma banana em uma esteira, analisa sua qualidade (boa ou ruim) e acende LEDs correspondentes no Arduino.

# Instalação (usando Anaconda)

## 1️⃣ Criar e ativar o ambiente
```bash
conda create -n Banana-classifier python=3.12.7
conda activate Banana-classifier
```
## 2️⃣ Instalar as dependências
```bash
pip install opencv-python numpy tensorflow pyserial
```

### 💡 Dica: caso o TensorFlow padrão não funcione bem no seu sistema, use a versão leve:
```bash
pip install tflite-runtime
```

## 3️⃣ Clonar o repositório
```bash

git clone https://github.com/garibaldidsj/Banana-classifier.git
cd Banana-classifier
```

## 4️⃣ Conectar o Arduino

Abra o código arduino_esteira.ino pelo Arduino IDE.

Verifique a porta serial (ex: COM6 no Windows ou /dev/ttyUSB0 no Linux).

Atualize a linha no Python:
```bash
arduino = serial.Serial('COM6', 9600)
```


# Execução

## Passos:

Certifique-se de que o Arduino está conectado e o sensor IR funciona.

Execute o script principal:
```bash


python camera_detecta_tflite.py
```

Quando o Arduino detectar uma banana, o script abrirá uma janela de vídeo.

O sistema fará várias predições e enviará o resultado (“boa” ou “ruim”) ao Arduino.

O LED correspondente acenderá brevemente.

Encerrar manualmente

Pressione q para fechar a janela de vídeo e encerrar o processo.

# Treinando modelos

Os modelos pré-treinados já estão disponíveis, tanto na versão .h5 como .tflite.

Caso haja necessidade de um novo treinamento, os scripts utilizados estão nos arquivos .ipynb, tanto para o detector de banana quando para o classificador de qualidade.

Para executar esses arquivos será necessário a utilização de um ambiente baseado em Jupiter.
