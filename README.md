# Banana Classifier

Sistema de visão computacional para detecção, classificação e separação automática de bananas. O projeto utiliza **YOLOv8n** para detecção da fruta e **MobileNetV2** para classificação da qualidade, integrado a um sistema físico controlado por **Arduino Uno**.

## 📋 Requisitos

### Software

* Python 3.11 ou superior;
* Git;
* Arduino IDE.

### Hardware

* Arduino Uno;
* Sensor ultrassônico HC-SR04;
* Webcam;
* Motor para esteira;
* Servo motor;
* Estrutura da esteira;
* Mecanismo de separação.

## 🔧 Instalação

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/garibaldidsj/Banana-classifier.git
cd Banana-classifier
```

### 2️⃣ Crie um ambiente virtual

#### Windows

```bash
python -m venv .venv
```

Ative o ambiente:

```powershell
.venv\Scripts\activate
```

#### Linux

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

### 3️⃣ Instale as dependências

Atualize o `pip`:

```bash
python -m pip install --upgrade pip
```

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

## 🤖 Configuração do Arduino

Abra o arquivo `arduino_esteira.ino` utilizando a **Arduino IDE**.

Conecte o Arduino Uno ao computador e faça o upload do código.

O Arduino é responsável pelo controle:

* do sensor ultrassônico;
* do motor da esteira;
* do servo motor utilizado na separação.

Após conectar o Arduino ao computador, verifique a porta serial utilizada.

No Windows, por exemplo:

```text
COM6
```

No Linux:

```text
/dev/ttyUSB0
```

Configure a porta serial no código Python:

```python
arduino = serial.Serial('COM6', 9600)
```

Substitua `COM6` pela porta correspondente ao seu Arduino.

## 📷 Configuração da webcam

O sistema utiliza uma webcam para capturar as imagens das bananas.

Por padrão, a câmera é inicializada utilizando:

```python
cap = cv2.VideoCapture(0)
```

Caso seja necessário utilizar outra câmera, altere o índice:

```python
cap = cv2.VideoCapture(1)
```

## 🧠 Modelos utilizados

O sistema utiliza dois modelos especializados.

### YOLOv8n

O **YOLOv8n** é responsável pela detecção e localização da banana na imagem.

```text
Imagem da webcam
       ↓
    YOLOv8n
       ↓
Detecção da banana
       ↓
Bounding Box
       ↓
Recorte da fruta
```

### MobileNetV2

A **MobileNetV2** recebe o recorte da banana detectada pelo YOLOv8n e realiza a classificação da qualidade.

```text
Recorte da banana
       ↓
   MobileNetV2
       ↓
┌──────┼──────┐
↓      ↓      ↓
Verde  Comestível  Podre
```

## 📦 Dataset

O modelo de classificação utiliza o **Banana Ripeness Classification Dataset**, disponibilizado por S. M. Shahriar no Kaggle.

O dataset contém 13.478 imagens de bananas em diferentes estágios de maturação, divididas em conjuntos de treinamento, validação e teste.

O dataset está disponível em:

https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset

### Organização do dataset

| Conjunto | Imagens |
|---|---:|
| Treinamento | 11.793 |
| Validação | 1.123 |
| Teste | 562 |
| **Total** | **13.478** |

O conjunto original possui quatro classes:

```text
ripe
overripe
unripe
rotten
```

Para este projeto, as classes foram reorganizadas em três categorias:

```text
ripe + overripe → comestivel
unripe          → verde
rotten          → podre
```

A estrutura utilizada pelo projeto é:

```text
dataset/
└── quality/
    ├── train/
    │   ├── comestivel/
    │   ├── verde/
    │   └── podre/
    │
    ├── val/
    │   ├── comestivel/
    │   ├── verde/
    │   └── podre/
    │
    └── test/
        ├── comestivel/
        ├── verde/
        └── podre/
```

### Fonte

**Shahriar, S. M.** *Banana Ripeness Classification Dataset*. Kaggle.

O dataset está disponível sob a licença **Apache 2.0**.

## 🏋️ Treinamento do classificador

O treinamento da MobileNetV2 utiliza **transfer learning** a partir de uma MobileNetV2 pré-treinada.

Para iniciar o treinamento:

```bash
python train_quality.py
```

As principais configurações utilizadas são:

| Parâmetro | Configuração |
|---|---|
| Arquitetura | MobileNetV2 |
| Entrada | 224 × 224 × 3 |
| Classes | 3 |
| Batch size | 64 |
| Otimizador | Adam |
| Função de perda | Categorical Cross Entropy |
| Learning rate inicial | 1×10⁻³ |
| Learning rate no fine-tuning | 1×10⁻⁵ |

O treinamento é realizado em duas etapas:

```text
Transfer Learning
       ↓
MobileNetV2 congelada
       ↓
Treinamento inicial
       ↓
Fine-tuning
       ↓
Modelo final
```

Também são utilizados mecanismos de **Early Stopping** e **ReduceLROnPlateau**.

## 🧪 Testando o detector

O modelo YOLOv8n pode ser testado separadamente utilizando:

```bash
python test_detector.py
```

Esse teste permite verificar se o modelo está identificando corretamente as bananas antes da etapa de classificação.

## 📊 Resultados

Após o treinamento, o modelo MobileNetV2 foi avaliado utilizando o conjunto de teste, composto por **562 imagens**.

O modelo apresentou **95,55% de acurácia** no conjunto de teste.

| Classe | Precision | Recall | F1-score | Imagens |
|---|---:|---:|---:|---:|
| Comestível | 94,93% | 98,13% | 96,50% | 267 |
| Verde | 98,11% | 94,55% | 96,30% | 110 |
| Podre | 95,00% | 92,43% | 93,70% | 185 |
| **Média** | **96,01%** | **95,04%** | **95,50%** | **562** |

A acurácia geral obtida foi:

```text
Accuracy: 95,55%
```

### Matriz de confusão

```text
                  Predito
              Comestível Verde Podre
Real
Comestível        262      0      5
Verde               2    104      4
Podre              12      2    171
```

O modelo apresentou desempenho equilibrado entre as três classes. A classe `comestivel` apresentou o maior recall, com **98,13%**, enquanto a classe `verde` apresentou a maior precisão, com **98,11%**.

A classe `podre` apresentou recall de **92,43%**, sendo responsável pela maior quantidade de erros de classificação. Entre os 185 exemplos de bananas podres presentes no conjunto de teste, 171 foram corretamente classificados, enquanto 12 foram classificados como `comestivel` e 2 como `verde`.

De forma geral, os resultados indicam que a MobileNetV2 apresenta desempenho adequado para a etapa de classificação de qualidade proposta no sistema.


## ▶️ Execução do sistema

Após instalar as dependências, configurar o Arduino e conectar a webcam, execute:

```bash
python main.py
```

O sistema realiza as seguintes etapas:

1. O sensor ultrassônico detecta a entrada da fruta;
2. O Arduino aciona a esteira;
3. A webcam captura a imagem;
4. O YOLOv8n detecta a banana;
5. A região da banana é recortada;
6. A MobileNetV2 classifica a qualidade;
7. O resultado é enviado ao Arduino;
8. O servo motor movimenta a barreira;
9. A banana é direcionada para a divisória correspondente.

O fluxo completo pode ser representado por:

```text
Sensor ultrassônico
        ↓
     Arduino
        ↓
      Esteira
        ↓
      Webcam
        ↓
     YOLOv8n
        ↓
Detecção da banana
        ↓
Recorte da fruta
        ↓
   MobileNetV2
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
Verde  Comestível  Podre
        ↓
     Arduino
        ↓
   Servo motor
        ↓
Barreira de separação
        ↓
Divisória correspondente
```

## 📁 Estrutura do projeto

```text
Banana-classifier/
├── dataset/             # Dataset utilizado no treinamento
├── models/              # Modelos treinados
├── arduino_esteira.ino  # Código do Arduino
├── train_quality.py     # Treinamento da MobileNetV2
├── test_detector.py     # Teste do detector YOLOv8n
├── main.py              # Execução do sistema
├── requirements.txt     # Dependências
└── README.md
```

## 🔄 Funcionamento

O sistema combina duas etapas de visão computacional:

```text
Webcam
  ↓
YOLOv8n
  ↓
Detecção da banana
  ↓
Recorte
  ↓
MobileNetV2
  ↓
Classificação da qualidade
  ↓
Arduino
  ↓
Servo motor
  ↓
Separação
```

A utilização de modelos especializados permite separar a etapa de **detecção da fruta** da etapa de **classificação da qualidade**.

Dessa forma, a MobileNetV2 recebe apenas a região da imagem correspondente à banana detectada pelo YOLOv8n.

## ⚠️ Problemas comuns

### O sistema não encontra o Arduino

Verifique:

* se o Arduino está conectado;
* se o código foi enviado corretamente para a placa;
* se a porta serial configurada no Python está correta;
* se outro programa não está utilizando a porta serial.

### A webcam não funciona

Verifique o índice utilizado:

```python
cap = cv2.VideoCapture(0)
```

Caso necessário, tente:

```python
cap = cv2.VideoCapture(1)
```

### O treinamento está lento

O treinamento dos modelos pode exigir maior capacidade computacional dependendo da configuração utilizada.

Para melhorar o desempenho, recomenda-se:

* utilizar uma máquina com maior capacidade de processamento;
* utilizar uma GPU compatível;
* ajustar o tamanho do batch de acordo com os recursos disponíveis.

### A banana está sendo classificada incorretamente

Verifique:

* a qualidade das imagens utilizadas no treinamento;
* o balanceamento das classes;
* o desempenho da MobileNetV2 no conjunto de teste;
* se o recorte produzido pelo YOLOv8n contém corretamente a banana.

## 📄 Licença

Este projeto está disponível sob a licença **GPL-3.0**.
