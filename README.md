Sistema de Detecção e Classificação de Bananas com Arduino e Visão Computacional

Este projeto integra visão computacional (usando TensorFlow Lite e OpenCV) com um sistema físico controlado por Arduino.
O sistema detecta a presença de uma banana em uma esteira, analisa sua qualidade (boa ou ruim) e acende LEDs correspondentes no Arduino.

Instalação (usando Anaconda)

1️⃣ Criar e ativar o ambiente

conda create -n banana-detector python=3.12.7
conda activate banana-detector

2️⃣ Instalar as dependências

pip install opencv-python numpy tensorflow pyserial


💡 Dica: caso o TensorFlow padrão não funcione bem no seu sistema, use a versão leve:

pip install tflite-runtime


3️⃣ Clonar o repositório

git clone https://github.com/<seu-usuario>/banana-detector.git
cd banana-detector


4️⃣ Conectar o Arduino


Faça upload do código arduino_banana.ino pelo Arduino IDE.

Verifique a porta serial (ex: COM6 no Windows ou /dev/ttyUSB0 no Linux).

Atualize a linha no Python:

arduino = serial.Serial('COM6', 9600)


▶️ Execução

Passos:

Certifique-se de que o Arduino está conectado e o sensor IR funciona.

Execute o script principal:

python main.py


Quando o Arduino detectar uma banana, o script abrirá uma janela de vídeo.

O sistema fará várias predições e enviará o resultado (“boa” ou “ruim”) ao Arduino.

O LED correspondente acenderá brevemente.

Encerrar manualmente

Pressione q para fechar a janela de vídeo e encerrar o processo.
