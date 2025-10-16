#include <Servo.h>

const int sensorPin = 7;   // Sensor IR reflexivo
const int servoPin = 11;   // Servo motor contínuo
const int ledBoa = 2;
const int ledRuim = 3;
Servo esteira;

bool esteiraLigada = false;
unsigned long ultimoDeteccao = 0;  // marca o tempo da última detecção
const unsigned long timeout = 3000; // 3 segundos

void setup() {
  pinMode(sensorPin, INPUT);
  pinMode(ledBoa, OUTPUT);
  pinMode(ledRuim, OUTPUT);
  
  esteira.attach(servoPin);
  esteira.write(90); // parado
  Serial.begin(9600);

}

void loop() {
  
  int sensorVal = digitalRead(sensorPin);

  // Quando objeto detectado
  if (sensorVal == LOW) {  
    if (!esteiraLigada) {
      esteira.write(120);   // gira para frente
      esteiraLigada = true;
      Serial.println("detected"); // informa ao Python
    }
    ultimoDeteccao = millis(); // atualiza tempo da última detecção
  }

  // Se passaram 3 segundos sem detecção, parar esteira
  if (esteiraLigada && (millis() - ultimoDeteccao > timeout)) {
    esteira.write(90); // para a esteira
    esteiraLigada = false;
    Serial.println("timeout"); // opcional: avisa ao Python
  }

  // Recebe comando do Python
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "servo_off") {
      esteira.write(90);
      esteiraLigada = false;
    }
    else if (cmd == "servo_on") {
      esteira.write(120);
      esteiraLigada = true;
      ultimoDeteccao = millis(); // reseta o tempo
    }
    else if (cmd == "boa") {
      digitalWrite(ledBoa, HIGH);
      digitalWrite(ledRuim, LOW);
      delay(1000);
      digitalWrite(ledBoa, LOW);
    }
    else if (cmd == "ruim") {
      digitalWrite(ledBoa, LOW);
      digitalWrite(ledRuim, HIGH);
      delay(1000);
      digitalWrite(ledRuim, LOW);
    }
    else {
      digitalWrite(ledBoa, LOW);
      digitalWrite(ledRuim, LOW);
    }
  }
}
