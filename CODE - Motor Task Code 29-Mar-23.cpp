#include <Stepper.h>

const int stepsPerRevolution = 516;  // change this to fit the number of steps per revolution for your stepper motor
const int ledPin = 13;  // the pin that the trial indicator LED is attached to (output)
const int irPin = 2;  // the pin that the IR Phototransistor is attached to (input)
unsigned long previousTime = 0;
const unsigned long bufferTime = 30000; // 30 seconds buffer time between trials, changes this value to alter time between trials

// initialize the stepper library on pins 9 through 12 (ssee schematic):
Stepper myStepper(stepsPerRevolution, 9, 10, 11, 12);

void setup() {
  pinMode(ledPin, OUTPUT);  // initialize the LED pin as an output
  pinMode(irPin, INPUT);  // initialize the IR Phototransistor pin as an input
  Serial.begin(9600);  // initialize serial communication at 9600 bits per second
}

void loop() {
  unsigned long currentTime = millis();
  // read the state of the IR Phototransistor
  int irState = digitalRead(irPin);

  // check if there is any incoming serial data
  if (Serial.available() > 0) {
    // read the incoming byte
    byte incomingByte = Serial.read();
    // if the incoming byte is 't', start a trial immediately
    if (incomingByte == 't') {
      Serial.print("Trial started at: ");
      Serial.println(currentTime);
      digitalWrite(ledPin, HIGH);  // turn on the indicator LED
      myStepper.setSpeed(50);  // set the speed of the stepper motor 
      myStepper.step(30);  // move the stepper motor 30 degrees
      previousTime = currentTime;
      delay(500);  // wait for half a second
    }
  }

  // if the IR Phototransistor is tripped and either 30 seconds have passed since the last activation
  // or a trial has been triggered by a serial input, turn on the LED and move the stepper motor 30 degrees
  if (irState == LOW && (currentTime - previousTime > bufferTime || Serial.available() > 0)) {
    Serial.print("Trial started at: ");
    Serial.println(currentTime);
    digitalWrite(ledPin, HIGH);  // turn on the LED
    myStepper.setSpeed(50);  // set the speed of the stepper motor
    myStepper.step(30);  // move the stepper motor 30 degrees
    previousTime = currentTime;
    delay(500);  // wait for half a second
  } else {
    digitalWrite(ledPin, LOW);  // turn off the LED
  }
}
