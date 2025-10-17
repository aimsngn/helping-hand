#include <Servo.h>

Servo servos[5];
int pins[5] = {3, 5, 6, 9, 10};  // adjust to your servo pins
int openAngle = 120;             // angle when "1" (open)
int closedAngle = 0;             // angle when "0" (closed)
String inputString = "";
bool reading = false;

void setup() {
  Serial.begin(9600);  // must match baudRate in HTML (9600)
  for (int i = 0; i < 5; i++) {
    servos[i].attach(pins[i]);
    servos[i].write(closedAngle); // start closed
  }
  Serial.println("Ready for commands like $01111");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '$') {           // start of message
      inputString = "";
      reading = true;
    } 
    else if (reading) {
      if (c >= '0' && c <= '1') {
        inputString += c;
        if (inputString.length() == 5) { // got all 5 digits
          for (int i = 0; i < 5; i++) {
            int state = inputString[i] - '0';
            servos[i].write(state ? openAngle : closedAngle);
          }
          Serial.print("Received: $");
          Serial.println(inputString);
          reading = false;
        }
      } 
      else {
        reading = false; // invalid character, reset
      }
    }
  }
}
