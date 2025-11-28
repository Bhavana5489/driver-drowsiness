#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// =========================================
// Connect to your Mobile Hotspot (STA Mode)
// =========================================
const char* ssid = "RosxIoTBroker";      // Change to your phone hotspot name
const char* password = "rosxtestbroker"; // Change to your phone hotspot password

// UDP configuration  
WiFiUDP Udp;
unsigned int localUdpPort = 4210;
char incomingPacket[255];

// LED pins
const int sleepLedPin = D4;
const int normalLedPin = D5;
const int noFaceLedPin = D6;

// Buzzer pin
const int buzzerPin = D1;  // Connect buzzer to D1

// Blinking variables
bool blinkState = false;
unsigned long previousBlinkMillis = 0;
const unsigned long blinkInterval = 500;  // 500ms blink interval

// Buzzer variables
bool buzzerActive = false;
unsigned long previousBuzzerMillis = 0;
const unsigned long buzzerBeepInterval = 300;  // 300ms beep interval
const unsigned long buzzerSilenceInterval = 700; // 700ms silence interval
bool buzzerBeepState = false;

// Current driver state
String currentState = "";

// Connection status
bool wifiConnected = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println();
  Serial.println("Connecting to Mobile Hotspot...");
  
  // Initialize LED pins
  pinMode(sleepLedPin, OUTPUT);
  pinMode(normalLedPin, OUTPUT);
  pinMode(noFaceLedPin, OUTPUT);
  
  // Initialize buzzer pin
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);  // Start with buzzer off
  
  // Start with all LEDs off
  digitalWrite(sleepLedPin, LOW);
  digitalWrite(normalLedPin, LOW);
  digitalWrite(noFaceLedPin, LOW);
  
  // Connect to Mobile Hotspot (STA Mode)
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
    // Blink green LED while connecting
    digitalWrite(normalLedPin, (attempts % 2 == 0) ? HIGH : LOW);
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println();
    Serial.println("Connected to Mobile Hotspot!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    
    // Start UDP
    Udp.begin(localUdpPort);
    Serial.printf("Now listening at IP %s, UDP port %d\n", 
                  WiFi.localIP().toString().c_str(), localUdpPort);
    
    // Solid green LED when connected
    digitalWrite(normalLedPin, HIGH);
  } else {
    Serial.println();
    Serial.println("Failed to connect to Mobile Hotspot!");
    wifiConnected = false;
    // Blink all LEDs rapidly to indicate error
    while(1) {
      digitalWrite(sleepLedPin, HIGH);
      digitalWrite(normalLedPin, HIGH);
      digitalWrite(noFaceLedPin, HIGH);
      delay(200);
      digitalWrite(sleepLedPin, LOW);
      digitalWrite(normalLedPin, LOW);
      digitalWrite(noFaceLedPin, LOW);
      delay(200);
    }
  }
}

void controlBuzzer(bool activate) {
  if (activate && !buzzerActive) {
    buzzerActive = true;
    previousBuzzerMillis = millis();
    buzzerBeepState = true;
    Serial.println("BUZZER: Activated with beeping");
  } else if (!activate && buzzerActive) {
    buzzerActive = false;
    digitalWrite(buzzerPin, LOW);
    Serial.println("BUZZER: Deactivated");
  }
}

void updateBlinking() {
  unsigned long currentMillis = millis();
  
  // Handle LED blinking for DRIVER_SLEPT state
  if (currentState == "DRIVER_SLEPT") {
    if (currentMillis - previousBlinkMillis >= blinkInterval) {
      previousBlinkMillis = currentMillis;
      blinkState = !blinkState;
      digitalWrite(sleepLedPin, blinkState ? HIGH : LOW);
    }
  }
  
  // Handle buzzer beeping pattern for DRIVER_SLEPT state
  if (buzzerActive) {
    if (buzzerBeepState) {
      // Currently in beep phase
      if (currentMillis - previousBuzzerMillis >= buzzerBeepInterval) {
        previousBuzzerMillis = currentMillis;
        buzzerBeepState = false;
        digitalWrite(buzzerPin, LOW);  // Turn off buzzer
      }
    } else {
      // Currently in silence phase
      if (currentMillis - previousBuzzerMillis >= buzzerSilenceInterval) {
        previousBuzzerMillis = currentMillis;
        buzzerBeepState = true;
        digitalWrite(buzzerPin, HIGH);  // Turn on buzzer
      }
    }
  }
}

void checkWiFiConnection() {
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > 10000) { // Check every 10 seconds
    lastCheck = millis();
    
    if (WiFi.status() != WL_CONNECTED && wifiConnected) {
      Serial.println("WiFi connection lost! Attempting to reconnect...");
      wifiConnected = false;
      digitalWrite(normalLedPin, LOW);
      
      WiFi.reconnect();
      int attempts = 0;
      while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
        digitalWrite(normalLedPin, (attempts % 2 == 0) ? HIGH : LOW);
      }
      
      if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.println("\nReconnected to WiFi!");
        digitalWrite(normalLedPin, HIGH);
      } else {
        Serial.println("\nReconnection failed!");
      }
    }
  }
}

void loop() {
  // Check WiFi connection periodically
  checkWiFiConnection();
  
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    int len = Udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0;
    }
    
    String receivedMessage = String(incomingPacket);
    Serial.printf("Received: %s\n", receivedMessage.c_str());
    
    // Store current state
    currentState = receivedMessage;
    
    // Control LEDs and buzzer based on driver state
    if (receivedMessage == "DRIVER_SLEPT") {
      // Driver is asleep - blink red LED, turn off others, activate buzzer
      digitalWrite(normalLedPin, LOW);
      digitalWrite(noFaceLedPin, LOW);
      controlBuzzer(true);
      Serial.println("ALERT: Driver slept - RED LED BLINKING, BUZZER BEEPING");
    } 
    else if (receivedMessage == "NORMAL") {
      // Driver is normal - turn on green LED, turn off others, deactivate buzzer
      digitalWrite(sleepLedPin, LOW);
      digitalWrite(normalLedPin, HIGH);
      digitalWrite(noFaceLedPin, LOW);
      controlBuzzer(false);
      Serial.println("Driver normal - GREEN LED ON");
    }
    else if (receivedMessage == "NO_FACE") {
      // No face detected - turn on yellow LED, turn off others, deactivate buzzer
      digitalWrite(sleepLedPin, LOW);
      digitalWrite(normalLedPin, LOW);
      digitalWrite(noFaceLedPin, HIGH);
      controlBuzzer(false);
      Serial.println("No face detected - YELLOW LED ON");
    }
    else {
      // Other states - default to normal
      digitalWrite(sleepLedPin, LOW);
      digitalWrite(normalLedPin, HIGH);
      digitalWrite(noFaceLedPin, LOW);
      controlBuzzer(false);
      Serial.println("Other state - GREEN LED ON");
    }
  }
  
  // Update blinking and buzzer patterns
  updateBlinking();
  
  delay(10);
}
