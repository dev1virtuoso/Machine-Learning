#include <Arduino.h>
#include <Servo.h>
#include <SPI.h>
#include <mcp_can.h>
#include <EEPROM.h>

#define MY_ID_ADDR      0
#define CFG_ADDR        1

struct NodeConfig {
  uint8_t nodeId;
  uint8_t servoCnt;
  uint8_t jointCnt;
  uint16_t canIds[2];
};

NodeConfig cfg;

#define CAN_CS   10
#define RS485_DE 2

MCP_CAN CAN(CAN_CS);

#define MAX_SERVOS 6
Servo servos[MAX_SERVOS];
const uint8_t servoPins[MAX_SERVOS] = {3, 5, 6, 9, 7, 8};

enum RxState { WAIT_SOF, PAYLOAD, CHKSUM, EOF_STATE };
RxState state = WAIT_SOF;

uint8_t payload[5];
uint8_t payloadIdx = 0;
uint8_t receivedChksum = 0;

template <typename T>
inline T clamp(T val, T lo, T hi) {
  return (val < lo) ? lo : ((val > hi) ? hi : val);
}

uint8_t crc8(const uint8_t *buf, uint8_t len) {
  uint8_t crc = 0x00;
  while (len--) {
    crc ^= *buf++;
    for (uint8_t i = 0; i < 8; ++i) {
      crc = (crc & 0x80) ? ((crc << 1) ^ 0x07) : (crc << 1);
    }
  }
  return crc;
}

bool canSend(uint16_t id, const uint8_t *data, uint8_t len) {
  for (uint8_t retry = 0; retry < 5; ++retry) {
    if (CAN.sendMsgBuf(id, 0, len, (byte*)data) == CAN_OK) return true;
    delay(5);
  }
  return false;
}

inline void setServo(uint8_t idx, int16_t pulse) {
  if (idx >= cfg.servoCnt) return;
  pulse = clamp<int16_t>(pulse, 1000, 2000);
  servos[idx].writeMicroseconds(pulse);
}

inline void setJoint(uint8_t idx, int16_t cmd) {
  if (idx >= cfg.jointCnt) return;
  uint16_t canId = cfg.canIds[idx];
  if (!canId) return;
  uint8_t buf[8] = {0};
  buf[0] = (uint8_t)(cmd >> 8);
  buf[1] = (uint8_t)(cmd & 0xFF);
  canSend(canId, buf, 8);
}

inline void rs485_tx() { digitalWrite(RS485_DE, HIGH); }
inline void rs485_rx() { digitalWrite(RS485_DE, LOW); }

void send_telemetry_to_master() {
  uint8_t payload[8];
  payload[0] = cfg.nodeId;
  payload[1] = 0xAF;

  int16_t angle = 1500;   // TODO: Replace with real encoder read
  int16_t torque = 0;     // TODO: Replace with current sense
  uint8_t temp = 35;
  uint8_t error = 0;

  payload[2] = (angle >> 8) & 0xFF;
  payload[3] = angle & 0xFF;
  payload[4] = (torque >> 8) & 0xFF;
  payload[5] = torque & 0xFF;
  payload[6] = temp;
  payload[7] = error;

  uint8_t chk = crc8(payload, 8);

  rs485_tx();
  Serial.write(0xAA);
  Serial.write(payload, 8);
  Serial.write(chk);
  Serial.write(0x55);
  Serial.flush();
  rs485_rx();
}

void setup() {
  Serial.begin(115200);
  pinMode(RS485_DE, OUTPUT);
  rs485_rx();

  EEPROM.get(MY_ID_ADDR, cfg.nodeId);
  if (cfg.nodeId < 1 || cfg.nodeId > 12) {
    cfg.nodeId = 1;
    cfg.servoCnt = 2;
    cfg.jointCnt = 1;
    cfg.canIds[0] = 0x201;
    cfg.canIds[1] = 0;
  } else {
    EEPROM.get(CFG_ADDR, cfg.servoCnt);
    EEPROM.get(CFG_ADDR + 1, cfg.jointCnt);
    EEPROM.get(CFG_ADDR + 2, cfg.canIds[0]);
    EEPROM.get(CFG_ADDR + 4, cfg.canIds[1]);
  }

  for (uint8_t i = 0; i < min(cfg.servoCnt, (uint8_t)MAX_SERVOS); i++) {
    servos[i].attach(servoPins[i]);
    servos[i].writeMicroseconds(1500);
  }

  if (cfg.jointCnt > 0) {
    while (CAN.begin(MCP_ANY, CAN_1MBPS, MCP_16MHZ) != CAN_OK) {
      delay(200);
    }
    CAN.setMode(MCP_NORMAL);
  }
}

unsigned long lastTelemetry = 0;

void loop() {
  while (Serial.available()) {
    uint8_t byteIn = Serial.read();
    switch (state) {
      case WAIT_SOF:
        if (byteIn == 0xAA) { state = PAYLOAD; payloadIdx = 0; }
        break;
      case PAYLOAD:
        payload[payloadIdx++] = byteIn;
        if (payloadIdx == 5) state = CHKSUM;
        break;
      case CHKSUM:
        receivedChksum = byteIn;
        state = EOF_STATE;
        break;
      case EOF_STATE:
        if (byteIn == 0x55) {
          uint8_t chk = crc8(payload, 5);
          if (chk == receivedChksum && payload[0] == cfg.nodeId) {
            uint8_t type = payload[1];
            uint8_t idx = payload[2];
            int16_t val = ((int16_t)payload[3] << 8) | payload[4];
            if (type == 0x01) setJoint(idx, val);
            else if (type == 0x02) setServo(idx, val);
          }
        }
        state = WAIT_SOF;
        break;
    }
  }

  if (millis() - lastTelemetry >= 100) {
    send_telemetry_to_master();
    lastTelemetry = millis();
  }
}