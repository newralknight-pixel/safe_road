from machine import Pin, PWM
import sys
import time

BUZZER_PIN = 15

TONES = {
    "roe_deer": (880, 0.18, 3),
    "trash": (660, 0.14, 2),
    "road_crack": (520, 0.16, 2),
    "pothole": (440, 0.22, 3),
    "unknown": (600, 0.12, 1),
}

led = Pin("LED", Pin.OUT)
buzzer = PWM(Pin(BUZZER_PIN))
buzzer.duty_u16(0)


def beep(hazard_type):
    frequency, duration, count = TONES.get(hazard_type, TONES["unknown"])
    for _ in range(count):
        led.on()
        buzzer.freq(frequency)
        buzzer.duty_u16(32000)
        time.sleep(duration)
        buzzer.duty_u16(0)
        led.off()
        time.sleep(0.08)


def hazard_type_from_line(line):
    for hazard_type in TONES:
        if hazard_type in line:
            return hazard_type
    return "unknown"


beep("unknown")

while True:
    line = sys.stdin.readline()
    if not line:
        time.sleep(0.05)
        continue
    if line.startswith("SAFE_ROAD_ALERT"):
        beep(hazard_type_from_line(line))
