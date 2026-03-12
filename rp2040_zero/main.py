from machine import Pin
import time


SHOT_PIN = 0
DEBOUNCE_MS = 30

shot_switch = Pin(SHOT_PIN, Pin.IN, Pin.PULL_DOWN)
last_stable = shot_switch.value()
last_reading = last_stable
last_change_ms = time.ticks_ms()

try:
    led = Pin("LED", Pin.OUT)
except Exception:
    led = None


def set_led(on):
    if led is not None:
        led.value(1 if on else 0)


def send_shot():
    print("SHOT")


print("RP2040 shot switch ready")
print("SHOT_PIN =", SHOT_PIN)
print("Initial state =", last_stable)
set_led(False)

while True:
    now = time.ticks_ms()
    reading = shot_switch.value()

    if reading != last_reading:
        last_reading = reading
        last_change_ms = now

    if time.ticks_diff(now, last_change_ms) >= DEBOUNCE_MS and reading != last_stable:
        last_stable = reading

        if last_stable == 0:
            print("BUTTON DOWN")
            set_led(True)
            send_shot()
        else:
            print("BUTTON UP")
            set_led(False)

    time.sleep_ms(1)
