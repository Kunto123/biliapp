import sys

PIN_SWITCH = 8  # BCM 8 - limit switch
PIN_FLASH  = 7  # BCM 7 - flash LED

print("=" * 50)
print("GPIO DIAGNOSTIK - Bilirubin App")
print("=" * 50)

# --- Cek library ---
print("\n[1] Cek library GPIO:")
libs = {}
for name in ["RPi.GPIO", "lgpio", "gpiozero"]:
    try:
        mod = __import__(name.replace(".", "_") if name == "RPi.GPIO" else name)
        libs[name] = "OK"
        print(f"  {name}: TERSEDIA")
    except ImportError:
        libs[name] = "TIDAK ADA"
        print(f"  {name}: TIDAK ADA")

# --- Coba RPi.GPIO ---
print(f"\n[2] Coba init RPi.GPIO di BCM {PIN_SWITCH} (switch) dan BCM {PIN_FLASH} (flash):")
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(PIN_FLASH,  GPIO.OUT, initial=GPIO.LOW)

    switch_val = GPIO.input(PIN_SWITCH)
    print(f"  Init OK")
    print(f"  BCM {PIN_SWITCH} (switch) = {switch_val}  ({'HIGH/idle' if switch_val else 'LOW/pressed'})")
    print(f"  BCM {PIN_FLASH} (flash)  = OUTPUT siap")

    # Test flash
    print(f"\n[3] Test flash LED (BCM {PIN_FLASH}):")
    import time
    GPIO.output(PIN_FLASH, GPIO.HIGH)
    print(f"  >> Flash ON  (cek apakah LED menyala)")
    time.sleep(1)
    GPIO.output(PIN_FLASH, GPIO.LOW)
    print(f"  >> Flash OFF")

    GPIO.cleanup()
    print("\n  HASIL: RPi.GPIO berfungsi normal")

except ImportError:
    print("  RPi.GPIO tidak terinstall")
except Exception as e:
    print(f"  ERROR: {e}")
    print(f"  Kemungkinan: konflik SPI, pin salah, atau RPi 5 tidak kompatibel")

# --- Coba lgpio sebagai alternatif ---
print(f"\n[4] Coba lgpio (alternatif untuk RPi 5):")
try:
    import lgpio
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(h, PIN_SWITCH, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_output(h, PIN_FLASH)

    switch_val = lgpio.gpio_read(h, PIN_SWITCH)
    print(f"  Init OK")
    print(f"  BCM {PIN_SWITCH} (switch) = {switch_val}  ({'HIGH/idle' if switch_val else 'LOW/pressed'})")

    lgpio.gpio_free(h, PIN_SWITCH)
    lgpio.gpio_free(h, PIN_FLASH)
    lgpio.gpiochip_close(h)
    print("  HASIL: lgpio berfungsi normal")
except ImportError:
    print("  lgpio tidak terinstall")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 50)
print("Selesai. Kirim output ini untuk analisa.")
print("=" * 50)
