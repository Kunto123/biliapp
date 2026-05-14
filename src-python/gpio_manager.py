"""
gpio_manager.py

GPIO 8: limit switch input (BCM, active-LOW, pull-up)
  - 0 (LOW)  = capture requested
  - 1 (HIGH) = idle / re-arm next capture

GPIO 7: flash LED output (BCM)
  - HIGH = on (during capture)
  - LOW  = off (idle)

State machine:
  capture_ready=True   → capture allowed
  capture_ready=False  → blocked until switch returns HIGH

Falls back gracefully when RPi.GPIO is not installed (desktop / Windows).
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)

_PIN_SWITCH = 8   # BCM 8 — limit switch (input, pull-up)
_PIN_FLASH  = 7   # BCM 7 — flash LED (output)


class GPIOManager:
    def __init__(self):
        self._gpio = None
        self._available = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture_ready = True       # True = new capture allowed
        self._capture_triggered = False  # True = switch went LOW, awaiting capture
        self._init()

    # ── Init / cleanup ────────────────────────────────────────────────────

    def _init(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(_PIN_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(_PIN_FLASH,  GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self._available = True
            logger.info(
                f"[gpio] Initialized  switch=BCM{_PIN_SWITCH} (input)  "
                f"flash=BCM{_PIN_FLASH} (output)"
            )
        except ImportError:
            logger.info("[gpio] RPi.GPIO not available — GPIO disabled")
        except Exception as exc:
            logger.warning(f"[gpio] Init failed: {exc}")

    def start(self):
        """Start background monitor thread."""
        if not self._available:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gpio-monitor"
        )
        self._thread.start()
        logger.info("[gpio] Monitor thread started")

    def stop(self):
        """Stop monitor and clean up GPIO pins."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._available:
            try:
                self._gpio.output(_PIN_FLASH, self._gpio.LOW)
                self._gpio.cleanup()
            except Exception:
                pass
        logger.info("[gpio] Stopped")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def capture_ready(self) -> bool:
        with self._lock:
            return self._capture_ready

    def mark_captured(self):
        """
        Call when a capture starts.
        Blocks further captures until switch returns HIGH (GPIO 8 = 1).
        """
        with self._lock:
            self._capture_ready    = False
            self._capture_triggered = False
        logger.debug("[gpio] Capture started — re-arm blocked until switch HIGH")

    def consume_trigger(self):
        """Clear the pending trigger flag (call at the start of /api/capture)."""
        with self._lock:
            self._capture_triggered = False

    def set_flash(self, on: bool):
        """Turn flash LED on GPIO 7 on (True) or off (False)."""
        if not self._available:
            return
        try:
            self._gpio.output(
                _PIN_FLASH,
                self._gpio.HIGH if on else self._gpio.LOW,
            )
        except Exception as exc:
            logger.warning(f"[gpio] Flash control error: {exc}")

    def get_status(self) -> dict:
        switch_state = None
        if self._available:
            try:
                switch_state = int(self._gpio.input(_PIN_SWITCH))
            except Exception:
                pass
        with self._lock:
            return {
                "available":         self._available,
                "capture_ready":     self._capture_ready,
                "capture_triggered": self._capture_triggered,
                "switch_state":      switch_state,  # 0=pressed, 1=released, null=N/A
                "switch_pin":        _PIN_SWITCH,
                "flash_pin":         _PIN_FLASH,
            }

    # ── Monitor loop ──────────────────────────────────────────────────────

    def _loop(self):
        GPIO = self._gpio
        prev = GPIO.input(_PIN_SWITCH)  # read initial state to detect transitions only

        while self._running:
            try:
                curr = GPIO.input(_PIN_SWITCH)
                if curr != prev:
                    if curr == GPIO.LOW:        # switch pressed → capture intent
                        with self._lock:
                            if self._capture_ready:
                                self._capture_triggered = True
                                logger.info("[gpio] Switch LOW — capture triggered")
                    else:                       # switch released → re-arm
                        with self._lock:
                            self._capture_ready = True
                        logger.info("[gpio] Switch HIGH — capture re-armed")
                    prev = curr
                time.sleep(0.02)  # 20 ms polling / debounce
            except Exception as exc:
                logger.warning(f"[gpio] Monitor loop error: {exc}")
                time.sleep(0.1)


gpio_manager = GPIOManager()
