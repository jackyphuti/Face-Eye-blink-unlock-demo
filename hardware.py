"""Hardware interface module for Raspberry Pi GPIO, Arduino (Serial), and physical biometrics.

Supports:
- Raspberry Pi GPIO (Relays, Solenoid Locks, Status LEDs, Piezo Buzzers, Push Buttons)
- Arduino / Microcontroller Serial communication (USB UART)
- Smart Home Webhook / MQTT / HTTP alerts
- Automatic fallback to simulated mock hardware driver on non-embedded systems
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

logger = logging.getLogger("hardware")

# Try to import Raspberry Pi GPIO libraries
try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    GPIO = None
    HAS_RPI_GPIO = False

# Try to import pyserial
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    serial = None
    HAS_SERIAL = False

# Try to import urllib for webhooks
try:
    import urllib.request
    import json
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False


@dataclass
class HardwareConfig:
    enable_gpio: bool = False
    relay_pin: int = 18          # BCM 18: Electronic Door Strike / Relay
    led_green_pin: Optional[int] = 23   # BCM 23: Access Granted Indicator
    led_red_pin: Optional[int] = 24     # BCM 24: Locked / Denied Indicator
    led_blue_pin: Optional[int] = 22    # BCM 22: Face Detected / Challenge Indicator
    buzzer_pin: Optional[int] = 25      # BCM 25: Piezo Buzzer
    button_pin: Optional[int] = 17      # BCM 17: Physical Exit / Doorbell Button
    unlock_duration: float = 3.0        # Seconds to keep lock energized
    active_low_relay: bool = True       # True if relay energizes on LOW (common for 5V relay modules)
    serial_port: Optional[str] = None   # Arduino Serial device (e.g., /dev/ttyACM0 or COM3)
    serial_baud: int = 115200           # Baud rate for Arduino
    webhook_url: Optional[str] = None   # HTTP webhook on unlock


class BaseHardwareDriver:
    """Base class for all hardware biometric drivers."""

    def setup(self) -> bool:
        return True

    def unlock(self, name: str, duration: float = 3.0) -> None:
        pass

    def lock(self) -> None:
        pass

    def indicate_sensing(self) -> None:
        pass

    def indicate_blink(self, count: int) -> None:
        pass

    def indicate_deny(self) -> None:
        pass

    def register_button_callback(self, callback: Callable[[], None]) -> None:
        pass

    def cleanup(self) -> None:
        pass


class DummyDriver(BaseHardwareDriver):
    """Simulated hardware driver for development and non-embedded platforms."""

    def __init__(self, log_events: bool = True) -> None:
        self.log_events = log_events
        self.is_locked = True
        self._lock_timer: Optional[threading.Timer] = None

    def unlock(self, name: str, duration: float = 3.0) -> None:
        self.is_locked = False
        if self.log_events:
            print(f"[HARDWARE-SIM] >>> RELAY ENERGIZED: Door UNLOCKED for '{name}' ({duration:.1f}s) <<<")

        if self._lock_timer and self._lock_timer.is_alive():
            self._lock_timer.cancel()

        self._lock_timer = threading.Timer(duration, self.lock)
        self._lock_timer.daemon = True
        self._lock_timer.start()

    def lock(self) -> None:
        self.is_locked = True
        if self.log_events:
            print("[HARDWARE-SIM] Relay released: Door LOCKED")

    def indicate_sensing(self) -> None:
        if self.log_events:
            print("[HARDWARE-SIM] LED: Sensing/Face Detected (Blue ON)")

    def indicate_blink(self, count: int) -> None:
        if self.log_events:
            print(f"[HARDWARE-SIM] Buzzer/LED: Blink #{count} registered (Short Beep)")

    def indicate_deny(self) -> None:
        if self.log_events:
            print("[HARDWARE-SIM] LED/Buzzer: ACCESS DENIED (Red ON, Long Buzz)")

    def cleanup(self) -> None:
        if self._lock_timer and self._lock_timer.is_alive():
            self._lock_timer.cancel()
        self.lock()


class RPiGPIODriver(BaseHardwareDriver):
    """Direct Raspberry Pi GPIO driver using RPi.GPIO."""

    def __init__(self, cfg: HardwareConfig) -> None:
        self.cfg = cfg
        self._lock_timer: Optional[threading.Timer] = None
        self._button_cb: Optional[Callable[[], None]] = None

    def setup(self) -> bool:
        if not HAS_RPI_GPIO:
            print("Warning: RPi.GPIO is not available on this platform. Falling back to simulated driver.")
            return False

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Relay output
            relay_initial = GPIO.HIGH if self.cfg.active_low_relay else GPIO.LOW
            GPIO.setup(self.cfg.relay_pin, GPIO.OUT, initial=relay_initial)

            # LEDs
            for pin in (self.cfg.led_green_pin, self.cfg.led_red_pin, self.cfg.led_blue_pin, self.cfg.buzzer_pin):
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

            # Red LED on by default to indicate locked state
            if self.cfg.led_red_pin is not None:
                GPIO.output(self.cfg.led_red_pin, GPIO.HIGH)

            # Button input with internal pull-up
            if self.cfg.button_pin is not None:
                GPIO.setup(self.cfg.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            print(f"[Raspberry Pi GPIO] Initialized: Relay=BCM{self.cfg.relay_pin}, "
                  f"LEDs(G/R/B)={self.cfg.led_green_pin}/{self.cfg.led_red_pin}/{self.cfg.led_blue_pin}, "
                  f"Buzzer={self.cfg.buzzer_pin}")
            return True
        except Exception as e:
            print(f"Error configuring RPi GPIO: {e}", file=sys.stderr)
            return False

    def unlock(self, name: str, duration: float = 3.0) -> None:
        if not HAS_RPI_GPIO:
            return

        # Energize relay
        relay_on = GPIO.LOW if self.cfg.active_low_relay else GPIO.HIGH
        GPIO.output(self.cfg.relay_pin, relay_on)

        # Status LEDs
        if self.cfg.led_green_pin is not None:
            GPIO.output(self.cfg.led_green_pin, GPIO.HIGH)
        if self.cfg.led_red_pin is not None:
            GPIO.output(self.cfg.led_red_pin, GPIO.LOW)
        if self.cfg.led_blue_pin is not None:
            GPIO.output(self.cfg.led_blue_pin, GPIO.LOW)

        # Non-blocking lock timeout
        if self._lock_timer and self._lock_timer.is_alive():
            self._lock_timer.cancel()

        self._lock_timer = threading.Timer(duration, self.lock)
        self._lock_timer.daemon = True
        self._lock_timer.start()

    def lock(self) -> None:
        if not HAS_RPI_GPIO:
            return

        # De-energize relay
        relay_off = GPIO.HIGH if self.cfg.active_low_relay else GPIO.LOW
        GPIO.output(self.cfg.relay_pin, relay_off)

        # Status LEDs
        if self.cfg.led_green_pin is not None:
            GPIO.output(self.cfg.led_green_pin, GPIO.LOW)
        if self.cfg.led_red_pin is not None:
            GPIO.output(self.cfg.led_red_pin, GPIO.HIGH)
        if self.cfg.led_blue_pin is not None:
            GPIO.output(self.cfg.led_blue_pin, GPIO.LOW)

    def indicate_sensing(self) -> None:
        if not HAS_RPI_GPIO:
            return
        if self.cfg.led_blue_pin is not None:
            GPIO.output(self.cfg.led_blue_pin, GPIO.HIGH)

    def indicate_blink(self, count: int) -> None:
        if not HAS_RPI_GPIO:
            return

        def _pulse():
            if self.cfg.buzzer_pin is not None:
                GPIO.output(self.cfg.buzzer_pin, GPIO.HIGH)
            time.sleep(0.06)
            if self.cfg.buzzer_pin is not None:
                GPIO.output(self.cfg.buzzer_pin, GPIO.LOW)

        threading.Thread(target=_pulse, daemon=True).start()

    def indicate_deny(self) -> None:
        if not HAS_RPI_GPIO:
            return

        def _buzz():
            for _ in range(3):
                if self.cfg.buzzer_pin is not None:
                    GPIO.output(self.cfg.buzzer_pin, GPIO.HIGH)
                time.sleep(0.1)
                if self.cfg.buzzer_pin is not None:
                    GPIO.output(self.cfg.buzzer_pin, GPIO.LOW)
                time.sleep(0.05)

        threading.Thread(target=_buzz, daemon=True).start()

    def register_button_callback(self, callback: Callable[[], None]) -> None:
        if not HAS_RPI_GPIO or self.cfg.button_pin is None:
            return
        self._button_cb = callback

        def _on_event(channel):
            if self._button_cb:
                self._button_cb()

        try:
            GPIO.add_event_detect(
                self.cfg.button_pin,
                GPIO.FALLING,
                callback=_on_event,
                bouncetime=300,
            )
        except Exception as e:
            print(f"Warning: Could not register GPIO button interrupt: {e}")

    def cleanup(self) -> None:
        if not HAS_RPI_GPIO:
            return
        try:
            self.lock()
            GPIO.cleanup()
        except Exception:
            pass


class SerialArduinoDriver(BaseHardwareDriver):
    """Communicates with Arduino / ESP32 microcontroller over USB Serial."""

    def __init__(self, port: str, baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self.serial_conn: Optional[Any] = None
        self._stop_reader = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._button_cb: Optional[Callable[[], None]] = None

    def setup(self) -> bool:
        if not HAS_SERIAL:
            print(f"Warning: 'pyserial' not installed. Cannot open serial port '{self.port}'.")
            return False

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.0)  # Allow Arduino bootloader to initialize
            print(f"[Arduino Serial] Connected to microcontroller on {self.port} at {self.baud} baud.")

            # Start background reader for incoming events (e.g. physical buttons or motion)
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            print(f"Error opening serial port '{self.port}': {e}", file=sys.stderr)
            return False

    def _send_command(self, cmd: str) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            try:
                line = (cmd.strip() + "\n").encode("utf-8")
                self.serial_conn.write(line)
                self.serial_conn.flush()
            except Exception as e:
                print(f"Serial write error: {e}", file=sys.stderr)

    def _read_loop(self) -> None:
        while not self._stop_reader.is_set() and self.serial_conn and self.serial_conn.is_open:
            try:
                raw = self.serial_conn.readline()
                if raw:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line == "EVENT:BUTTON" and self._button_cb:
                        self._button_cb()
                    elif line:
                        logger.debug(f"[Arduino] {line}")
            except Exception:
                break

    def unlock(self, name: str, duration: float = 3.0) -> None:
        # Protocol: UNLOCK:<name>:<duration_ms>
        ms = int(duration * 1000)
        self._send_command(f"UNLOCK:{name}:{ms}")

    def lock(self) -> None:
        self._send_command("LOCK")

    def indicate_sensing(self) -> None:
        self._send_command("STATUS:SENSING")

    def indicate_blink(self, count: int) -> None:
        self._send_command(f"BLINK:{count}")

    def indicate_deny(self) -> None:
        self._send_command("DENY")

    def register_button_callback(self, callback: Callable[[], None]) -> None:
        self._button_cb = callback

    def cleanup(self) -> None:
        self._stop_reader.set()
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.lock()
                self.serial_conn.close()
            except Exception:
                pass


class WebhookDriver(BaseHardwareDriver):
    """Dispatches HTTP webhook notifications on unlock."""

    def __init__(self, url: str) -> None:
        self.url = url

    def unlock(self, name: str, duration: float = 3.0) -> None:
        if not HAS_HTTP or not self.url:
            return

        def _dispatch():
            try:
                payload = json.dumps({
                    "event": "unlock",
                    "user": name,
                    "duration": duration,
                    "timestamp": time.time(),
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    pass
            except Exception as e:
                print(f"Warning: Webhook dispatch to {self.url} failed: {e}")

        threading.Thread(target=_dispatch, daemon=True).start()


class HardwareManager:
    """Coordinates physical actuators and sensory feedback across active drivers."""

    def __init__(self, cfg: HardwareConfig) -> None:
        self.cfg = cfg
        self.drivers: List[BaseHardwareDriver] = []
        self._is_active = False

        # Initialize Raspberry Pi GPIO driver if requested
        if cfg.enable_gpio:
            rpi_driver = RPiGPIODriver(cfg)
            if rpi_driver.setup():
                self.drivers.append(rpi_driver)
            else:
                self.drivers.append(DummyDriver())
        elif cfg.serial_port:
            pass
        else:
            # Simulated driver for test/development
            self.drivers.append(DummyDriver())

        # Initialize Arduino Serial driver if port configured
        if cfg.serial_port:
            ard_driver = SerialArduinoDriver(cfg.serial_port, cfg.serial_baud)
            if ard_driver.setup():
                self.drivers.append(ard_driver)

        # Initialize Webhook driver if configured
        if cfg.webhook_url:
            self.drivers.append(WebhookDriver(cfg.webhook_url))

        self._is_active = len(self.drivers) > 0

    def unlock(self, name: str, duration: Optional[float] = None) -> None:
        dur = duration if duration is not None else self.cfg.unlock_duration
        for d in self.drivers:
            d.unlock(name, dur)

    def lock(self) -> None:
        for d in self.drivers:
            d.lock()

    def indicate_sensing(self) -> None:
        for d in self.drivers:
            d.indicate_sensing()

    def indicate_blink(self, count: int) -> None:
        for d in self.drivers:
            d.indicate_blink(count)

    def indicate_deny(self) -> None:
        for d in self.drivers:
            d.indicate_deny()

    def register_button_callback(self, callback: Callable[[], None]) -> None:
        for d in self.drivers:
            d.register_button_callback(callback)

    def cleanup(self) -> None:
        for d in self.drivers:
            d.cleanup()
