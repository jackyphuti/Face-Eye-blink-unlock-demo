import json
import time
from unittest.mock import MagicMock, patch
import pytest

from hardware import (
    HardwareConfig,
    HardwareManager,
    DummyDriver,
    SerialArduinoDriver,
    WebhookDriver,
    RPiGPIODriver,
)


def test_dummy_driver_unlock_cycle():
    driver = DummyDriver(log_events=False)
    assert driver.is_locked is True

    # Unlock for a short duration
    driver.unlock("Alice", duration=0.05)
    assert driver.is_locked is False

    # Wait for non-blocking lock timer to expire
    time.sleep(0.08)
    assert driver.is_locked is True

    # Indicators should run without errors
    driver.indicate_sensing()
    driver.indicate_blink(2)
    driver.indicate_deny()
    driver.cleanup()


def test_hardware_manager_dispatch():
    cfg = HardwareConfig()
    hw = HardwareManager(cfg)
    assert len(hw.drivers) >= 1

    # Verify calls pass through cleanly
    hw.unlock("Bob", duration=0.1)
    hw.indicate_sensing()
    hw.indicate_blink(1)
    hw.indicate_deny()
    hw.lock()

    # Button callback
    button_pressed = []
    hw.register_button_callback(lambda: button_pressed.append(True))
    hw.cleanup()


def test_serial_arduino_driver_protocol():
    with patch("hardware.serial") as mock_serial_module, patch("hardware.HAS_SERIAL", True):
        mock_conn = MagicMock()
        mock_conn.is_open = True
        mock_serial_module.Serial.return_value = mock_conn

        ard = SerialArduinoDriver(port="COM3", baud=115200)
        assert ard.setup() is True

        # Test unlock command formatting
        ard.unlock("Alice", duration=2.5)
        mock_conn.write.assert_called_with(b"UNLOCK:Alice:2500\n")

        # Test lock command
        ard.lock()
        mock_conn.write.assert_called_with(b"LOCK\n")

        # Test blink command
        ard.indicate_blink(3)
        mock_conn.write.assert_called_with(b"BLINK:3\n")

        # Test deny command
        ard.indicate_deny()
        mock_conn.write.assert_called_with(b"DENY\n")

        # Test sensing command
        ard.indicate_sensing()
        mock_conn.write.assert_called_with(b"STATUS:SENSING\n")

        ard.cleanup()


def test_webhook_driver_dispatch():
    with patch("urllib.request.urlopen") as mock_urlopen, patch("hardware.HAS_HTTP", True):
        driver = WebhookDriver(url="http://example.com/webhook")
        driver.unlock("Alice", duration=3.0)

        # Allow daemon thread to run
        time.sleep(0.05)
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://example.com/webhook"
        data = json.loads(req.data.decode("utf-8"))
        assert data["user"] == "Alice"
        assert data["event"] == "unlock"


def test_rpi_gpio_driver_graceful_on_pc():
    # On non-Pi systems without RPi.GPIO, setup returns False
    cfg = HardwareConfig(enable_gpio=True)
    driver = RPiGPIODriver(cfg)
    # If not on Raspberry Pi, setup returns False gracefully
    res = driver.setup()
    if not res:
        # Safe no-ops
        driver.unlock("Alice")
        driver.lock()
        driver.cleanup()
