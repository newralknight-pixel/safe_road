import json
import os
import threading
import time

PICO_PORT_ENV = "SAFE_ROAD_PICO_PORT"
PICO_BAUD_ENV = "SAFE_ROAD_PICO_BAUD"
PICO_ENABLED_ENV = "SAFE_ROAD_PICO_ENABLED"
DEFAULT_BAUD = 115200

LAST_ALERT = {
    "ok": False,
    "message": "No alert sent yet.",
    "sent_at": None,
    "port": None,
}
LAST_ALERT_LOCK = threading.Lock()


def _serial_module():
    try:
        import serial
        from serial.tools import list_ports
    except ModuleNotFoundError:
        return None, None
    return serial, list_ports


def configured_port():
    return os.environ.get(PICO_PORT_ENV, "").strip()


def is_enabled():
    return os.environ.get(PICO_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def list_serial_ports():
    _, list_ports = _serial_module()
    if list_ports is None:
        return []
    ports = []
    for port in list_ports.comports():
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
            }
        )
    return ports


def pico_status():
    serial, _ = _serial_module()
    return {
        "enabled": is_enabled(),
        "configured_port": configured_port(),
        "baud": int(os.environ.get(PICO_BAUD_ENV, DEFAULT_BAUD)),
        "pyserial_available": serial is not None,
        "ports": list_serial_ports(),
        "last_alert": LAST_ALERT.copy(),
    }


def _set_last_alert(ok, message, port=None):
    with LAST_ALERT_LOCK:
        LAST_ALERT.update(
            {
                "ok": ok,
                "message": message,
                "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "port": port,
            }
        )


def _event_to_signal(event):
    return {
        "type": event.get("type", "unknown"),
        "label": event.get("label", "Road hazard"),
        "severity": event.get("severity", "medium"),
        "confidence": event.get("confidence", 0),
    }


def send_picoboard_alert(events):
    if not events:
        _set_last_alert(False, "No events to send.")
        return False
    if not is_enabled():
        _set_last_alert(False, "PicoBoard alert output is disabled.")
        return False

    port = configured_port()
    if not port:
        _set_last_alert(False, f"Set {PICO_PORT_ENV}=COMx before sending to PicoBoard.")
        return False

    serial, _ = _serial_module()
    if serial is None:
        _set_last_alert(False, "pyserial is not installed. Run: pip install pyserial", port)
        return False

    baud = int(os.environ.get(PICO_BAUD_ENV, DEFAULT_BAUD))
    signal = _event_to_signal(events[0])
    payload = f"SAFE_ROAD_ALERT {json.dumps(signal, separators=(',', ':'))}\n".encode("utf-8")

    try:
        with serial.Serial(port, baudrate=baud, timeout=1, write_timeout=1) as connection:
            connection.write(payload)
            connection.flush()
        _set_last_alert(True, f"Sent {signal['type']} alert to PicoBoard.", port)
        return True
    except Exception as error:
        _set_last_alert(False, f"Could not send PicoBoard alert: {error}", port)
        return False
