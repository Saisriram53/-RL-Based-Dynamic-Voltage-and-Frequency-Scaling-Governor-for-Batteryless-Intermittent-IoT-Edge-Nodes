import socket
import json
import time

class RenodeCoSimBridge:
    """
    TCP Socket Bridge connecting the Python Gymnasium RL Governor
    to a Renode ARM Cortex-M4 Microcontroller Emulator running FreeRTOS.
    """
    def __init__(self, host='127.0.0.1', port=4000):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.rfile = self.sock.makefile('r', encoding='utf-8')
            print(f"[Co-Sim Bridge] Successfully connected to Renode at {self.host}:{self.port}")
        except Exception as e:
            print(f"[Co-Sim Bridge] Could not connect to Renode server at {self.host}:{self.port} ({e}). Operating in standalone PC simulation mode.")
            self.sock = None
            self.rfile = None

    def send_frequency_command(self, freq_mhz, voltage_v):
        """Sends CPU frequency scaling command to Renode virtual core."""
        if not self.sock:
            return
        payload = json.dumps({
            'command': 'set_frequency',
            'frequency_mhz': float(freq_mhz),
            'voltage_v': float(voltage_v)
        }) + '\n'
        try:
            self.sock.sendall(payload.encode('utf-8'))
        except Exception as e:
            print(f"[Co-Sim Bridge] Error sending payload: {e}")

    def receive_telemetry(self):
        """Receives active cycle counts and RAM footprint metrics from Renode with robust line framing."""
        if not self.rfile:
            return {'cycles': 0, 'ram_used_bytes': 0}
        
        try:
            line = self.rfile.readline()
            if not line:
                return {'cycles': 0, 'ram_used_bytes': 0}
            return json.loads(line.strip())
        except Exception:
            return {'cycles': 0, 'ram_used_bytes': 0}

    def close(self):
        if self.rfile:
            try:
                self.rfile.close()
            except Exception:
                pass
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            print("[Co-Sim Bridge] Socket connection closed.")

if __name__ == "__main__":
    bridge = RenodeCoSimBridge()
    bridge.connect()
    bridge.send_frequency_command(80.0, 1.5)
    metrics = bridge.receive_telemetry()
    print("Received Telemetry:", metrics)
    bridge.close()
