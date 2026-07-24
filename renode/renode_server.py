import socket
import json
import time

class RenodeTargetEmulatorServer:
    """
    TCP Socket Server modeling an emulated ARM Cortex-M4 microcontroller running FreeRTOS.
    Implements the Renode external telemetry plugin interface on port 4000.
    """
    def __init__(self, host='127.0.0.1', port=4000):
        self.host = host
        self.port = port
        self.current_freq_mhz = 8.0
        self.current_voltage_v = 0.9
        self.total_cycles = 0
        self.ram_used_bytes = 1840  # Static SRAM footprint (FreeRTOS TCBs + stack)

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        server_sock.settimeout(10.0)
        
        print(f"[Renode MCU Emulator] Server listening on {self.host}:{self.port}...")
        
        try:
            client_sock, addr = server_sock.accept()
            print(f"[Renode MCU Emulator] Connection established from {addr}")
            rfile = client_sock.makefile('r', encoding='utf-8')
            
            while True:
                line = rfile.readline()
                if not line:
                    break
                try:
                    cmd = json.loads(line.strip())
                    if cmd.get('command') == 'set_frequency':
                        self.current_freq_mhz = float(cmd.get('frequency_mhz', 8.0))
                        self.current_voltage_v = float(cmd.get('voltage_v', 0.9))
                        
                        # Accumulate 100ms control step cycles
                        cycles_step = int(self.current_freq_mhz * 1e6 * 0.1)
                        self.total_cycles += cycles_step
                        
                        response = json.dumps({
                            'status': 'OK',
                            'current_freq_mhz': self.current_freq_mhz,
                            'current_voltage_v': self.current_voltage_v,
                            'step_cycles': cycles_step,
                            'total_cycles': self.total_cycles,
                            'ram_used_bytes': self.ram_used_bytes
                        }) + '\n'
                        client_sock.sendall(response.encode('utf-8'))
                except Exception as e:
                    err_resp = json.dumps({'status': 'ERROR', 'message': str(e)}) + '\n'
                    client_sock.sendall(err_resp.encode('utf-8'))
                    
            client_sock.close()
            print("[Renode MCU Emulator] Co-simulation session finished successfully.")
        except socket.timeout:
            print("[Renode MCU Emulator] Server timed out waiting for connection.")
        finally:
            server_sock.close()

if __name__ == "__main__":
    emulator = RenodeTargetEmulatorServer()
    emulator.run()
