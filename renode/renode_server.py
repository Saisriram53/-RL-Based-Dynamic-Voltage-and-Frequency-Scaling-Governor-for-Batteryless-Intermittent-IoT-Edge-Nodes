import argparse
import socket
import json
import re
import time

class RenodeTargetEmulatorServer:
    """
    Dual-Protocol Renode Telnet Monitor & Socket Server.
    Supports both Renode Telnet Monitor commands (Port 1234 / 4000)
    and JSON telemetry payloads for cross-platform fallback execution.
    """
    def __init__(self, host='127.0.0.1', port=1234):
        self.host = host
        self.port = port
        self.current_freq_mhz = 8.0
        self.current_voltage_v = 0.9
        self.total_cycles = 1000
        self.sp_register = 0x200038D0  # FreeRTOS TCB stack pointer (1,840 bytes stack depth)

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        server_sock.settimeout(10.0)
        
        print(f"[Renode Telnet Monitor Server] Listening on {self.host}:{self.port}...")
        
        try:
            client_sock, addr = server_sock.accept()
            print(f"[Renode Telnet Monitor Server] Client connected from {addr}")
            client_sock.sendall(b"(stm32f4_dvfs) \n")
            rfile = client_sock.makefile('r', encoding='utf-8')
            
            while True:
                line = rfile.readline()
                if not line:
                    break
                line_str = line.strip()
                if not line_str:
                    continue
                
                # 1. Plaintext Renode Telnet Monitor Command Parsing
                if "PerformanceInMips" in line_str:
                    parts = line_str.split()
                    if len(parts) >= 2:
                        try:
                            self.current_freq_mhz = float(parts[-1])
                        except ValueError:
                            pass
                    self.total_cycles += int(self.current_freq_mhz * 1e6 * 0.1)
                    resp = f"PerformanceInMips set to {int(self.current_freq_mhz)}\n(stm32f4_dvfs) "
                    client_sock.sendall(resp.encode('utf-8'))
                elif "ExecutedInstructions" in line_str:
                    self.total_cycles += int(self.current_freq_mhz * 1e6 * 0.1)
                    resp = f"ExecutedInstructions: {self.total_cycles}\n(stm32f4_dvfs) "
                    client_sock.sendall(resp.encode('utf-8'))
                elif "SP" in line_str:
                    resp = f"SP: 0x{self.sp_register:08x}\n(stm32f4_dvfs) "
                    client_sock.sendall(resp.encode('utf-8'))
                else:
                    # 2. JSON Lines Protocol Fallback
                    try:
                        cmd = json.loads(line_str)
                        if cmd.get('command') == 'set_frequency':
                            self.current_freq_mhz = float(cmd.get('frequency_mhz', 8.0))
                            self.total_cycles += int(self.current_freq_mhz * 1e6 * 0.1)
                            resp = json.dumps({
                                'status': 'OK',
                                'current_freq_mhz': self.current_freq_mhz,
                                'total_cycles': self.total_cycles,
                                'sp_register': hex(self.sp_register),
                                'ram_used_bytes': 1840
                            }) + '\n'
                            client_sock.sendall(resp.encode('utf-8'))
                        else:
                            client_sock.sendall(b"(stm32f4_dvfs) \n")
                    except Exception:
                        client_sock.sendall(b"(stm32f4_dvfs) \n")
                    
            client_sock.close()
            print("[Renode Telnet Monitor Server] Session finished.")
        except socket.timeout:
            print("[Renode Telnet Monitor Server] Server timed out waiting for connection.")
        finally:
            server_sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renode Telnet Monitor / JSON emulator server")
    parser.add_argument("--port", type=int, default=1234,
                         help="Port to listen on (must match RenodeMonitorBridge's port in the caller)")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    emulator = RenodeTargetEmulatorServer(host=args.host, port=args.port)
    print(f"[Renode Telnet Monitor Server] Starting on {args.host}:{args.port}")
    emulator.run()
