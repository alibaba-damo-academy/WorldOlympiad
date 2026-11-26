# server.py
import argparse
import socket
import threading

def handle_client(conn, addr):
    print(f"[+] client connected: {addr}")
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print(f"[-] client disconnected: {addr}")
                break

            text = data.decode("utf-8", errors="replace")
            print(f"[recv from {addr}] {text}")

            reply = f"echo from server: {text}".encode("utf-8")
            conn.sendall(reply)
    except Exception as e:
        print(f"[!] error with {addr}: {e}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(128)

    print(f"[*] listening on {args.host}:{args.port}")

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[*] server stopped")
    finally:
        server.close()

if __name__ == "__main__":
    main()