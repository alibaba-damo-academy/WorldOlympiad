# client.py
import argparse
import socket

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--message", default="hello from client")
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=5) as s:
        print(f"[*] connected to {args.host}:{args.port}")
        s.sendall(args.message.encode("utf-8"))
        data = s.recv(4096)
        print("[*] server reply:", data.decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()