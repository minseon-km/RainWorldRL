# -*- coding: utf-8 -*-

import socket
import struct
import numpy as np
import time

HOST = '127.0.0.1'
PORT = 50000
DATA_SIZE = 8 # 전송받을 float 개수 (8개)
FLOAT_BYTE_SIZE = 4 # float 하나당 4바이트

def connect_to_server():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Python] Attempting to connect to C# server on {HOST}:{PORT}...")
    while True:
        try:
            client.connect((HOST, PORT))
            print("[Python] Connection successful!")
            return client
        except ConnectionRefusedError:
            print("Connection refused. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"An unexpected error occurred during connection: {e}")
            time.sleep(2)

def receive_all(sock, n):
    """지정된 바이트 수(n)를 모두 수신합니다."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def receive_data(client):
    """데이터를 수신하고 확인 신호를 보냅니다."""
    try:
        # 1. 4바이트 길이 헤더 수신 (Int32)
        length_header = receive_all(client, 4)
        if length_header is None:
            raise ConnectionAbortedError("Server disconnected during header receive.")

        # C#의 Int32(4바이트)를 Python 정수로 변환
        payload_length = struct.unpack('<I', length_header)[0] 
        
        # 예상되는 데이터 크기가 맞는지 확인
        if payload_length != DATA_SIZE * FLOAT_BYTE_SIZE:
            print(f"Warning: Unexpected payload size received: {payload_length} bytes.")
            
        # 2. 실제 데이터 (float 배열) 수신
        float_data_bytes = receive_all(client, payload_length)
        if float_data_bytes is None:
            raise ConnectionAbortedError("Server disconnected during payload receive.")

        # 바이트 배열을 float 배열로 변환
        # <f: little-endian float, DATA_SIZE: 개수
        received_floats = struct.unpack(f'<{DATA_SIZE}f', float_data_bytes)
        
        # 3. 받은 데이터를 별도로 저장 (강화 학습의 Observation)
        observation = np.array(received_floats)
        
        return observation

    except ConnectionAbortedError:
        print("\n[Python] C# server connection closed. Exiting.")
        return
    except Exception as e:
        print(f"\n[Python] An error occurred in the loop: {e}. Exiting.")
        return
        
def send_data(client, action):
    try:
        # 4. 행동 계산 (강화 학습 무시, 1 전송)
        
        # C#에 4바이트 정수 (int)로 응답 전송
        confirmation_bytes = struct.pack('<i', action) 
        client.sendall(confirmation_bytes)
        
    except ConnectionAbortedError:
        print("\n[Python] C# server connection closed. Exiting.")
        return
    except Exception as e:
        print(f"\n[Python] An error occurred in the loop: {e}. Exiting.")
        return


def main_connector() :
    # 1. Python 스크립트 실행
    # 2. C# 서버에 연결을 시도
    client_socket = connect_to_server() 
    return client_socket
    # 3. 메인 통신 루프 시작
    # run_rl_loop(client_socket)
    
    # client_socket.close()