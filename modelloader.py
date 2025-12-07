import torch
import numpy as np

def save_model(model, path, model_name="qnet", algorithm_name="DQN"):
    """
    주어진 모델의 state_dict를 지정된 경로에 저장합니다.
    
    :param model: 저장할 PyTorch 모델 인스턴스 (예: Qnet)
    :param path: 모델 파일을 저장할 경로 (폴더 경로)
    :param model_name: 저장할 파일 이름 (확장자 제외)
    """
    
    # 모델 파일 전체 경로 생성 (예: /path/to/models/qnet_weights.pth)
    full_path = f"{path}/{algorithm_name}_{model_name}_weights.pth"
    
    # 모델의 상태 딕셔너리(학습된 가중치)만 저장
    torch.save(model.state_dict(), full_path)
    print(f"Model weights saved to: {full_path}")

def save_score(scoredata, path, algorithm_name="DQN") :
    full_path = f"{path}/{algorithm_name}_scores.npy"
    np.save(full_path, np.array(scoredata))
    print(f"Model scores saved to: {full_path}")

# --- 사용 예시 ---
# save_model(q, "./models")

def load_model(model_instance, path, model_name="qnet", algorithm_name = "DQN"):
    """
    저장된 state_dict를 불러와 모델 인스턴스에 로드합니다.
    
    :param model_instance: 가중치를 로드할 빈 PyTorch 모델 인스턴스 (예: Qnet(5))
    :param path: 모델 파일이 있는 경로 (폴더 경로)
    :param model_name: 저장된 파일 이름 (확장자 제외)
    :return: 가중치가 로드된 모델 인스턴스
    """
    full_path = f"{path}/{algorithm_name}_{model_name}_weights.pth"
    
    if not torch.cuda.is_available():
        # CPU 환경에서 로드: 저장된 장치와 관계없이 CPU에 로드
        state_dict = torch.load(full_path, map_location=torch.device('cpu'))
    else:
        # GPU 환경에서 로드: GPU에 로드
        state_dict = torch.load(full_path)
        
    # 모델의 state_dict에 저장된 가중치를 로드합니다.
    model_instance.load_state_dict(state_dict)
    # model_instance.eval()  # 평가 모드로 설정
    print(f"Model weights loaded from: {full_path}")
    
    return model_instance

def load_scores(path, algorithm_type) :
    full_path = f"{path}/{algorithm_type}_scores.npy"
    return np.load(full_path)

# --- 사용 예시 ---
# loaded_q = load_model(Qnet(action_length), "./models")
