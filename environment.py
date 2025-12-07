import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RhythmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, notes, frame_ms=16, hit_window=100):
        """
        notes : (x, y, time) 리스트 (ms 단위)
        frame_ms : 시뮬레이션 시간 단위 (한 step = frame_ms ms)
        hit_window : 노트를 성공으로 간주할 타이밍 오차 (±ms)
        """
        super(RhythmEnv, self).__init__()

        self.notes = notes[:200]  # 처음 200개 노트만 사용
        self.max_time = self.notes[-1][2] + 1000

        self.frame_ms = frame_ms
        self.hit_window = hit_window

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0  # ms
        self.note_index = 0
        self.combo = 0
        self.score = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_time += self.frame_ms
        reward = 0.0
        done = False

        # 현재 노트 판정: 누를 타이밍 근처면 성공
        if self.note_index < len(self.notes):
            x, y, note_time = self.notes[self.note_index]
            time_diff = note_time - self.current_time

            # 노트 타이밍이 현재 시점과 가깝다면
            if abs(time_diff) <= self.hit_window:
                if action == 1:
                    # 정타 성공
                    reward += max(0, 1 - abs(time_diff) / self.hit_window)
                    self.combo += 1
                    self.score += reward
                    self.note_index += 1
                elif action == 0 and time_diff < -self.hit_window:
                    # 타이밍 놓침
                    reward -= 0.5
                    self.combo = 0
                    self.note_index += 1

        # 종료 조건: 모든 노트를 처리했거나 시간 초과
        if self.current_time >= self.max_time:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        """
        현재 시점 상태를 반환.
        다음 노트까지 남은 시간, x, y, combo(정규화).
        """
        if self.note_index < len(self.notes):
            x, y, t = self.notes[self.note_index]
            dt = max(0, t - self.current_time)
            dt_norm = np.clip(dt / 5000.0, 0, 1)
            obs = np.array([dt_norm, x / 512, y / 384, min(self.combo / 100, 1)], dtype=np.float32)
        else:
            obs = np.zeros(4, dtype=np.float32)
        return obs

    def render(self):
        print(f"t={self.current_time} combo={self.combo} score={self.score}")

    def close(self):
        pass
