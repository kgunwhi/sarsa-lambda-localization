# === Imports ===
import os
import pygame
import numpy as np
import sys
import gym
import timestamp
from gym import spaces
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from sympy import false

plt.ion()
import random
import time
from datetime import datetime


# === SARSA(λ) Agent ===
class SARSAAgent:
    def __init__(self, n_rows, n_cols, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, lambda_=0.8):
        self.q_table = np.random.uniform(0, 0.01, size=(n_rows, n_cols, n_actions))
        self.e_trace = np.zeros((n_rows, n_cols, n_actions))  # eligibility traces
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        row, col = state
        return np.argmax(self.q_table[row, col])

    def update(self, state, action, reward, next_state, next_action):
        row, col = state
        next_row, next_col = next_state

        delta = reward + self.gamma * self.q_table[next_row, next_col, next_action] - self.q_table[row, col, action]
        self.e_trace[row, col, action] += 1
        self.q_table += self.alpha * delta * self.e_trace
        self.e_trace *= self.gamma * self.lambda_

    def reset_traces(self):
        self.e_trace.fill(0)

    def dynamic_decay_epsilon(self, episode):
        decay_rate = 0.995
        self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)


# === Environment ===
class ImageTileEnv(gym.Env):
    def __init__(self, image_path, target_path, n_tiles=3, window_size=800, true_target_tile=(2, 1), fixed_start = False):
        super().__init__()
        self.image_path = image_path
        self.target_path = target_path
        self.n_tiles = n_tiles
        self.window_size = window_size
        self.true_target_tile = true_target_tile 
        self.fixed_start = fixed_start

        pygame.init()
        self.clock = pygame.time.Clock()

        raw_image = pygame.image.load(self.image_path)
        scaled_size = window_size - (window_size % self.n_tiles)
        self.image = pygame.transform.scale(raw_image, (scaled_size, scaled_size))
        self.img_width, self.img_height = self.image.get_size()
        self.tile_width = self.img_width // self.n_tiles
        self.tile_height = self.img_height // self.n_tiles

        self.screen = pygame.display.set_mode((self.img_width, self.img_height))
        pygame.display.set_caption("Environment")

        self.tiles = []
        for row in range(self.n_tiles):
            for col in range(self.n_tiles):
                rect = pygame.Rect(col * self.tile_width, row * self.tile_height, self.tile_width, self.tile_height)
                tile = self.image.subsurface(rect).copy()
                self.tiles.append(tile)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=self.n_tiles - 1, shape=(2,), dtype=np.int32)

        self.reset_vars()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.target_feature = self.load_target_feature(self.target_path)
        self.sift = cv2.SIFT_create()

    def reset_vars(self):
        if self.fixed_start:
            self.window_pos = [1,1]
        else:
            self.window_pos = [np.random.randint(0, self.n_tiles), np.random.randint(0, self.n_tiles)]

        self.last_tile = tuple(self.window_pos)
        self.stay_count = 0
        self.mode = 'move'
        self.found_target = False
        self.truly_done = False
        self.stay_in_target_count = 0

    def reset(self):
        self.reset_vars()
        return np.array(self.window_pos, dtype=np.int32)

    def step(self, action):
        reward = 0.0
        done = False
        row, col = self.window_pos

        if self.mode == 'move':
            reward -= 1#1 #0.02
            if action == 0 and col < self.n_tiles - 1: col += 1
            elif action == 1 and row < self.n_tiles - 1: row += 1
            elif action == 2 and col > 0: col -= 1
            elif action == 3 and row > 0: row -= 1

            current_tile = (row, col)
            if current_tile == self.last_tile:
                self.stay_count += 1
                reward -= 1 * self.stay_count #1
            else:
                self.stay_count = 0
            self.last_tile = current_tile
            self.window_pos = [row, col]
            self.mode = 'decision'

        elif self.mode == 'decision':
            if action == 4:
                reward -= 1 #1
                sift_reward, match_percent, color_sim = self.perform_sift_localization()
                reward += sift_reward

                is_correct_tile = (tuple(self.window_pos) == self.true_target_tile)

                if is_correct_tile:
                    if not self.found_target:
                        self.found_target = True
                        reward += max(54, 62- step_count ** 1.5) #45 #10
                    self.stay_in_target_count += 1
                else:
                    self.stay_in_target_count = 0
            else:
                self.stay_in_target_count = 0
            self.mode = 'move'

        if self.found_target and self.stay_in_target_count >= 2 and not self.truly_done:
            reward += 65 #25
            done = True
            self.truly_done = True

        obs = np.array(self.window_pos, dtype=np.int32)
        return obs, reward, done, {}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        for idx, tile in enumerate(self.tiles):
            row = idx // self.n_tiles
            col = idx % self.n_tiles
            self.screen.blit(tile, (col * self.tile_width, row * self.tile_height))

        for i in range(1, self.n_tiles):
            pygame.draw.line(self.screen, (255, 255, 255), (i * self.tile_width, 0), (i * self.tile_width, self.img_height), 2)
            pygame.draw.line(self.screen, (255, 255, 255), (0, i * self.tile_height), (self.img_width, i * self.tile_height), 2)

        row, col = self.window_pos
        highlight_rect = pygame.Rect(col * self.tile_width, row * self.tile_height, self.tile_width, self.tile_height)
        pygame.draw.rect(self.screen, (0, 255, 0), highlight_rect, 4)

        pygame.display.flip()
        self.clock.tick(10)

    def extract_features(self, surface):
        data = pygame.surfarray.array3d(surface)
        data = np.transpose(data, (1, 0, 2))
        img = self.transform(data).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(img).squeeze().cpu().numpy()

    def load_target_feature(self, path):
        raw = pygame.image.load(path)
        raw = pygame.transform.scale(raw, (224, 224))
        arr = pygame.surfarray.array3d(raw)
        arr = np.transpose(arr, (1, 0, 2))
        img = self.transform(arr).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(img).squeeze().cpu().numpy()

    def perform_sift_localization(self):
        row, col = self.window_pos
        tile_idx = row * self.n_tiles + col
        tile_surface = self.tiles[tile_idx]
        data_tile = pygame.surfarray.array3d(tile_surface)
        data_tile = np.transpose(data_tile, (1, 0, 2))
        img_tile = cv2.cvtColor(data_tile, cv2.COLOR_RGB2BGR)

        raw_target = pygame.image.load(self.target_path)
        raw_target = pygame.transform.scale(raw_target, (224, 224))
        data_target = pygame.surfarray.array3d(raw_target)
        data_target = np.transpose(data_target, (1, 0, 2))
        img_target = cv2.cvtColor(data_target, cv2.COLOR_BGR2RGB)

        kp1, des1 = self.sift.detectAndCompute(img_target, None)
        kp2, des2 = self.sift.detectAndCompute(img_tile, None)

        if des1 is None or des2 is None:
            return -0.5, 0.0, 0.0

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = img_target.shape[:2]
                dst_box = cv2.perspectiveTransform(np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2), M)
                x_crop, y_crop, w_box, h_box = cv2.boundingRect(np.int32(dst_box))
                crop = img_tile[y_crop:y_crop+h_box, x_crop:x_crop+w_box]
                if crop.size == 0:
                    return -0.5, 0.0, 0.0

                resized = cv2.resize(crop, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb).unsqueeze(0)
                with torch.no_grad():
                    feat = self.resnet(tensor).squeeze()
                sim = F.cosine_similarity(feat.unsqueeze(0), torch.tensor(self.target_feature).unsqueeze(0)).item()

                hsv1 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(img_target, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist1, hist1)
                cv2.normalize(hist2, hist2)
                color_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                reward = (((sim + 1) / 2 - 0.5) * 2.0 + (color_sim - 0.5) * 1.0) * 35

                return reward, (sim + 1) / 2 * 100, color_sim

        return -0.5, 0.0, 0.0

    def close(self):
        pygame.quit()
        cv2.destroyAllWindows()


# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    env = ImageTileEnv("../../data/image.png", "../../data/target.png", n_tiles=3, window_size=800,
                       true_target_tile=(2, 1))
    agent = SARSAAgent(env.n_tiles, env.n_tiles, 6, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, lambda_=0.8)

    num_episodes = 200
    rewards_per_episode = []
    steps_per_episode = []

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"./logs/log_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    log_filename = f"{run_dir}/log.txt"

    with open(log_filename, "w") as log_file:
        log_file.write("=== SARSA(\u03bb) Agent Hyperparameters ===\n")
        log_file.write(f"alpha: {agent.alpha}\n")
        log_file.write(f"gamma: {agent.gamma}\n")
        log_file.write(f"lambda: {agent.lambda_}\n")
        log_file.write(f"epsilon_start: 1.0\n")
        log_file.write(f"epsilon_min: {agent.epsilon_min}\n")
        log_file.write(f"num_episodes: {num_episodes}\n")
        log_file.write("===================================\n\n")
        log_file.write("Episode,Steps,EpisodeReward,CumulativeReward\n")

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        agent.reset_traces()
        total_reward = 0.0
        step_count = 0
        done = False
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, info = env.step(action)
            next_action = agent.select_action(next_obs)
            agent.update(obs, action, reward, next_obs, next_action)
            obs, action = next_obs, next_action
            total_reward += reward
            step_count += 1

            if env.mode == 'move':
                print(f"Episode {episode} | Step {step_count} | Tile {obs.tolist()} | Reward: {reward:.3f} | Cumulative: {total_reward:.3f}")

            env.render()

        agent.dynamic_decay_epsilon(episode)
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)

        cumulative_reward = sum(rewards_per_episode)

        with open(log_filename, "a") as log_file:
            log_file.write(f"{episode},{step_count},{total_reward:.4f},{cumulative_reward:.4f}\n")

        print(f"Episode {episode} finished! Reward: {total_reward:.3f} | Steps: {step_count} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    elapsed = time.time() - start_time

    # === Final Summary ===
    max_reward = np.max(rewards_per_episode)
    min_reward = np.min(rewards_per_episode)
    avg_reward = np.mean(rewards_per_episode)
    avg_last10 = np.mean(rewards_per_episode[-10:])

    # Last 10 episode statistics
    num_last = min(10, len(rewards_per_episode))
    last_10 = rewards_per_episode[-num_last:]
    last_10_steps = steps_per_episode[-num_last:]

    with open(log_filename, "a") as log_file:
        log_file.write("\n=== Final Summary ===\n")
        log_file.write(f"Total time: {elapsed:.2f} sec\n")
        log_file.write(f"Max reward: {max_reward:.2f}\n")
        log_file.write(f"Min reward: {min_reward:.2f}\n")
        log_file.write(f"Avg reward: {avg_reward:.2f}\n")
        log_file.write(f"Avg reward (last {num_last}): {avg_last10:.2f}\n")
        log_file.write(f"Max reward (last {num_last}): {np.max(last_10):.2f}\n")
        log_file.write(f"Min reward (last {num_last}): {np.min(last_10):.2f}\n")
        log_file.write(f"Avg steps (last {num_last}): {np.mean(last_10_steps):.2f}\n")
        log_file.write("========================\n")

    # === Plot 1: Reward per Episode ===
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, marker='o', label='Reward')
    if len(rewards_per_episode) >= 10:
        moving_avg = np.convolve(rewards_per_episode, np.ones(10) / 10, mode='valid')
        plt.plot(range(9, len(rewards_per_episode)), moving_avg, linestyle='--', label='10-Episode Moving Avg')
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    reward_plot_filename = f"{run_dir}/reward_plot.png"
    plt.savefig(reward_plot_filename)
    plt.show()
    plt.close()
    print(f"Reward plot saved to: {reward_plot_filename}")

    # === Plot 2: Steps per Episode ===
    plt.figure(figsize=(12, 6))
    plt.plot(steps_per_episode, marker='o', color='green', label='Steps per Episode')
    if len(steps_per_episode) >= 10:
        moving_avg_steps = np.convolve(steps_per_episode, np.ones(10) / 10, mode='valid')
        plt.plot(range(9, len(steps_per_episode)), moving_avg_steps, linestyle='--', color='darkgreen', label='10-Episode Moving Avg (Steps)')
    plt.title("Steps Taken per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid()
    plt.legend()
    steps_plot_filename = f"{run_dir}/steps_plot.png"
    plt.savefig(steps_plot_filename)
    plt.show()
    plt.close()
    print(f"Steps plot saved to: {steps_plot_filename}")

    # === Plot 3: Last N Rewards ===
    num_last = min(10, len(rewards_per_episode))
    last_10 = rewards_per_episode[-num_last:]
    last_10_indices = list(range(len(rewards_per_episode) - num_last + 1, len(rewards_per_episode) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(last_10_indices, last_10, marker='o', label='Last Rewards')
    y_min = 5 * (np.floor(min(last_10) / 5))
    y_max = 5 * (np.ceil(max(last_10) / 5))
    plt.yticks(np.arange(y_min, y_max + 1, 5))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Last {num_last} Episodes: Reward per Episode")
    plt.grid(True)
    plt.legend()
    last10_plot_filename = f"{run_dir}/last10_reward_plot.png"
    plt.savefig(last10_plot_filename)
    plt.close()
    print(f"Zoomed reward plot saved to: {last10_plot_filename}")

    # === EVALUATION PHASE ===
    eval_episodes = 20
    eval_rewards = []
    eval_steps = []
    eval_successes = 0

    agent.epsilon = 0.0  # disable exploration
    print("\n=== EVALUATION PHASE (no learning, no exploration) ===")

    for ep in range(1, eval_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_obs)  # used for consistency (but not updating)
            obs, action = next_obs, next_action
            total_reward += reward
            step_count += 1

            # Optional: env.render()

        eval_rewards.append(total_reward)
        eval_steps.append(step_count)
        if env.found_target:
            eval_successes += 1

        print(f"[Eval {ep}] Reward: {total_reward:.2f} | Steps: {step_count} | Success: {env.found_target}")

    # === Evaluation Summary ===
    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_steps = np.mean(eval_steps)
    success_rate = eval_successes / eval_episodes * 100

    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {avg_eval_reward:.2f}")
    print(f"Average Steps: {avg_eval_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")

    with open(log_filename, "a") as log_file:
        log_file.write("\n=== Evaluation Summary ===\n")
        log_file.write(f"Evaluation episodes: {eval_episodes}\n")
        log_file.write(f"Average reward: {avg_eval_reward:.2f}\n")
        log_file.write(f"Average steps: {avg_eval_steps:.2f}\n")
        log_file.write(f"Success rate: {success_rate:.1f}%\n")
        log_file.write("========================\n")

    # === DETERMINISTIC EVALUATION PHASE ===
    print("\n=== DETERMINISTIC EVALUATION PHASE ===")
    det_env = ImageTileEnv("../../data/image.png", "../../data/target.png", n_tiles=3, window_size=800,
                           true_target_tile=(2, 1), fixed_start=True)

    eval_rewards_det = []
    eval_steps_det = []
    eval_successes_det = 0

    agent.epsilon = 0.0  # no exploration

    for ep in range(1, eval_episodes + 1):
        obs = det_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, _ = det_env.step(action)
            next_action = agent.select_action(next_obs)
            obs, action = next_obs, next_action
            total_reward += reward
            step_count += 1

            # Optional: det_env.render()

        eval_rewards_det.append(total_reward)
        eval_steps_det.append(step_count)
        if det_env.found_target:
            eval_successes_det += 1

        print(f"[Eval-Det {ep}] Reward: {total_reward:.2f} | Steps: {step_count} | Success: {det_env.found_target}")

    # === Deterministic Evaluation Summary ===
    avg_reward_det = np.mean(eval_rewards_det)
    avg_steps_det = np.mean(eval_steps_det)
    success_rate_det = eval_successes_det / eval_episodes * 100

    print("\n=== Deterministic Evaluation Results ===")
    print(f"Average Reward: {avg_reward_det:.2f}")
    print(f"Average Steps: {avg_steps_det:.2f}")
    print(f"Success Rate: {success_rate_det:.1f}%")

    with open(log_filename, "a") as log_file:
        log_file.write("\n=== Deterministic Evaluation Summary ===\n")
        log_file.write(f"Evaluation episodes: {eval_episodes}\n")
        log_file.write(f"Average reward: {avg_reward_det:.2f}\n")
        log_file.write(f"Average steps: {avg_steps_det:.2f}\n")
        log_file.write(f"Success rate: {success_rate_det:.1f}%\n")
        log_file.write("========================\n")

    input("Press Enter to exit and close plots...")
