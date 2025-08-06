import os
import pygame
import numpy as np
import sys
import gym
from gym import spaces
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

plt.ion()
import random
import time
from datetime import datetime

# === SARSA(λ) Agent ===
class SARSAAgent:
    def __init__(self, n_rows, n_cols, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, lambda_=0.8):
        self.q_table = np.random.uniform(0, 0.01,
                                         size=(n_rows, n_cols, n_actions))
        self.e_trace = np.zeros((n_rows, n_cols, n_actions))
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
        nr, nc = next_state
        delta = (reward
                 + self.gamma * self.q_table[nr, nc, next_action]
                 - self.q_table[row, col, action])
        self.e_trace[row, col, action] += 1
        self.q_table += self.alpha * delta * self.e_trace
        self.e_trace *= self.gamma * self.lambda_

    def reset_traces(self):
        self.e_trace.fill(0)

    def dynamic_decay_epsilon(self, episode):
        decay_rate = 0.995
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * decay_rate)


# === Environment ===
class ImageTileEnv(gym.Env):
    def __init__(self, image_path, target_path,
                 n_tiles=5, window_size=800,
                 manual_mode=False, true_target_tile=(2, 1), fixed_start = False):
        super().__init__()
        self.image_path = image_path
        self.target_path = target_path
        self.n_tiles = n_tiles
        self.window_size = window_size
        self.manual_mode = manual_mode
        self.true_target_tile = true_target_tile
        self.fixed_start = fixed_start

        pygame.init()
        self.clock = pygame.time.Clock()

        raw_image = pygame.image.load(self.image_path)
        scaled_size = window_size - (window_size % n_tiles)
        self.image = pygame.transform.scale(raw_image,
                                            (scaled_size, scaled_size))
        self.img_width, self.img_height = self.image.get_size()
        self.tile_width = self.img_width // n_tiles
        self.tile_height = self.img_height // n_tiles

        self.screen = pygame.display.set_mode(
            (self.img_width, self.img_height))
        pygame.display.set_caption("Sliding Window Agent")

        self.tiles = []
        for r in range(n_tiles):
            for c in range(n_tiles):
                rect = pygame.Rect(c*self.tile_width,
                                   r*self.tile_height,
                                   self.tile_width,
                                   self.tile_height)
                self.tiles.append(self.image.subsurface(rect).copy())

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0,
                                            high=n_tiles-1,
                                            shape=(2,),
                                            dtype=np.int32)
        self.reset_vars()

        # ResNet for deep features
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

        # Precompute target feature
        self.target_feature = self.load_target_feature(self.target_path)

    def reset_vars(self):
        if self.fixed_start:
            self.window_pos = [1, 1]
        else:
            self.window_pos = [np.random.randint(0, self.n_tiles), np.random.randint(0, self.n_tiles)]

        self.last_tile = tuple(self.window_pos)
        self.stay_count = 0
        self.mode = 'move'
        self.found_target = False
        self.truly_done = False
        self.stay_in_target_count = 0
        self.last_target_tile = None

    def reset(self):
        self.reset_vars()
        return np.array(self.window_pos, dtype=np.int32)

    def step(self, action):
        reward = 0.0
        done = False
        r, c = self.window_pos

        if self.mode == 'move':
            reward -= 1
            if action == 0 and c < self.n_tiles-1: c += 1
            elif action == 1 and r < self.n_tiles-1: r += 1
            elif action == 2 and c > 0: c -= 1
            elif action == 3 and r > 0: r -= 1

            cur = (r, c)
            if cur == self.last_tile:
                self.stay_count += 1
                reward -= 1 * self.stay_count
            else:
                self.stay_count = 0
            self.last_tile = cur
            self.window_pos = [r, c]
            self.mode = 'decision'

        elif self.mode == 'decision':
            if action == 4:
                reward -= 1
                loc_r, match_pct, color_sim = self.perform_localization()
                reward += loc_r

                # Debug prints
                print("match percentage is: ", match_pct)
                print("color_sim is: ", color_sim)

                if self.manual_mode:
                    print("\n--- MANUAL LOCALIZATION CHECK ---")
                    print(f"Match %:        {match_pct:.1f}")
                    print(f"Color sim:      {color_sim:.4f}")
                    rect = pygame.Rect(
                        c*self.tile_width, r*self.tile_height,
                        self.tile_width, self.tile_height)
                    pygame.draw.rect(self.screen, (255,0,0), rect, 4)
                    pygame.display.flip()
                    print("Press any key to continue…")
                    while True:
                        evt = pygame.event.wait()
                        if evt.type in (pygame.KEYDOWN,
                                        pygame.MOUSEBUTTONDOWN):
                            break

                # check if agent lands on correct tile
                is_correct_tile = (tuple(self.window_pos) == self.true_target_tile)
                if is_correct_tile:
                    if not self.found_target:
                        self.found_target = True
                        self.last_target_tile = tuple(self.window_pos)
                        reward += max(54, 62- step_count ** 1.5) #45 #10
                    if tuple(self.window_pos) == self.last_target_tile:
                        self.stay_in_target_count += 1
                    else:
                        self.stay_in_target_count = 0
                else:
                    self.stay_in_target_count = 0
            else:
                self.stay_in_target_count = 0
            self.mode = 'move'

        if (self.found_target
            and self.stay_in_target_count >= 2
            and not self.truly_done):
            reward += 65
            done = True
            self.truly_done = True

        return np.array(self.window_pos, dtype=np.int32), reward, done, {}

    def render(self, mode='human'):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.close()

        for idx, tile in enumerate(self.tiles):
            rr, cc = divmod(idx, self.n_tiles)
            self.screen.blit(tile,
                             (cc*self.tile_width,
                              rr*self.tile_height))

        # draw grid
        for i in range(1, self.n_tiles):
            pygame.draw.line(self.screen, (255,255,255),
                             (i*self.tile_width, 0),
                             (i*self.tile_width, self.img_height), 2)
            pygame.draw.line(self.screen, (255,255,255),
                             (0, i*self.tile_height),
                             (self.img_width, i*self.tile_height), 2)

        # highlight current
        rr, cc = self.window_pos
        rect = pygame.Rect(cc*self.tile_width,
                           rr*self.tile_height,
                           self.tile_width,
                           self.tile_height)
        pygame.draw.rect(self.screen, (0,255,0), rect, 4)
        pygame.display.flip()
        self.clock.tick(10)

    def extract_features(self, surf):
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1,0,2))
        img = self.transform(arr).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(img).squeeze().cpu().numpy()

    def load_target_feature(self, path):
        raw = pygame.image.load(path)
        raw = pygame.transform.scale(raw, (224,224))
        arr = pygame.surfarray.array3d(raw)
        arr = np.transpose(arr, (1,0,2))
        img = self.transform(arr).unsqueeze(0)
        with torch.no_grad():
            return self.resnet(img).squeeze().cpu().numpy()

    def perform_localization(self):
        r, c = self.window_pos
        idx = r*self.n_tiles + c
        surf = self.tiles[idx]
        feat = self.extract_features(surf)
        sim = F.cosine_similarity(
            torch.tensor(feat).unsqueeze(0),
            torch.tensor(self.target_feature).unsqueeze(0)
        ).item()
        match_pct = (sim + 1)/2 * 100

        # color hist
        tile_arr = pygame.surfarray.array3d(surf)
        tile_arr = np.transpose(tile_arr,(1,0,2))
        img_tile = cv2.cvtColor(tile_arr, cv2.COLOR_RGB2BGR)
        raw_t = pygame.image.load(self.target_path)
        raw_t = pygame.transform.scale(
            raw_t, (self.tile_width, self.tile_height))
        t_arr = pygame.surfarray.array3d(raw_t)
        t_arr = np.transpose(t_arr,(1,0,2))
        img_t = cv2.cvtColor(t_arr, cv2.COLOR_BGR2RGB)
        hsv1 = cv2.cvtColor(img_tile, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img_t, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1],[0,1],None,[50,60],[0,180,0,256])
        hist2 = cv2.calcHist([hsv2],[0,1],None,[50,60],[0,180,0,256])
        cv2.normalize(hist1,hist1)
        cv2.normalize(hist2,hist2)
        color_sim = cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

        reward = (((sim + 1) / 2 - 0.5) * 2.0 + (color_sim - 0.5) * 1.0) * 35
        return reward, match_pct, color_sim

    def close(self):
        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # === User parameters ===
    true_target_tile = (2, 1)

    env = ImageTileEnv(
        "../../data/image.png",
        "../../data/target.png",
        n_tiles=3,
        window_size=800,
        manual_mode=False,
        true_target_tile=true_target_tile
    )
    agent = SARSAAgent(
        env.n_tiles, env.n_tiles, env.action_space.n,
        alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, lambda_=0.8
    )

    num_episodes = 100
    rewards_per_episode = []
    steps_per_episode = []

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"./logs/log_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    log_filename = os.path.join(run_dir, "log.txt")

    with open(log_filename, "w") as f:
        f.write("=== SARSA(λ) Agent Hyperparameters ===\n")
        f.write(f"alpha: {agent.alpha}\n")
        f.write(f"gamma: {agent.gamma}\n")
        f.write(f"lambda: {agent.lambda_}\n")
        f.write(f"epsilon_start: {agent.epsilon}\n")
        f.write(f"epsilon_min: {agent.epsilon_min}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write("===================================\n\n")
        f.write("Episode,Steps,TotalReward\n")

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        agent.reset_traces()
        total_reward = 0.0
        step_count = 0
        done = False
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_obs)
            agent.update(obs, action, reward, next_obs, next_action)
            obs, action = next_obs, next_action

            total_reward += reward
            step_count += 1

            if env.mode == 'move':
                print(f"Episode {episode} | Step {step_count} | Tile {obs.tolist()} — "
                      f"Reward: {reward:.3f} | Cumulative: {total_reward:.3f}")

            env.render()

        agent.dynamic_decay_epsilon(episode)
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)

        with open(log_filename, "a") as f:
            f.write(f"{episode},{step_count},{total_reward:.4f}\n")

        print(f"Episode {episode} finished! Reward: {total_reward:.3f} | "
              f"Steps: {step_count} | Epsilon: {agent.epsilon:.3f}")

    elapsed = time.time() - start_time
    with open(log_filename, "a") as f:
        f.write("\n=== Final Summary ===\n")
        f.write(f"Total time: {elapsed:.2f} sec\n")
        f.write(f"Max reward: {np.max(rewards_per_episode):.2f}\n")
        f.write(f"Min reward: {np.min(rewards_per_episode):.2f}\n")
        f.write(f"Avg reward: {np.mean(rewards_per_episode):.2f}\n")
        f.write(f"Avg reward (last 10): {np.mean(rewards_per_episode[-10:]):.2f}\n")
        f.write("========================\n")

    # === Plot 1: Reward per Episode ===
    fig = plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, marker='o', label='Reward')
    if len(rewards_per_episode) >= 10:
        ma = np.convolve(rewards_per_episode, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(rewards_per_episode)), ma, linestyle='--', label='10-episode MA')
    plt.title("Cumulative Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    reward_plot_path = os.path.join(run_dir, "reward_plot.png")
    fig.savefig(reward_plot_path)
    print(f"Saved reward plot → {reward_plot_path}")
    plt.close(fig)

    # === Plot 2: Steps per Episode ===
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(steps_per_episode, marker='o', label='Steps per Episode')
    if len(steps_per_episode) >= 10:
        ma2 = np.convolve(steps_per_episode, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(steps_per_episode)), ma2, linestyle='--', label='10-episode MA')
    plt.title("Steps Taken per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.legend()
    steps_plot_path = os.path.join(run_dir, "steps_plot.png")
    fig2.savefig(steps_plot_path)
    print(f"Saved steps plot → {steps_plot_path}")
    plt.close(fig2)

    # === EVALUATION PHASE ===
    eval_episodes = 20
    eval_rewards = []
    eval_steps = []
    eval_successes = 0

    agent.epsilon = 0.0
    print("\n=== EVALUATION PHASE (no learning, no exploration) ===")

    for ep in range(1, eval_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_obs)
            obs, action = next_obs, next_action
            total_reward += reward
            step_count += 1

        eval_rewards.append(total_reward)
        eval_steps.append(step_count)
        if env.found_target:
            eval_successes += 1

        print(f"[Eval {ep}] Reward: {total_reward:.2f} | Steps: {step_count} | Success: {env.found_target}")

    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_steps = np.mean(eval_steps)
    success_rate = eval_successes / eval_episodes * 100

    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {avg_eval_reward:.2f}")
    print(f"Average Steps: {avg_eval_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")

    with open(log_filename, "a") as f:
        f.write("\n=== Evaluation Summary ===\n")
        f.write(f"Evaluation episodes: {eval_episodes}\n")
        f.write(f"Average reward: {avg_eval_reward:.2f}\n")
        f.write(f"Average steps: {avg_eval_steps:.2f}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write("========================\n")

    # === DETERMINISTIC EVALUATION PHASE ===
    print("\n=== DETERMINISTIC EVALUATION PHASE ===")
    det_env = ImageTileEnv("../../data/image.png", "../../data/car.png", n_tiles=3, window_size=800,
                           true_target_tile=true_target_tile, fixed_start=True)

    eval_rewards_det = []
    eval_steps_det = []
    eval_successes_det = 0

    agent.epsilon = 0.0

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

        eval_rewards_det.append(total_reward)
        eval_steps_det.append(step_count)
        if det_env.found_target:
            eval_successes_det += 1

        print(f"[Eval-Det {ep}] Reward: {total_reward:.2f} | Steps: {step_count} | Success: {det_env.found_target}")

    avg_reward_det = np.mean(eval_rewards_det)
    avg_steps_det = np.mean(eval_steps_det)
    success_rate_det = eval_successes_det / eval_episodes * 100

    print("\n=== Deterministic Evaluation Results ===")
    print(f"Average Reward: {avg_reward_det:.2f}")
    print(f"Average Steps: {avg_steps_det:.2f}")
    print(f"Success Rate: {success_rate_det:.1f}%")

    with open(log_filename, "a") as f:
        f.write("\n=== Deterministic Evaluation Summary ===\n")
        f.write(f"Evaluation episodes: {eval_episodes}\n")
        f.write(f"Average reward: {avg_reward_det:.2f}\n")
        f.write(f"Average steps: {avg_steps_det:.2f}\n")
        f.write(f"Success rate: {success_rate_det:.1f}%\n")
        f.write("========================\n")

    env.close()
    print(f"Training completed in {elapsed:.2f} seconds")
