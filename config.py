# config.py

import pygame
import os

# Initialize Pygame
pygame.init()

# Game window parameters
WINDOW_WIDTH = 500        # Game window width
WINDOW_HEIGHT = 1000      # Game window height
FPS = 60                  # Game frame rate

# Color definitions
WHITE = (255, 255, 255)   # White
BLACK = (0, 0, 0)         # Black
RED = (255, 0, 0)         # Red
GREEN = (0, 255, 0)       # Green (for positive rewards)
BLUE = (0, 0, 255)        # Blue
YELLOW = (255, 255, 0)    # Yellow

# Game object parameters
PLAYER_SPEED = 15          # Player movement speed
BULLET_SPEED = 45          # Bullet speed
ENEMY_SPEED = 2           # Enemy movement speed
PLAYER_SIZE = (30, 30)    # Player size
ENEMY_SIZE = (40, 40)    # Enemy size
BULLET_SIZE = (4, 10)     # Bullet size

# Q-learning parameters
LEARNING_RATE = 0.1       # Learning rate
DISCOUNT_FACTOR = 0.95    # Discount factor
EPSILON = 0.1             # Exploration rate
STATE_SIZE = 8            # Discrete state space size

# Reward parameters
REWARD_KILL = 50          # Reward for killing enemy
REWARD_MISS = -1          # Penalty for missing shot
REWARD_HIT = -10          # Penalty for being hit by enemy
REWARD_ESCAPE = -5        # Penalty for enemy escape
COLLISION_PENALTY = -10.0  # Collision penalty in training mode

# Path configuration
PTH_DIR = os.path.join(os.path.dirname(__file__), 'pth')  # Model save path
if not os.path.exists(PTH_DIR):
    os.makedirs(PTH_DIR)