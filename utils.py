# utils.py

import numpy as np
from config import *


def get_state(player_rect, enemies, bullets):
    """
    Get current game state
    Args:
        player_rect: Player rectangle object
        enemies: List of enemies
        bullets: List of bullets
    Returns:
        tuple: Contains discretized game state information
    """
    # Discretize player position
    player_x = player_rect.centerx // (WINDOW_WIDTH // STATE_SIZE)

    # Get nearest enemy information
    nearest_enemy_x = -1
    nearest_enemy_y = -1
    min_distance = float('inf')

    for enemy in enemies:
        # Calculate the Euclidean distance
        dist = ((enemy.centerx - player_rect.centerx) ** 2 +
                (enemy.centery - player_rect.centery) ** 2) ** 0.5
        if dist < min_distance:
            min_distance = dist
            nearest_enemy_x = enemy.centerx // (WINDOW_WIDTH // STATE_SIZE)
            nearest_enemy_y = enemy.centery // (WINDOW_HEIGHT // STATE_SIZE)
    # Discretize enemy position
    if nearest_enemy_x == -1:
        nearest_enemy_x = 0
        nearest_enemy_y = 0

    # Calculate bullet density
    bullet_density = len(bullets) / (WINDOW_WIDTH * WINDOW_HEIGHT)
    bullet_density = int(bullet_density * STATE_SIZE)

    return player_x, nearest_enemy_x, nearest_enemy_y, bullet_density


def calculate_reward(player_rect, enemies, bullets, killed_enemies, missed_shots, last_nearest_distance):
    """
    Calculate reward value, focusing on x-axis alignment and distance changes
    Args:
        player_rect: Player position rectangle
        enemies: List of enemies
        bullets: List of bullets
        killed_enemies: Number of enemies killed in this step
        missed_shots: Number of missed shots in this step
        last_nearest_distance: Distance to nearest enemy in previous step
    Returns:
        float: Calculated reward value
    """
    reward = killed_enemies * 10.0 - missed_shots * 0.5 + 0.01

    if enemies:
        current_min_distance = min(
            ((enemy.centerx - player_rect.centerx) ** 2 + 
             (enemy.centery - player_rect.centery) ** 2) ** 0.5
            for enemy in enemies
        )
        
        # Reward based on distance change
        reward += 3.0 if current_min_distance < last_nearest_distance else -1.0
        
        # X-axis alignment reward (small weight)
        for enemy in enemies:
            x_diff = abs(player_rect.centerx - enemy.centerx)
            if x_diff < 10:
                reward += 2.0
            elif x_diff < 30:
                reward += 1.0

    # Prevent sticking to boundaries
    if player_rect.left < 10 or player_rect.right > WINDOW_WIDTH - 10:
        reward -= 2.0

    return reward
