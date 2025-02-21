# game.py

import pygame
import random
from config import *
from utils import get_state, calculate_reward


class SpaceShooter:
    def __init__(self, training_mode=True):
        """
        Initialize the game
        Args:
            training_mode: Whether in training mode
        """
        # Initialize game window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Learning Space Shooter")
        self.clock = pygame.time.Clock()
        self.training_mode = training_mode
        self.render_interval = 10 if training_mode else 1
        self.frame_skip = 4 if training_mode else 1
        
        # Game state
        self.screen = pygame.display.get_surface()
        self.frame_count = 0
        self.last_shot_frame = 0
        self.last_spawn_frame = 0
        self.last_nearest_distance = float('inf')
        self.deaths = 0  # Death counter
        
        # Game data
        self.level = 1
        self.score = 0
        self.health = 2
        self.enemies_in_level = 0
        self.enemies_spawned_this_wave = 0
        self.max_enemies_per_level = 8
        self.max_level = 10  # Maximum level
        self.wave_spawn_delay = 120
        self.spawn_interval = 120  # Initial spawn interval (2 seconds)
        
        # Game objects
        self.bullets = []
        self.enemies = []
        self.player_rect = self._create_player()
        self.game_won = False  # Victory flag

    def _create_player(self):
        return pygame.Rect(
            WINDOW_WIDTH // 2 - PLAYER_SIZE[0] // 2,
            WINDOW_HEIGHT - PLAYER_SIZE[1] - 10,
            PLAYER_SIZE[0],
            PLAYER_SIZE[1]
        )

    def reset_game(self):
        self.frame_count = self.last_shot_frame = self.last_spawn_frame = 0
        self.last_nearest_distance = float('inf')
        self.level = 1
        self.score = 0
        self.health = 2
        self.enemies_in_level = self.enemies_spawned_this_wave = 0
        self.bullets.clear()
        self.enemies.clear()
        self.player_rect = self._create_player()
        return get_state(self.player_rect, self.enemies, self.bullets)

    def spawn_enemy(self):
        if self.game_won:
            return

        # Adjust spawn interval based on level (from 2s to 0.5s)
        self.spawn_interval = max(30, 120 - (self.level - 1) * 10)
        
        if self.frame_count - self.last_spawn_frame < self.spawn_interval:
            return

        max_enemies = self.max_enemies_per_level * self.level
        if self.enemies_in_level >= max_enemies:
            if len(self.enemies) == 0:  # All enemies in current level eliminated
                if self.level >= self.max_level:
                    self.game_won = True
                    return
                self.level += 1
                self.enemies_in_level = self.enemies_spawned_this_wave = 0
            return

        enemy = pygame.Rect(
            random.randint(0, WINDOW_WIDTH - ENEMY_SIZE[0]),
            0,
            *ENEMY_SIZE
        )
        self.enemies.append(enemy)
        self.enemies_in_level += 1
        self.enemies_spawned_this_wave += 1
        self.last_spawn_frame = self.frame_count

    def shoot(self):
        if self.frame_count - self.last_shot_frame > 60:
            self.bullets.append(pygame.Rect(
                self.player_rect.centerx - BULLET_SIZE[0] // 2,
                self.player_rect.top,
                *BULLET_SIZE
            ))
            self.last_shot_frame = self.frame_count

    def step(self, action):
        reward = killed_enemies = missed_shots = collisions = 0
        self.last_action = action  # Record current action

        for _ in range(self.frame_skip):
            self.frame_count += 1
            self._handle_action(action)
            self._update_bullets(missed_shots)
            self.spawn_enemy()
            game_over, reward_delta = self._handle_collisions()
            reward += reward_delta
            if game_over:
                break

        reward += calculate_reward(
            self.player_rect, self.enemies, self.bullets,
            killed_enemies, missed_shots, self.last_nearest_distance
        )
        self.last_reward = reward  # Record current reward

        return get_state(self.player_rect, self.enemies, self.bullets), reward, game_over, {
            'score': self.score,
            'killed_enemies': killed_enemies,
            'missed_shots': missed_shots,
            'collisions': collisions
        }

    def _handle_action(self, action):
        if action in [0, 3]:
            self.player_rect.x -= PLAYER_SPEED
        elif action in [1, 4]:
            self.player_rect.x += PLAYER_SPEED
        if action in [2, 3, 4]:
            self.shoot()
        self.player_rect.clamp_ip(pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))

    def _update_bullets(self, missed_shots):
        for bullet in self.bullets[:]:
            bullet.y -= BULLET_SPEED
            if bullet.bottom < 0:
                self.bullets.remove(bullet)
                missed_shots += 1

    def _handle_collisions(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy.y += ENEMY_SPEED
            
            if enemy.colliderect(self.player_rect):
                if not self.training_mode:
                    return True, 0
                reward -= 5.0
                continue

            for bullet in self.bullets[:]:
                if enemy.colliderect(bullet):
                    self.score += 1
                    self.enemies.remove(enemy)
                    self.bullets.remove(bullet)
                    break

            if enemy.top > WINDOW_HEIGHT:
                self.enemies.remove(enemy)
                self.health -= 1
                if self.health <= 0:
                    self.deaths += 1  # Increment death counter
                    return True, 0
                if self.training_mode:
                    reward -= 3.0

        return False, reward

    def render(self):
        if self.training_mode and self.frame_count % self.render_interval != 0:
            return

        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.player_rect)
        
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, WHITE, bullet)
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, RED, enemy)

        self._render_ui()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _render_ui(self):
        self._render_game_stats()
        self._render_progress()
        self._render_agent_info()
        if self.game_won:
            self._render_victory_message()

    def _render_game_stats(self):
        font = pygame.font.Font(None, 36)
        texts = [
            (f'Score: {self.score}', (10, 10)),
            (f'Level: {self.level}/{self.max_level}', (10, 50)),
            (f'Health: {self.health}', (10, 90))
        ]
        if self.training_mode:
            texts.append((f'Deaths: {self.deaths}', (10, 130)))

        for text, pos in texts:
            self.screen.blit(font.render(text, True, WHITE), pos)

    def _render_progress(self):
        font = pygame.font.Font(None, 36)
        max_enemies = self.max_enemies_per_level * self.level
        progress = f'Progress: {self.enemies_in_level - len(self.enemies)}/{max_enemies}'
        progress_percent = f'({(self.enemies_in_level - len(self.enemies)) / max_enemies * 100:.1f}%)'
        text = progress + ' ' + progress_percent
        self.screen.blit(font.render(text, True, WHITE), (10, 170))

    def _render_agent_info(self):
        if not self.training_mode:
            return

        font = pygame.font.Font(None, 28)
        info_x = WINDOW_WIDTH - 200
        
        # Display current action
        action_names = ['Move Left', 'Move Right', 'Shoot', 'Move Left + Shoot', 'Move Right + Shoot']
        last_action = getattr(self, 'last_action', 0)
        action_text = f'Action: {action_names[last_action]}'
        self.screen.blit(font.render(action_text, True, WHITE), (info_x, 10))
        
        # Display recent reward
        last_reward = getattr(self, 'last_reward', 0.0)
        reward_color = GREEN if last_reward > 0 else RED if last_reward < 0 else WHITE
        reward_text = f'Reward: {last_reward:.2f}'
        self.screen.blit(font.render(reward_text, True, reward_color), (info_x, 40))

    def _render_victory_message(self):
        victory_font = pygame.font.Font(None, 72)
        victory_text = victory_font.render('Victory!', True, WHITE)
        text_rect = victory_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(victory_text, text_rect)

    def close(self):
        pygame.quit()