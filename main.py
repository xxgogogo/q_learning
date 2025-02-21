# main.py

import pygame
import os
import time
from game import SpaceShooter
from train import create_trainer
from config import *
from utils import get_state

def train(target_deaths):
    print("\nStarting training mode...")
    print(f"Training target: {target_deaths} deaths")
    print("Press 'Q' to quit training")

    game = SpaceShooter(training_mode=True)
    trainer = create_trainer()
    episode = total_deaths = 0

    while total_deaths < target_deaths:
        episode += 1
        state = game.reset_game()
        total_reward = 0

        while True:
            action = trainer.get_action(state)
            next_state, reward, done, info = game.step(action)
            trainer.memory.push(state, action, reward, next_state, done)
            trainer.train(trainer.batch_size)

            state = next_state
            total_reward += reward
            game.render()

            if pygame.event.get(pygame.QUIT) or pygame.key.get_pressed()[pygame.K_q]:
                game.close()
                return

            if done:
                total_deaths += 1
                break

            if total_deaths >= target_deaths:
                trainer.save_model(episode)
                print(f"\nTarget deaths reached, training complete! (Total deaths: {total_deaths})")
                game.close()
                return

        print(f"Episode: {episode}, Score: {info['score']}, Deaths: {total_deaths}/{target_deaths}, Total reward: {total_reward:.2f}")

        if episode % 50 == 0 or total_deaths in [target_deaths//4, target_deaths//2, target_deaths*3//4]:
            trainer.save_model(episode)
            print(f"Model saved: Episode {episode} (Total deaths: {total_deaths})")

    game.close()

def play(use_ai=False):
    game = SpaceShooter(training_mode=False)
    trainer = None

    if use_ai:
        print("\nStarting AI mode...")
        trainer = create_trainer()
        models = [f for f in os.listdir(PTH_DIR) if f.endswith('.pth')]
        if not models:
            print("No available models found!")
            return

        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input("Select model number: ").strip())
                if 1 <= choice <= len(models):
                    selected_model = models[choice-1]
                    break
                print("Invalid selection, please try again")
            except ValueError:
                print("Please enter a valid number")

        trainer.load_model(os.path.join(PTH_DIR, selected_model))
        print(f"Model loaded: {selected_model}")

        start_time = time.time()
        shots_fired = hits = 0
    else:
        print("\nStarting game mode...\nControls:\n← →: Move\nSpace: Shoot\nESC: Quit")

    running = True
    while running:
        action = None
        if trainer:
            state = get_state(game.player_rect, game.enemies, game.bullets)
            action = trainer.get_action(state)
            if action in [2, 3, 4]:
                shots_fired += 1
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
                action = 3
            elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
                action = 4
            elif keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            elif keys[pygame.K_SPACE]:
                action = 2

        if pygame.event.get(pygame.QUIT) or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False
            break

        _, _, done, info = game.step(action if action is not None else -1)
        game.render()

        if trainer and 'killed_enemies' in info:
            hits += info['killed_enemies']

        if done:
            if trainer:
                survival_time = time.time() - start_time
                hit_rate = hits / max(1, shots_fired)
                print(f"Game Over! Score: {info['score']}, Survival time: {survival_time:.2f}s, Hit rate: {hit_rate:.2%}")
            else:
                print(f"Game Over! Score: {info['score']}")
            running = False

    game.close()

def main():
    while True:
        print("\n=== Space Shooter Game ===")
        print("1. Start Game (Player Mode)")
        print("2. Start Game (AI Mode)")
        print("3. Train Model")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == '1':
            play(use_ai=False)
        elif choice == '2':
            play(use_ai=True)
        elif choice == '3':
            episodes = int(input("Enter number of training episodes: ").strip())
            train(episodes)
        elif choice == '0':
            print("Game exited")
            break
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()