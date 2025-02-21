import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from evaluate import evaluate_model
from game import SpaceShooter

def generate_model_report(model_number):
    """Generate evaluation report for the specified model number"""
    # Build model file path
    model_path = f'pth/model_episode_{model_number}.pth'
    if not os.path.exists(model_path):
        return f"Error: Model file not found {model_path}"

    # Execute model evaluation
    episodes = 100  # Number of evaluation episodes
    scores, survival_times, hit_rates = [], [], []
    
    for _ in range(episodes):
        score, survival_time, hits, misses = evaluate_model(model_path)
        scores.append(score)
        survival_times.append(survival_time)
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        hit_rates.append(hit_rate)

    # Generate report content
    report = []
    report.append(f"Q-Learning Model Evaluation Report")
    report.append(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model File: {model_path}\n")

    # Performance statistics
    report.append("Performance Statistics:")
    report.append(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    report.append(f"Average Survival Time: {np.mean(survival_times):.2f} ± {np.std(survival_times):.2f} frames")
    report.append(f"Average Hit Rate: {np.mean(hit_rates)*100:.2f}% ± {np.std(hit_rates)*100:.2f}%\n")

    # Return report text
    return '\n'.join(report)

def main():
    """Main function, handle user input and generate report"""
    try:
        model_number = int(input("Enter the model number to evaluate (e.g., 25): "))
        report = generate_model_report(model_number)
        print(report)
    except ValueError:
        print("Error: Please enter a valid model number (integer)")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()