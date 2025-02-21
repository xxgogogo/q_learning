# evaluate.py

import os
import json
import torch
import numpy as np
from config import PTH_DIR
from game import SpaceShooter
from train import DQNNetwork  # 导入DQN网络模型类

class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.metrics_file = os.path.join(PTH_DIR, 'model_metrics.json')
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """加载评估指标"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_metrics(self):
        """保存评估指标"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)

    def add_game_result(self, model_name, score, survival_time, hit_rate):
        """添加游戏结果
        Args:
            model_name: 模型名称
            score: 得分
            survival_time: 存活时间（秒）
            hit_rate: 命中率
        """
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                'games_played': 0,
                'total_score': 0,
                'max_score': 0,
                'min_score': float('inf'),
                'total_survival_time': 0,
                'max_survival_time': 0,
                'total_hit_rate': 0,
                'max_hit_rate': 0,
                'levels_reached': []
            }

        metrics = self.metrics[model_name]
        metrics['games_played'] += 1
        metrics['total_score'] += score
        metrics['max_score'] = max(metrics['max_score'], score)
        metrics['min_score'] = min(metrics['min_score'], score)
        metrics['total_survival_time'] += survival_time
        metrics['max_survival_time'] = max(metrics['max_survival_time'], survival_time)
        metrics['total_hit_rate'] += hit_rate
        metrics['max_hit_rate'] = max(metrics['max_hit_rate'], hit_rate)

        # 更新平均值和标准差
        metrics['avg_score'] = metrics['total_score'] / metrics['games_played']
        metrics['avg_survival_time'] = metrics['total_survival_time'] / metrics['games_played']
        metrics['avg_hit_rate'] = metrics['total_hit_rate'] / metrics['games_played']

        self.save_metrics()

    def get_model_metrics(self, model_name):
        """获取模型评估指标
        Args:
            model_name: 模型名称
        Returns:
            dict: 模型评估指标
        """
        metrics = self.metrics.get(model_name, {})
        if metrics:
            return {
                '游戏场数': metrics['games_played'],
                '平均得分': f"{metrics['avg_score']:.2f}",
                '最高得分': metrics['max_score'],
                '最低得分': metrics['min_score'],
                '平均存活时间': f"{metrics['avg_survival_time']:.2f}秒",
                '最长存活时间': f"{metrics['max_survival_time']:.2f}秒",
                '平均命中率': f"{metrics['avg_hit_rate']:.2%}",
                '最高命中率': f"{metrics['max_hit_rate']:.2%}"
            }
        return {}

    def get_all_models_metrics(self):
        """获取所有模型的评估指标
        Returns:
            dict: 所有模型的评估指标
        """
        return {name: self.get_model_metrics(name) for name in self.metrics}

    def get_best_model(self, metric='avg_score'):
        """获取最佳模型
        Args:
            metric: 评估指标（'avg_score', 'avg_survival_time', 'avg_hit_rate'）
        Returns:
            tuple: (最佳模型名称, 最佳指标值)
        """
        if not self.metrics:
            return None, 0

        best_model = max(self.metrics.items(),
                        key=lambda x: x[1].get(metric, 0))
        return best_model[0], best_model[1].get(metric, 0)

    def get_performance_summary(self, model_name):
        """获取模型性能总结
        Args:
            model_name: 模型名称
        Returns:
            str: 性能总结文本
        """
        metrics = self.metrics.get(model_name)
        if not metrics:
            return "未找到该模型的评估数据"

        return f"模型性能总结:\n" \
               f"- 总场数: {metrics['games_played']}\n" \
               f"- 得分: 平均 {metrics['avg_score']:.2f} (最高 {metrics['max_score']}, 最低 {metrics['min_score']})\n" \
               f"- 存活时间: 平均 {metrics['avg_survival_time']:.2f}秒 (最长 {metrics['max_survival_time']:.2f}秒)\n" \
               f"- 命中率: 平均 {metrics['avg_hit_rate']:.2%} (最高 {metrics['max_hit_rate']:.2%})"

def evaluate_model(model_path):
    """评估模型性能
    Args:
        model_path: 模型文件路径
    Returns:
        tuple: (得分, 存活时间, 命中数, 未命中数)
    """
    # 初始化游戏环境
    env = SpaceShooter(training_mode=True)
    
    # 加载模型
    state_size = 4  # 状态空间大小
    action_size = 5  # 动作空间大小
    model = DQNNetwork(state_size, action_size)
    checkpoint = torch.load(model_path, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 重置环境
    state = env.reset_game()
    done = False
    score = 0
    frame_count = 0
    hits = misses = 0
    
    # 运行一个完整的游戏回合
    while not done:
        frame_count += 1
        # 选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = model(state_tensor).argmax().item()
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 更新统计信息
        score = info['score']
        hits += info['killed_enemies']
        misses += info['missed_shots']
        
        state = next_state
    
    env.close()
    return score, frame_count, hits, misses