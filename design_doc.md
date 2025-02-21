# 太空射击游戏设计文档

## 1. 系统架构图

```mermaid
graph TB
    subgraph 游戏核心系统
        Game[游戏引擎 SpaceShooter] --> Physics[物理系统]
        Game --> Renderer[渲染系统]
        Game --> Input[输入处理]
        Physics --> CollisionDetection[碰撞检测]
        Renderer --> UI[用户界面]
    end

    subgraph AI系统
        DQN[DQN网络] --> ActionSelection[动作选择]
        DQN --> StateProcessing[状态处理]
        DQN --> RewardCalculation[奖励计算]
    end

    subgraph 评估系统
        Evaluator[模型评估器] --> Metrics[性能指标]
        Evaluator --> Reporter[报告生成]
    end

    Game <--> DQN
    DQN --> Evaluator
```

## 2. 状态转换图

```mermaid
stateDiagram-v2
    [*] --> 初始化: 游戏启动
    初始化 --> 游戏中: 开始游戏
    游戏中 --> 暂停: ESC键
    暂停 --> 游戏中: 继续游戏
    游戏中 --> 游戏结束: 生命值为0
    游戏中 --> 胜利: 通过所有关卡
    游戏结束 --> 初始化: 重新开始
    胜利 --> 初始化: 重新开始
```

## 3. 交互流程图

```mermaid
sequenceDiagram
    participant Player as 玩家/AI
    participant Game as 游戏系统
    participant Physics as 物理系统
    participant Reward as 奖励系统

    loop 游戏循环
        Game->>Player: 提供当前状态
        Player->>Game: 选择动作(左移/右移/射击)
        Game->>Physics: 更新物理状态
        Physics->>Game: 返回碰撞结果
        Game->>Reward: 计算奖励
        Reward->>Player: 返回奖励值
        Game->>Game: 更新游戏状态
    end
```

## 4. 核心功能说明

### 4.1 游戏引擎 (SpaceShooter)
- 管理游戏主循环
- 处理物理碰撞
- 控制敌人生成
- 管理得分和生命值

### 4.2 AI系统
- 基于DQN的强化学习
- 状态空间：玩家位置、敌人位置、子弹位置
- 动作空间：左移、右移、射击、左移+射击、右移+射击
- 奖励机制：击中敌人(+)、敌人逃脱(-)、被击中(-)

### 4.3 评估系统
- 记录模型性能指标
- 生成评估报告
- 跟踪最佳模型

## 5. 技术特点

1. **模块化设计**
   - 游戏逻辑、AI、评估系统解耦
   - 便于维护和扩展

2. **可配置性**
   - 支持训练模式和游戏模式
   - 可调节游戏参数（速度、难度等）

3. **实时反馈**
   - 显示游戏状态
   - 展示AI决策过程
   - 实时性能指标