# tic-tac-toe-generator
Tic-Tac-Toe Generator

```python
from tic_tac_toe_generator import TicTacToeGenerator, GameConfig

# Generate 10,000 episodes for pretraining
config = GameConfig(n_episodes=10000)
generator = TicTacToeGenerator(config)
dataset = generator.generate_dataset()
```

```python
from tic_tac_toe_generator import TicTacToeGenerator, GameConfig

# Generate 10,000 episodes for pretraining
config = GameConfig(n_episodes=10000)
generator = TicTacToeGenerator(config)
dataset = generator.generate_dataset()
```

```python
config = GameConfig(
    n_episodes=50000,           # 50K games
    strategy_level=0.7,         # 70% strategic moves
    include_advanced_features=True,
    save_to_file=True,
    output_dir="training_data",
    random_seed=42
)
generator = TicTacToeGenerator(config)
dataset = generator.generate_dataset()
```
