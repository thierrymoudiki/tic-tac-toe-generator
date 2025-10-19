import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GameConfig:
    """Configuration for game generation"""
    n_episodes: int = 10000
    strategy_level: float = 0.6
    include_advanced_features: bool = True
    save_to_file: bool = True
    output_dir: str = "data"
    random_seed: Optional[int] = 42

class TicTacToeGenerator:
    """
    A comprehensive class for generating Tic-Tac-Toe game datasets
    for pretraining machine learning models.
    """
    
    def __init__(self, config: GameConfig = GameConfig()):
        self.config = config
        self.win_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns  
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Create output directory
        if self.config.save_to_file:
            Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def empty_board(self) -> List[str]:
        """Return an empty Tic-Tac-Toe board"""
        return ["-"] * 9
    
    def available_actions(self, board: List[str]) -> List[int]:
        """Get list of available moves"""
        return [i for i, cell in enumerate(board) if cell == "-"]
    
    def is_winner(self, board: List[str], player: str) -> bool:
        """Check if player has won"""
        return any(all(board[i] == player for i in pos) for pos in self.win_positions)
    
    def is_draw(self, board: List[str]) -> bool:
        """Check if game is a draw"""
        return "-" not in board
    
    def make_move(self, board: List[str], action: int, player: str) -> Tuple[List[str], float, bool]:
        """
        Execute a move and return new board, reward, and done flag
        
        Returns:
            new_board: Updated board state
            reward: 1 for win, -1 for loss, 0 for draw/continue  
            done: Whether game ended
        """
        if board[action] != "-":
            raise ValueError(f"Position {action} is already occupied")
        
        new_board = board.copy()
        new_board[action] = player
        
        reward = 0.0
        done = False
        
        if self.is_winner(new_board, player):
            reward = 1.0
            done = True
        elif self.is_draw(new_board):
            reward = 0.0
            done = True
            
        return new_board, reward, done
    
    def get_board_features(self, board: List[str], player: str) -> List[int]:
        """
        Convert board state to numerical features
        
        Returns:
            List of features: [cell_0..cell_8, player, x_count, o_count, empty_count]
        """
        # Board encoding: 0=empty, 1=X, 2=O
        board_features = [0 if cell == "-" else (1 if cell == "X" else 2) for cell in board]
        
        # Player encoding: 1=X, 2=O
        player_encoding = 1 if player == "X" else 2
        
        if not self.config.include_advanced_features:
            return board_features + [player_encoding]
        
        # Advanced features
        x_count = board.count("X")
        o_count = board.count("O") 
        empty_count = board.count("-")
        
        return board_features + [player_encoding, x_count, o_count, empty_count]
    
    def strategic_move(self, board: List[str], player: str) -> int:
        """
        Choose a move using basic Tic-Tac-Toe strategy
        """
        available = self.available_actions(board)
        opponent = "O" if player == "X" else "X"
        
        # 1. Check for winning move
        for action in available:
            test_board = board.copy()
            test_board[action] = player
            if self.is_winner(test_board, player):
                return action
        
        # 2. Block opponent's winning move
        for action in available:
            test_board = board.copy()
            test_board[action] = opponent  
            if self.is_winner(test_board, opponent):
                return action
        
        # 3. Take center if available
        if 4 in available:
            return 4
        
        # 4. Take corners if available
        corners = [0, 2, 6, 8]
        available_corners = [pos for pos in corners if pos in available]
        if available_corners:
            return random.choice(available_corners)
        
        # 5. Take edges
        edges = [1, 3, 5, 7]
        available_edges = [pos for pos in edges if pos in available]
        if available_edges:
            return random.choice(available_edges)
        
        # Fallback to random
        return random.choice(available)
    
    def choose_action(self, board: List[str], player: str) -> int:
        """Choose action mixing strategy and randomness"""
        available = self.available_actions(board)
        
        if not available:
            raise ValueError("No available actions")
        
        # Mix of strategic and random moves based on strategy_level
        if random.random() < self.config.strategy_level:
            return self.strategic_move(board, player)
        else:
            return random.choice(available)
    
    def generate_episode(self, episode_id: int) -> List[List]:
        """
        Generate a single complete game episode
        
        Returns:
            List of game states: [episode, step, features..., action, player, reward, done]
        """
        board = self.empty_board()
        player = "X"
        step_id = 0
        episode_data = []
        game_history = []
        
        while True:
            # Get current state features
            state_features = self.get_board_features(board, player)
            available_actions = self.available_actions(board)
            
            if not available_actions:
                break
                
            # Choose and execute action
            action = self.choose_action(board, player)
            next_board, reward, done = self.make_move(board, action, player)
            
            # Final reward assignment
            final_reward = 0.0
            if done:
                if reward == 1:
                    final_reward = 1.0 if player == "X" else -1.0
                else:
                    final_reward = 0.0
            
            # Record game state
            row = [episode_id, step_id] + state_features + [action, player, final_reward, int(done)]
            episode_data.append(row)
            
            # Store move for game history
            game_history.append({
                'step': step_id,
                'player': player,
                'action': action,
                'board': board.copy(),
                'reward': final_reward
            })
            
            # Update for next step
            board = next_board
            player = "O" if player == "X" else "X"
            step_id += 1
            
            if done:
                break
        
        return episode_data
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete Tic-Tac-Toe dataset
        
        Returns:
            pandas DataFrame with all game episodes
        """
        print(f"🎮 Generating {self.config.n_episodes} Tic-Tac-Toe episodes...")
        
        all_data = []
        game_stats = {
            'x_wins': 0,
            'o_wins': 0, 
            'draws': 0,
            'total_moves': 0
        }
        
        for episode_id in range(self.config.n_episodes):
            if (episode_id + 1) % 1000 == 0:
                print(f"  Generated {episode_id + 1}/{self.config.n_episodes} episodes...")
            
            episode_data = self.generate_episode(episode_id)
            all_data.extend(episode_data)
            
            # Update statistics
            final_reward = episode_data[-1][-2]  # reward is second last column
            if final_reward == 1:
                game_stats['x_wins'] += 1
            elif final_reward == -1:
                game_stats['o_wins'] += 1
            else:
                game_stats['draws'] += 1
            game_stats['total_moves'] += len(episode_data)
        
        # Create DataFrame
        feature_names = [f"cell_{i}" for i in range(9)] + ["player"]
        if self.config.include_advanced_features:
            feature_names.extend(["x_count", "o_count", "empty_count"])
            
        columns = ["episode", "step"] + feature_names + ["action", "player_symbol", "reward", "done"]
        
        df = pd.DataFrame(all_data, columns=columns)
        
        # Print summary
        self._print_summary(df, game_stats)
        
        # Save to file if requested
        if self.config.save_to_file:
            self._save_dataset(df, game_stats)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame, stats: Dict):
        """Print dataset summary statistics"""
        print(f"\n✅ Dataset Generation Complete!")
        print("=" * 50)
        print(f"📊 Dataset Statistics:")
        print(f"   Total episodes: {self.config.n_episodes:,}")
        print(f"   Total moves: {stats['total_moves']:,}")
        print(f"   Average moves per game: {stats['total_moves'] / self.config.n_episodes:.1f}")
        print(f"   X wins: {stats['x_wins']} ({stats['x_wins']/self.config.n_episodes*100:.1f}%)")
        print(f"   O wins: {stats['o_wins']} ({stats['o_wins']/self.config.n_episodes*100:.1f}%)")
        print(f"   Draws: {stats['draws']} ({stats['draws']/self.config.n_episodes*100:.1f}%)")
        print(f"   Strategy level: {self.config.strategy_level}")
        print(f"   Features per state: {len(df.columns) - 4}")  # excluding metadata columns
    
    def _save_dataset(self, df: pd.DataFrame, stats: Dict):
        """Save dataset and metadata to files"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"tic_tac_toe_{self.config.n_episodes}ep_{timestamp}"
        
        # Save dataset
        csv_path = Path(self.config.output_dir) / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'generation_timestamp': timestamp,
            'n_episodes': self.config.n_episodes,
            'strategy_level': self.config.strategy_level,
            'include_advanced_features': self.config.include_advanced_features,
            'total_moves': len(df),
            'game_stats': stats,
            'columns': df.columns.tolist(),
            'feature_descriptions': {
                'episode': 'Game identifier (0 to n_episodes-1)',
                'step': 'Move number within game (0-based)',
                'cell_0 to cell_8': 'Board positions: 0=empty, 1=X, 2=O',
                'player': 'Current player: 1=X, 2=O', 
                'x_count': 'Number of X pieces on board',
                'o_count': 'Number of O pieces on board',
                'empty_count': 'Number of empty positions',
                'action': 'Move made (0-8 board position)',
                'player_symbol': 'Player as symbol: X or O',
                'reward': 'Game outcome: 1=X win, -1=O win, 0=draw',
                'done': 'Game ended: 1=true, 0=false'
            }
        }
        
        metadata_path = Path(self.config.output_dir) / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Dataset saved to: {csv_path}")
        print(f"📄 Metadata saved to: {metadata_path}")
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """Get detailed information about the generated dataset"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'episodes': df['episode'].nunique(),
            'total_moves': len(df),
            'avg_moves_per_game': len(df) / df['episode'].nunique(),
            'action_distribution': df['action'].value_counts().to_dict(),
            'outcome_distribution': df.groupby('episode')['reward'].last().value_counts().to_dict(),
            'player_distribution': df['player_symbol'].value_counts().to_dict()
        }
        return info


# Example usage and demonstration
def demonstrate_generator():
    """Demonstrate how to use the TicTacToeGenerator class"""
    
    print("🚀 TIC-TAC-TOE DATASET GENERATOR DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Small dataset for testing
    print("\n1. Generating small test dataset (100 episodes)...")
    config_small = GameConfig(
        n_episodes=100,
        strategy_level=0.5,
        include_advanced_features=True,
        save_to_file=False
    )
    
    generator_small = TicTacToeGenerator(config_small)
    df_small = generator_small.generate_dataset()
    
    # Example 2: Production dataset
    print("\n2. Generating production dataset (10,000 episodes)...")
    config_prod = GameConfig(
        n_episodes=10000,
        strategy_level=0.6, 
        include_advanced_features=True,
        save_to_file=True,
        output_dir="tic_tac_toe_data"
    )
    
    generator_prod = TicTacToeGenerator(config_prod)
    df_prod = generator_prod.generate_dataset()
    
    # Show dataset info
    info = generator_prod.get_dataset_info(df_prod)
    print(f"\n📋 Dataset Information:")
    print(f"   Shape: {info['shape']}")
    print(f"   Episodes: {info['episodes']:,}")
    print(f"   Total moves: {info['total_moves']:,}")
    print(f"   Avg moves/game: {info['avg_moves_per_game']:.1f}")
    
    return generator_prod, df_prod


# Quick usage example
def quick_start():
    """Quick start example for immediate use"""
    config = GameConfig(n_episodes=1000)  # Change to 10000 for production
    generator = TicTacToeGenerator(config)
    dataset = generator.generate_dataset()
    return dataset


if __name__ == "__main__":
    # Run demonstration
    generator, dataset = demonstrate_generator()
    
    print(f"\n🎯 Your dataset is ready!")
    print(f"   Use: dataset = quick_start() to generate a new dataset")
    print(f"   Or customize: config = GameConfig(n_episodes=10000, ...)")
