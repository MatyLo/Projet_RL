#!/usr/bin/env python3
"""
Test Simple du Workflow - LineWorld + Q-Learning

Test minimal pour valider que le workflow fonctionne.
"""

import sys
import os
import numpy as np

# Ajout des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

try:
    from src.rl_environments.line_world import LineWorld
    from src.rl_algorithms.temporal_difference.q_learning import QLearning
    from utils.config_loader import load_config
    from utils.agent import Agent
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def simple_test():
    """Test simple et direct."""
    print("🧪 TEST SIMPLE DU WORKFLOW")
    
    try:
        # 1. Créer environnement
        env = LineWorld(line_length=5, start_position=0, target_position=4)
        print(f"✅ Environnement créé: {env.env_name}")
        
        # 2. Créer algorithme avec config simple
        config = {
            'learning_rate': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1,
            'num_episodes': 100
        }
        
        algorithm = QLearning.from_config(config, env)
        print(f"✅ Algorithme créé: {algorithm.algo_name}")
        
        # 3. Entraîner
        print("🚀 Entraînement...")
        algorithm.train(env, num_episodes=100, verbose=False)
        print(f"✅ Entraînement terminé")
        
        # 4. Agent wrapper
        agent = Agent(algorithm, env)
        print(f"✅ Agent créé: {agent.agent_name}")
        
        # 5. Test simple
        results = agent.evaluate_performance(num_episodes=10, verbose=False)
        print(f"✅ Performance: {results['avg_reward']:.2f}")
        
        print("🎉 WORKFLOW VALIDÉ!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()