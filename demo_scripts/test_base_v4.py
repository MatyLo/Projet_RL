"""
Test Complet du Workflow Base - Version 4

Ce script teste l'architecture complète du projet RL :
1. Chargement configuration JSON
2. Création environnement LineWorld
3. Entraînement Q-Learning autonome  
4. Agent wrapper post-entraînement
5. Évaluation et démonstration
6. Sauvegarde des résultats

Usage:
    python demo_scripts/test_base_v4.py
    python demo_scripts/test_base_v4.py --config test_config.json
"""

import sys
import os
import json
import time
import argparse
import pickle
from pathlib import Path

# Ajout des chemins pour imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

try:
    from src.rl_environments.line_world import LineWorld, create_lineworld
    from src.rl_algorithms.temporal_difference.q_learning import QLearning
    from utils.agent import Agent
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Vérifiez que vous lancez depuis la racine du projet")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """
    Charge la configuration depuis un fichier JSON.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dict avec la configuration chargée
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configuration chargée: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Fichier de configuration non trouvé: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON dans {config_path}: {e}")
        sys.exit(1)


def create_environment(env_config: dict):
    """
    Crée l'environnement selon la configuration.
    
    Args:
        env_config: Configuration de l'environnement
        
    Returns:
        Instance de l'environnement
    """
    env_type = env_config.get('type', 'lineworld').lower()
    
    if env_type == 'lineworld':
        max_steps = env_config.get('max_steps', 100)
        env = LineWorld(max_steps=max_steps)
        print(f"✅ Environnement créé: {env.env_name} (max_steps={max_steps})")
        return env
    else:
        raise ValueError(f"Type d'environnement non supporté: {env_type}")


def create_algorithm(algo_config: dict, environment):
    """
    Crée l'algorithme selon la configuration.
    
    Args:
        algo_config: Configuration de l'algorithme
        environment: Environnement pour dimensionnement
        
    Returns:
        Instance de l'algorithme
    """
    algo_type = algo_config.get('type', 'q_learning').lower()
    
    if algo_type == 'q_learning':
        algorithm = QLearning.from_config(algo_config, environment)
        print(f"✅ Algorithme créé: {algorithm.algo_name}")
        print(f"   Hyperparamètres: α={algorithm.learning_rate}, γ={algorithm.gamma}, ε={algorithm.epsilon}")
        return algorithm
    else:
        raise ValueError(f"Type d'algorithme non supporté: {algo_type}")


def train_algorithm(algorithm, environment, training_config: dict):
    """
    Entraîne l'algorithme selon la configuration.
    
    Args:
        algorithm: Algorithme à entraîner
        environment: Environnement d'entraînement
        training_config: Configuration d'entraînement
        
    Returns:
        Résultats d'entraînement
    """
    print(f"\n🚀 PHASE 1: ENTRAÎNEMENT AUTONOME")
    print("=" * 50)
    
    num_episodes = training_config.get('num_episodes', 1000)
    verbose = training_config.get('verbose', True)
    
    start_time = time.time()
    training_results = algorithm.train(
        environment=environment,
        num_episodes=num_episodes,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Entraînement terminé en {training_time:.2f}s")
    print(f"Récompense finale: {training_results['final_reward']:.2f}")
    print(f"Épisodes entraînés: {training_results['episodes_trained']}")
    
    return training_results


def evaluate_agent(agent, eval_config: dict):
    """
    Évalue l'agent selon la configuration.
    
    Args:
        agent: Agent à évaluer
        eval_config: Configuration d'évaluation
        
    Returns:
        Résultats d'évaluation
    """
    print(f"\n📊 PHASE 2: ÉVALUATION POST-ENTRAÎNEMENT")
    print("=" * 50)
    
    num_episodes = eval_config.get('num_episodes', 100)
    verbose = eval_config.get('verbose', True)
    
    eval_results = agent.evaluate_performance(
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    return eval_results


def demonstrate_agent(agent, demo_config: dict):
    """
    Démonstration de l'agent selon la configuration.
    
    Args:
        agent: Agent à démontrer
        demo_config: Configuration de démonstration
        
    Returns:
        Résultats de démonstration
    """
    print(f"\n🎬 PHASE 3: DÉMONSTRATION PAS-À-PAS")
    print("=" * 50)
    
    num_episodes = demo_config.get('num_episodes', 1)
    show_q_values = demo_config.get('show_q_values', True)
    pause_between_steps = demo_config.get('pause_between_steps', False)
    
    demo_results = agent.demonstrate_step_by_step(
        num_episodes=num_episodes,
        show_q_values=show_q_values,
        pause_between_steps=pause_between_steps
    )
    
    return demo_results


def save_results(algorithm, agent, training_results, eval_results, demo_results, 
                output_config: dict, experiment_config: dict):
    """
    Sauvegarde tous les résultats selon la configuration.
    
    Args:
        algorithm: Algorithme entraîné
        agent: Agent wrapper
        training_results: Résultats d'entraînement
        eval_results: Résultats d'évaluation  
        demo_results: Résultats de démonstration
        output_config: Configuration de sortie
        experiment_config: Configuration de l'expérience
    """
    if not output_config.get('save_results', True):
        print("\n💾 Sauvegarde désactivée dans la configuration")
        return
    
    print(f"\n💾 PHASE 4: SAUVEGARDE DES RÉSULTATS")
    print("=" * 50)
    
    # Création du répertoire de sortie
    base_dir = output_config.get('base_dir', 'outputs/test_v4')
    os.makedirs(base_dir, exist_ok=True)
    
    # Sauvegarde du modèle (déjà en JSON + pickle dans l'algorithme)
    if output_config.get('save_model', True):
        model_path = f"{base_dir}/model"
        if algorithm.save_model(model_path):
            print(f"✅ Modèle sauvegardé: {model_path}")
    
    # Sauvegarde des résultats de l'agent (déjà en JSON dans l'agent)
    agent_path = f"{base_dir}/agent"
    if agent.save_results(agent_path):
        print(f"✅ Résultats agent sauvegardés: {agent_path}")
    
    # Sauvegarde du résumé simple en pickle (pas de problème NumPy)
    summary_data = {
        "experiment": experiment_config,
        "training_results": training_results,
        "evaluation_results": eval_results,
        "demonstration_results": demo_results,
        "agent_summary": agent.get_performance_summary()
    }
    
    summary_path = f"{base_dir}/experiment_summary.pkl"
    try:
        with open(summary_path, 'wb') as f:
            pickle.dump(summary_data, f)
        print(f"✅ Résumé expérience sauvegardé: {summary_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde résumé: {e}")


def display_final_summary(training_results, eval_results, demo_results):
    """
    Affiche le résumé final de l'expérience.
    
    Args:
        training_results: Résultats d'entraînement
        eval_results: Résultats d'évaluation
        demo_results: Résultats de démonstration
    """
    print(f"\n🎉 RÉSUMÉ FINAL DE L'EXPÉRIENCE")
    print("=" * 50)
    
    print(f"📈 ENTRAÎNEMENT:")
    print(f"   Épisodes: {training_results['episodes_trained']}")
    print(f"   Récompense finale: {training_results['final_reward']:.2f}")
    
    print(f"\n📊 ÉVALUATION:")
    print(f"   Épisodes testés: {eval_results['num_episodes']}")
    print(f"   Récompense moyenne: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Taux de succès: {eval_results['success_rate']:.1%}")
    print(f"   Longueur moyenne: {eval_results['avg_episode_length']:.1f} étapes")
    
    print(f"\n🎬 DÉMONSTRATION:")
    print(f"   Épisodes démontrés: {len(demo_results)}")
    for i, demo in enumerate(demo_results, 1):
        success_icon = "✅" if demo['success'] else "❌"
        print(f"   Épisode {i}: {demo['total_reward']:.2f} points, {len(demo['steps'])} étapes {success_icon}")
    
    # Évaluation globale
    if eval_results['success_rate'] > 0.8 and eval_results['avg_reward'] > 5.0:
        print(f"\n🏆 RÉSULTAT: EXCELLENT! L'agent a très bien appris.")
    elif eval_results['success_rate'] > 0.5:
        print(f"\n👍 RÉSULTAT: BON! L'agent a correctement appris.")
    else:
        print(f"\n⚠️ RÉSULTAT: À améliorer. L'agent pourrait nécessiter plus d'entraînement.")


def main():
    """Fonction principale du test complet."""
    parser = argparse.ArgumentParser(description="Test complet du workflow RL")
    parser.add_argument('--config', '-c', default='experiments/configs/test_config.json',
                       help='Chemin vers le fichier de configuration JSON')
    args = parser.parse_args()
    
    print("🧪 TEST COMPLET DU WORKFLOW BASE - VERSION 4")
    print("=" * 60)
    print("🎯 Objectif: Valider LineWorld + Q-Learning + Agent")
    print("📋 Workflow: Config → Environnement → Entraînement → Évaluation → Démo")
    print("=" * 60)
    
    try:
        # 1. Chargement de la configuration
        print(f"\n📁 ÉTAPE 1: CHARGEMENT CONFIGURATION")
        config = load_config(args.config)
        print(f"Expérience: {config['experiment']['name']}")
        print(f"Description: {config['experiment']['description']}")
        
        # 2. Création de l'environnement
        print(f"\n🌍 ÉTAPE 2: CRÉATION ENVIRONNEMENT")
        environment = create_environment(config['environment'])
        
        # Test rapide de l'environnement
        state = environment.reset()
        print(f"État initial: {state}")
        environment.render('console')
        
        # 3. Création de l'algorithme
        print(f"\n🧠 ÉTAPE 3: CRÉATION ALGORITHME")
        algorithm = create_algorithm(config['algorithm'], environment)
        
        # 4. Entraînement autonome
        training_results = train_algorithm(algorithm, environment, config['training'])
        
        # 5. Création de l'agent wrapper
        print(f"\n🤖 ÉTAPE 4: CRÉATION AGENT WRAPPER")
        agent = Agent(algorithm, environment, f"TestAgent_{config['experiment']['name']}")
        
        # 6. Évaluation
        eval_results = evaluate_agent(agent, config['evaluation'])
        
        # 7. Démonstration
        demo_results = demonstrate_agent(agent, config['demonstration'])
        
        # 8. Sauvegarde
        save_results(algorithm, agent, training_results, eval_results, demo_results,
                    config['outputs'], config['experiment'])
        
        # 9. Résumé final
        display_final_summary(training_results, eval_results, demo_results)
        
        print(f"\n✅ TEST COMPLET RÉUSSI!")
        print("🎓 L'architecture de base est validée et fonctionnelle.")
        print("🚀 Prêt pour l'extension avec d'autres algorithmes et environnements!")
        
    except KeyboardInterrupt:
        print(f"\n\n⏸️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n💥 Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()