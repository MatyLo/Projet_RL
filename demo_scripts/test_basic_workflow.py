#!/usr/bin/env python3
"""
Script de test du workflow de base - Line World + Q-Learning

Ce script valide que le workflow complet fonctionne :
1. Création d'un environnement Line World
2. Création d'un algorithme Q-Learning  
3. Création d'un Agent
4. Entraînement
5. Évaluation
6. Démonstration
7. Sauvegarde/Chargement

Usage:
    python demo_scripts/test_basic_workflow.py

Placement: demo_scripts/test_basic_workflow.py
"""

import sys
import os
import numpy as np

# Ajout des chemins pour imports simples
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Imports simples
try:
    from src.rl_environments.line_world import LineWorld, create_simple_lineworld
    from src.rl_algorithms.temporal_difference.q_learning import QLearning, create_standard_qlearning
    from core.agent import Agent, quick_train_and_evaluate
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Vérifiez que vous lancez depuis la racine du projet:")
    print("   python demo_scripts/test_basic_workflow.py")
    sys.exit(1)


def test_environment_creation():
    """Test de création et fonctionnement de base de l'environnement."""
    print("\n" + "="*60)
    print("TEST 1: CRÉATION DE L'ENVIRONNEMENT LINE WORLD")
    print("="*60)
    
    try:
        # Création d'un environnement simple
        env = create_simple_lineworld()
        print(f"✅ Environnement créé: {env}")
        print(f"   - Espace d'états: {env.state_space_size}")
        print(f"   - Espace d'actions: {env.action_space_size}")
        print(f"   - Position cible: {env.target_position}")
        
        # Test de reset
        initial_state = env.reset()
        print(f"   - État initial: {initial_state}")
        
        # Test de quelques actions
        print("\n   Test de 3 actions aléatoires:")
        for i in range(3):
            action = np.random.choice([0, 1])
            action_name = env.ACTION_NAMES[action]
            state, reward, done, info = env.step(action)
            print(f"     Action {action} ({action_name}): état={state}, récompense={reward:.2f}, terminé={done}")
            
            if done:
                print(f"     Episode terminé en {i+1} étapes!")
                break
        
        # Test de rendu
        print(f"\n   État final de l'environnement:")
        env.render('console')
        
        return True, env
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'environnement: {e}")
        return False, None


def test_algorithm_creation():
    """Test de création et fonctionnement de base de l'algorithme."""
    print("\n" + "="*60)
    print("TEST 2: CRÉATION DE L'ALGORITHME Q-LEARNING")
    print("="*60)
    
    try:
        # Création d'un algorithme Q-Learning
        algorithm = create_standard_qlearning(state_space_size=5, action_space_size=2)
        print(f"✅ Algorithme créé: {algorithm}")
        print(f"   - Type: {algorithm.algorithm_type}")
        print(f"   - Hyperparamètres: {algorithm.get_hyperparameters()}")
        
        # Test de sélection d'action avant entraînement
        print(f"\n   Test de sélection d'actions avant entraînement:")
        for state in range(3):
            action = algorithm.select_action(state, training=True)
            print(f"     État {state} -> Action {action}")
        
        # Affichage Q-table initiale
        print(f"\n   Q-table initiale:")
        print(algorithm.visualize_q_table())
        
        return True, algorithm
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'algorithme: {e}")
        return False, None


def test_agent_creation_and_training(env, algorithm):
    """Test de création d'agent et d'entraînement."""
    print("\n" + "="*60)
    print("TEST 3: CRÉATION D'AGENT ET ENTRAÎNEMENT")
    print("="*60)
    
    try:
        # Création de l'agent
        agent = Agent(algorithm, env, "TestAgent_LineWorld_QLearning")
        print(f"✅ Agent créé: {agent}")
        
        # Entraînement rapide
        print(f"\n   Démarrage de l'entraînement (200 épisodes)...")
        train_results = agent.train(
            num_episodes=200,
            verbose=True
        )
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   - Récompense moyenne: {train_results['avg_reward']:.2f}")
        print(f"   - Récompense finale: {train_results['final_episode_reward']:.2f}")
        print(f"   - Temps d'entraînement: {train_results['training_time']:.2f}s")
        
        # Affichage Q-table finale
        print(f"\n   Q-table après entraînement:")
        print(algorithm.visualize_q_table())
        
        return True, agent
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False, None


def test_evaluation_and_demonstration(agent):
    """Test d'évaluation et de démonstration."""
    print("\n" + "="*60)
    print("TEST 4: ÉVALUATION ET DÉMONSTRATION")
    print("="*60)
    
    try:
        # Évaluation de l'agent
        print(f"   Évaluation de l'agent sur 50 épisodes...")
        eval_results = agent.evaluate(num_episodes=50, verbose=True)
        
        print(f"\n✅ Évaluation terminée!")
        print(f"   - Performance moyenne: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"   - Meilleure performance: {eval_results['max_reward']:.2f}")
        print(f"   - Longueur moyenne d'épisode: {eval_results['avg_episode_length']:.1f}")
        
        # Démonstration d'un épisode
        print(f"\n   Démonstration d'un épisode:")
        demo_history = agent.demonstrate(
            num_episodes=1,
            render_mode='console',
            step_by_step=False,
            delay_between_steps=0.5
        )
        
        print(f"\n✅ Démonstration terminée!")
        episode = demo_history[0]
        print(f"   - Récompense de l'épisode: {episode['total_reward']:.2f}")
        print(f"   - Nombre d'étapes: {episode['steps']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation/démonstration: {e}")
        return False


def test_save_and_load(agent):
    """Test de sauvegarde et chargement."""
    print("\n" + "="*60)
    print("TEST 5: SAUVEGARDE ET CHARGEMENT")
    print("="*60)
    
    try:
        # Sauvegarde
        save_path = "outputs/test_agent"
        os.makedirs("outputs", exist_ok=True)
        
        print(f"   Sauvegarde de l'agent...")
        save_success = agent.save_agent(save_path)
        
        if save_success:
            print(f"✅ Agent sauvegardé dans {save_path}")
        else:
            print(f"❌ Échec de la sauvegarde")
            return False
        
        # Test de chargement
        print(f"   Test de chargement...")
        
        # Créer un nouvel agent identique
        new_env = create_simple_lineworld()
        new_algorithm = create_standard_qlearning(
            state_space_size=new_env.state_space_size,
            action_space_size=new_env.action_space_size
        )
        new_agent = Agent(new_algorithm, new_env, "LoadedTestAgent")
        
        load_success = new_agent.load_agent(save_path)
        
        if load_success:
            print(f"✅ Agent chargé avec succès")
            
            # Vérification que l'agent chargé fonctionne
            quick_eval = new_agent.evaluate(num_episodes=10, verbose=False)
            print(f"   - Test de l'agent chargé: récompense moyenne = {quick_eval['avg_reward']:.2f}")
            
        else:
            print(f"❌ Échec du chargement")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde/chargement: {e}")
        return False


def test_policy_analysis(agent):
    """Test d'analyse de la politique apprise."""
    print("\n" + "="*60)
    print("TEST 6: ANALYSE DE LA POLITIQUE APPRISE")
    print("="*60)
    
    try:
        # Affichage de la politique
        print(agent.get_policy_visualization())
        
        # Comparaison avec la politique optimale
        optimal_policy = agent.environment.get_optimal_policy()
        print(f"\n   Politique optimale de référence:")
        for state, action in optimal_policy.items():
            print(f"     État {state}: Action {action}")
        
        # Comparaison
        learned_policy = agent.algorithm.get_policy()
        correct_actions = 0
        total_states = len(optimal_policy)
        
        print(f"\n   Comparaison avec la politique optimale:")
        for state in range(total_states):
            learned_action = learned_policy[state]
            optimal_action = optimal_policy[state]
            is_correct = learned_action == optimal_action
            
            if is_correct:
                correct_actions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"     État {state}: Appris={learned_action}, Optimal={optimal_action} {status}")
        
        accuracy = correct_actions / total_states * 100
        print(f"\n   Précision de la politique: {accuracy:.1f}% ({correct_actions}/{total_states})")
        
        if accuracy >= 80:
            print(f"✅ Politique correcte apprise!")
        else:
            print(f"⚠️  Politique partiellement correcte - pourrait nécessiter plus d'entraînement")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de politique: {e}")
        return False


def main():
    """Fonction principale de test."""
    print("🚀 DÉBUT DES TESTS DU WORKFLOW DE BASE")
    print("🎯 Objectif: Valider Line World + Q-Learning + Agent")
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Environnement
    success, env = test_environment_creation()
    if success:
        tests_passed += 1
    else:
        print("❌ Test d'environnement échoué - arrêt des tests")
        return
    
    # Test 2: Algorithme
    success, algorithm = test_algorithm_creation()
    if success:
        tests_passed += 1
    else:
        print("❌ Test d'algorithme échoué - arrêt des tests")
        return
    
    # Test 3: Agent et entraînement
    success, agent = test_agent_creation_and_training(env, algorithm)
    if success:
        tests_passed += 1
    else:
        print("❌ Test d'agent échoué - arrêt des tests")
        return
    
    # Test 4: Évaluation et démonstration
    success = test_evaluation_and_demonstration(agent)
    if success:
        tests_passed += 1
    
    # Test 5: Sauvegarde et chargement
    success = test_save_and_load(agent)
    if success:
        tests_passed += 1
    
    # Test 6: Analyse de politique
    success = test_policy_analysis(agent)
    if success:
        tests_passed += 1
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    print(f"Tests réussis: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ Le workflow de base fonctionne correctement")
        print("🚀 Vous pouvez maintenant ajouter d'autres algorithmes et environnements")
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) ont échoué")
        print("🔧 Vérifiez les erreurs ci-dessus avant de continuer")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Configuration de NumPy pour des résultats reproductibles
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()