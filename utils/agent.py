"""
Agent Hybride - Wrapper post-entraînement pour algorithmes RL.

Cette classe encapsule un algorithme entraîné et fournit des fonctionnalités
d'évaluation, de démonstration et de comparaison pour la phase post-entraînement.

Architecture Hybride :
- Phase 1 : Algorithmes autonomes avec configuration JSON
- Phase 2 : Agent wrapper pour évaluation/démonstration/comparaison

Placement: utils/agent.py
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import json
from datetime import datetime

from src.rl_algorithms.base_algorithm import BaseAlgorithm
from src.rl_environments.base_environment import BaseEnvironment


class Agent:
    """
    Agent wrapper pour la phase post-entraînement.
    
    Cette classe fait le lien entre un algorithme entraîné et un environnement
    pour l'évaluation, la démonstration et l'analyse des performances.
    """
    
    def __init__(self, 
                 algorithm: BaseAlgorithm,
                 environment: BaseEnvironment,
                 agent_name: str = None):
        """
        Initialise l'agent wrapper.
        
        Args:
            algorithm (BaseAlgorithm): Algorithme d'apprentissage entraîné
            environment (BaseEnvironment): Environnement de test
            agent_name (str, optional): Nom de l'agent pour identification
        """
        self.algorithm = algorithm
        self.environment = environment
        self.agent_name = agent_name or f"{algorithm.algo_name}_{environment.env_name}_Agent"
        
        # Vérification de compatibilité
        self._validate_compatibility()
        
        # Historique des évaluations
        self.evaluation_history = []
        self.demonstration_history = []
        
        # Métadonnées de session
        self.session_info = {
            "created_at": datetime.now().isoformat(),
            "algorithm": algorithm.algo_name,
            "environment": environment.env_name,
            "agent_type": "post_training_wrapper"
        }
    
    def _validate_compatibility(self):
        """Vérifie la compatibilité entre l'algorithme et l'environnement."""
        if self.algorithm.state_space_size != self.environment.state_space_size:
            raise ValueError(
                f"Incompatibilité d'espace d'états: "
                f"algorithme={self.algorithm.state_space_size}, "
                f"environnement={self.environment.state_space_size}"
            )
        
        if self.algorithm.action_space_size != self.environment.action_space_size:
            raise ValueError(
                f"Incompatibilité d'espace d'actions: "
                f"algorithme={self.algorithm.action_space_size}, "
                f"environnement={self.environment.action_space_size}"
            )
        
        print(f"✅ Agent wrapper créé: {self.agent_name}")
    
    def evaluate_performance(self, 
                           num_episodes: int = 100,
                           max_steps_per_episode: int = 1000,
                           verbose: bool = True,
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Évalue les performances de l'algorithme entraîné.
        
        Args:
            num_episodes (int): Nombre d'épisodes d'évaluation
            max_steps_per_episode (int): Nombre maximum d'étapes par épisode
            verbose (bool): Affichage des informations détaillées
            save_results (bool): Sauvegarde des résultats d'évaluation
            
        Returns:
            Dict[str, Any]: Statistiques d'évaluation complètes
        """
        if not self.algorithm.is_trained:
            raise ValueError("L'algorithme doit être entraîné avant l'évaluation")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ÉVALUATION DE L'AGENT: {self.agent_name}")
            print(f"{'='*60}")
            print(f"Épisodes d'évaluation: {num_episodes}")
        
        start_time = time.time()
        
        # Métriques de performance
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Sélection d'action sans exploration (politique gloutonne)
                action = self.algorithm.select_action(state, training=False)
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    # Vérifier si c'est un succès (atteint la cible)
                    if info.get("target_reached", False):
                        success_rate += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            if verbose and (episode + 1) % (num_episodes // 10) == 0:
                progress = (episode + 1) / num_episodes * 100
                print(f"Progression: {progress:.0f}% - Récompense moyenne actuelle: {np.mean(episode_rewards):.2f}")
        
        evaluation_time = time.time() - start_time
        success_rate = success_rate / num_episodes
        
        # Calcul des statistiques
        evaluation_results = {
            "agent_name": self.agent_name,
            "num_episodes": num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
            "success_rate": success_rate,
            "evaluation_time": evaluation_time,
            "timestamp": datetime.now().isoformat(),
            "algorithm_config": self.algorithm.get_hyperparameters(),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
        
        # Sauvegarde dans l'historique
        if save_results:
            self.evaluation_history.append(evaluation_results)
        
        if verbose:
            print(f"\n📊 RÉSULTATS D'ÉVALUATION:")
            print(f"Récompense moyenne: {evaluation_results['avg_reward']:.3f} ± {evaluation_results['std_reward']:.3f}")
            print(f"Récompense min/max: {evaluation_results['min_reward']:.2f} / {evaluation_results['max_reward']:.2f}")
            print(f"Longueur d'épisode: {evaluation_results['avg_episode_length']:.1f} ± {evaluation_results['std_episode_length']:.1f}")
            print(f"Taux de succès: {success_rate:.2%}")
            print(f"Temps d'évaluation: {evaluation_time:.2f}s")
            print(f"{'='*60}\n")
        
        return evaluation_results
    
    def demonstrate_policy(self, 
                         num_episodes: int = 1,
                         step_by_step: bool = True,
                         render_mode: str = 'console',
                         delay_between_steps: float = 1.0,
                         show_q_values: bool = False,
                         save_demo: bool = True) -> List[Dict[str, Any]]:
        """
        Démontre la politique apprise (pour soutenance).
        
        Args:
            num_episodes (int): Nombre d'épisodes à démontrer
            step_by_step (bool): Pause entre chaque étape
            render_mode (str): Mode d'affichage ('console' ou 'pygame')
            delay_between_steps (float): Délai entre les étapes
            show_q_values (bool): Afficher les Q-values
            save_demo (bool): Sauvegarder la démonstration
            
        Returns:
            List[Dict[str, Any]]: Historique détaillé des démonstrations
        """
        if not self.algorithm.is_trained:
            raise ValueError("L'algorithme doit être entraîné avant la démonstration")
        
        print(f"\n🎬 DÉMONSTRATION DE LA POLITIQUE APPRISE")
        print(f"Agent: {self.agent_name}")
        print(f"Mode: {render_mode} | Pas-à-pas: {step_by_step}")
        print(f"{'='*60}")
        
        demonstrations = []
        
        for episode_num in range(num_episodes):
            print(f"\n--- 🎯 Épisode {episode_num + 1}/{num_episodes} ---")
            
            # Initialisation de l'épisode
            state = self.environment.reset()
            episode_demo = {
                "episode": episode_num + 1,
                "steps": [],
                "total_reward": 0.0,
                "success": False
            }
            
            # Affichage initial
            print(f"État initial: {state} - {self.environment.get_state_description(state)}")
            self.environment.render(mode=render_mode)
            
            if show_q_values and hasattr(self.algorithm, 'q_function'):
                self._show_q_values_for_state(state)
            
            if step_by_step:
                input("▶️ Appuyez sur Entrée pour commencer...")
            else:
                time.sleep(delay_between_steps)
            
            # Déroulement de l'épisode
            max_steps = getattr(self.environment, 'max_steps', 1000)
            for step in range(max_steps):
                # Sélection d'action optimale
                action = self.algorithm.select_action(state, training=False)
                
                # Exécution de l'action
                next_state, reward, done, info = self.environment.step(action)
                
                # Enregistrement de l'étape
                step_info = {
                    "step": step + 1,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "state_description": self.environment.get_state_description(state),
                    "info": info
                }
                episode_demo["steps"].append(step_info)
                episode_demo["total_reward"] += reward
                
                # Affichage de l'étape
                print(f"\n⏯️ Étape {step + 1}:")
                print(f"   État: {state} → Action: {action} → Nouvel état: {next_state}")
                print(f"   Récompense: {reward} | Récompense totale: {episode_demo['total_reward']:.2f}")
                
                if info.get("target_reached", False):
                    print("   🎉 CIBLE ATTEINTE!")
                    episode_demo["success"] = True
                
                self.environment.render(mode=render_mode)
                
                if show_q_values and hasattr(self.algorithm, 'q_function'):
                    self._show_q_values_for_state(next_state)
                
                state = next_state
                
                if done:
                    print(f"\n✅ Épisode terminé à l'étape {step + 1}")
                    break
                
                # Pause entre les étapes
                if step_by_step:
                    input("▶️ Appuyez sur Entrée pour l'étape suivante...")
                else:
                    time.sleep(delay_between_steps)
            
            # Résumé de l'épisode
            print(f"\n📋 RÉSUMÉ ÉPISODE {episode_num + 1}:")
            print(f"   Récompense totale: {episode_demo['total_reward']:.2f}")
            print(f"   Nombre d'étapes: {len(episode_demo['steps'])}")
            print(f"   Succès: {'✅' if episode_demo['success'] else '❌'}")
            
            demonstrations.append(episode_demo)
        
        # Sauvegarde de la démonstration
        if save_demo:
            self.demonstration_history.append({
                "timestamp": datetime.now().isoformat(),
                "demonstrations": demonstrations,
                "parameters": {
                    "num_episodes": num_episodes,
                    "step_by_step": step_by_step,
                    "render_mode": render_mode,
                    "show_q_values": show_q_values
                }
            })
        
        return demonstrations
    
    def _show_q_values_for_state(self, state: int):
        """Affiche les Q-values pour un état donné."""
        if hasattr(self.algorithm, 'q_function'):
            q_values = self.algorithm.q_function[state]
            print(f"   Q-values état {state}: {q_values}")
            best_action = np.argmax(q_values)
            print(f"   Meilleure action: {best_action} (Q={q_values[best_action]:.3f})")
    
    def compare_with_other_agents(self, other_agents: List['Agent'], 
                                num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare les performances avec d'autres agents.
        
        Args:
            other_agents (List[Agent]): Autres agents à comparer
            num_episodes (int): Nombre d'épisodes pour la comparaison
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        all_agents = [self] + other_agents
        comparison_results = {}
        
        print(f"\n🏆 COMPARAISON D'AGENTS")
        print(f"Environnement: {self.environment.env_name}")
        print(f"Épisodes de test: {num_episodes}")
        print(f"{'='*60}")
        
        for agent in all_agents:
            if not agent.algorithm.is_trained:
                print(f"⚠️ Agent {agent.agent_name} non entraîné - ignoré")
                continue
            
            print(f"🔍 Évaluation de {agent.agent_name}...")
            eval_results = agent.evaluate_performance(
                num_episodes=num_episodes,
                verbose=False
            )
            
            comparison_results[agent.agent_name] = {
                "algorithm": agent.algorithm.algo_name,
                "avg_reward": eval_results["avg_reward"],
                "std_reward": eval_results["std_reward"],
                "success_rate": eval_results["success_rate"],
                "avg_episode_length": eval_results["avg_episode_length"],
                "hyperparameters": agent.algorithm.get_hyperparameters()
            }
        
        # Classement par performance moyenne
        if comparison_results:
            sorted_results = sorted(comparison_results.items(), 
                                   key=lambda x: x[1]["avg_reward"], 
                                   reverse=True)
            
            print(f"\n🏆 CLASSEMENT DES AGENTS:")
            print(f"{'Rang':<5}{'Agent':<25}{'Algorithme':<15}{'Récompense':<15}{'Succès':<10}")
            print("-" * 70)
            
            for rank, (agent_name, result) in enumerate(sorted_results, 1):
                print(f"{rank:<5}{agent_name[:24]:<25}{result['algorithm']:<15}"
                      f"{result['avg_reward']:<15.3f}{result['success_rate']:<10.2%}")
        
        comparison_summary = {
            "environment": self.environment.env_name,
            "num_episodes": num_episodes,
            "num_agents": len(comparison_results),
            "results": comparison_results,
            "best_agent": sorted_results[0][0] if comparison_results else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison_summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des performances de l'agent.
        
        Returns:
            Dict[str, Any]: Résumé complet des performances
        """
        summary = {
            "agent_info": {
                "name": self.agent_name,
                "algorithm": self.algorithm.algo_name,
                "environment": self.environment.env_name,
                "is_trained": self.algorithm.is_trained
            },
            "session_info": self.session_info,
            "algorithm_config": self.algorithm.get_hyperparameters()
        }
        
        # Dernière évaluation
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-1]
            summary["latest_evaluation"] = {
                "avg_reward": latest_eval["avg_reward"],
                "success_rate": latest_eval["success_rate"],
                "avg_episode_length": latest_eval["avg_episode_length"],
                "timestamp": latest_eval["timestamp"]
            }
        
        # Historique des évaluations
        if len(self.evaluation_history) > 1:
            rewards = [eval_data["avg_reward"] for eval_data in self.evaluation_history]
            summary["evaluation_trends"] = {
                "num_evaluations": len(self.evaluation_history),
                "improvement": rewards[-1] - rewards[0] if len(rewards) >= 2 else 0.0,
                "best_performance": max(rewards),
                "worst_performance": min(rewards)
            }
        
        return summary
    
    def save_agent_results(self, filepath: str) -> bool:
        """
        Sauvegarde tous les résultats de l'agent.
        
        Args:
            filepath (str): Chemin de sauvegarde (sans extension)
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            results_data = {
                "agent_info": {
                    "name": self.agent_name,
                    "algorithm": self.algorithm.algo_name,
                    "environment": self.environment.env_name
                },
                "session_info": self.session_info,
                "evaluation_history": self.evaluation_history,
                "demonstration_history": self.demonstration_history,
                "performance_summary": self.get_performance_summary(),
                "algorithm_config": self.algorithm.get_hyperparameters()
            }
            
            with open(f"{filepath}_agent_results.json", 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"✅ Résultats de l'agent sauvegardés: {filepath}_agent_results.json")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def __str__(self) -> str:
        """Représentation textuelle de l'agent."""
        status = "entraîné" if self.algorithm.is_trained else "non entraîné"
        return f"Agent({self.agent_name}, {status})"
    
    def __repr__(self) -> str:
        """Représentation détaillée de l'agent."""
        return (f"Agent(name='{self.agent_name}', "
                f"algorithm={self.algorithm.algo_name}, "
                f"environment={self.environment.env_name})")


if __name__ == "__main__":
    # Test de la classe Agent
    print("🧪 Test de la classe Agent wrapper")
    print("Classe Agent prête pour utilisation avec algorithmes entraînés!")