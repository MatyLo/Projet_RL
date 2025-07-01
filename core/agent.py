"""
Classe Agent - Orchestrateur central entre algorithmes et environnements.

L'Agent encapsule un algorithme d'apprentissage par renforcement et un environnement,
fournissant une interface unifiée pour l'entraînement, l'évaluation et la démonstration.

Placement: core/agent.py
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import simple depuis src
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from src.rl_algorithms.base_algorithm import BaseAlgorithm
from src.rl_environments.base_environment import BaseEnvironment


class Agent:
    """
    Classe Agent pour orchestrer l'apprentissage par renforcement.
    
    Cette classe fait le lien entre un algorithme et un environnement,
    facilitant l'entraînement, l'évaluation et l'analyse des performances.
    """
    
    def __init__(self, 
                 algorithm: BaseAlgorithm,
                 environment: BaseEnvironment,
                 agent_name: str = None):
        """
        Initialise l'agent avec un algorithme et un environnement.
        
        Args:
            algorithm (BaseAlgorithm): Algorithme d'apprentissage par renforcement
            environment (BaseEnvironment): Environnement d'entraînement
            agent_name (str, optional): Nom de l'agent pour identification
        """
        self.algorithm = algorithm
        self.environment = environment
        self.agent_name = agent_name or f"{algorithm.algo_name}_{environment.env_name}"
        
        # Vérification de compatibilité
        self._validate_compatibility()
        
        # Historique des performances
        self.training_history = []
        self.evaluation_history = []
        
        # Métadonnées de session
        self.session_info = {
            "created_at": datetime.now().isoformat(),
            "algorithm": algorithm.algo_name,
            "environment": environment.env_name,
            "state_space_size": environment.state_space_size,
            "action_space_size": environment.action_space_size
        }
    
    def _validate_compatibility(self):
        """Vérifie la compatibilité entre l'algorithme et l'environnement."""
        # Vérification des tailles d'espaces
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
        
        print(f"✅ Agent créé avec succès: {self.agent_name}")
    
    def train(self, 
              num_episodes: int,
              verbose: bool = True,
              **kwargs) -> Dict[str, Any]:
        """
        Entraîne l'agent sur l'environnement.
        
        Args:
            num_episodes (int): Nombre d'épisodes d'entraînement
            verbose (bool): Affichage des informations de progression
            **kwargs: Paramètres supplémentaires pour l'algorithme
            
        Returns:
            Dict[str, Any]: Résultats d'entraînement
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT DE L'AGENT: {self.agent_name}")
            print(f"{'='*60}")
            print(f"Algorithme: {self.algorithm.algo_name}")
            print(f"Environnement: {self.environment.env_name}")
            print(f"Épisodes: {num_episodes}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Entraînement de l'algorithme
        training_results = self.algorithm.train(
            environment=self.environment,
            num_episodes=num_episodes,
            verbose=verbose,
            **kwargs
        )
        
        training_time = time.time() - start_time
        
        # Ajout des métadonnées de session
        training_results.update({
            "agent_name": self.agent_name,
            "training_time": training_time,
            "episodes_per_second": num_episodes / training_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Sauvegarde de l'historique
        self.training_history.append(training_results)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT TERMINÉ")
            print(f"Temps total: {training_time:.2f}s")
            print(f"Vitesse: {num_episodes/training_time:.1f} épisodes/sec")
            print(f"Récompense moyenne: {training_results.get('avg_reward', 'N/A')}")
            print(f"Récompense finale: {training_results.get('final_episode_reward', 'N/A')}")
            print(f"{'='*60}\n")
        
        return training_results
    
    def evaluate(self, 
                 num_episodes: int = 100,
                 max_steps_per_episode: int = 1000,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Évalue les performances de l'agent entraîné.
        
        Args:
            num_episodes (int): Nombre d'épisodes d'évaluation
            max_steps_per_episode (int): Nombre maximum d'étapes par épisode
            verbose (bool): Affichage des informations détaillées
            
        Returns:
            Dict[str, Any]: Statistiques d'évaluation
        """
        if not self.algorithm.is_trained:
            raise ValueError("L'agent doit être entraîné avant l'évaluation")
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"ÉVALUATION DE L'AGENT: {self.agent_name}")
            print(f"{'='*50}")
        
        start_time = time.time()
        
        # Utilise la méthode d'évaluation de l'algorithme
        evaluation_results = self.algorithm.evaluate_policy(
            environment=self.environment,
            num_episodes=num_episodes,
            max_steps=max_steps_per_episode
        )
        
        evaluation_time = time.time() - start_time
        
        # Ajout des métadonnées
        evaluation_results.update({
            "agent_name": self.agent_name,
            "evaluation_time": evaluation_time,
            "timestamp": datetime.now().isoformat(),
            "episodes_evaluated": num_episodes
        })
        
        # Sauvegarde dans l'historique
        self.evaluation_history.append(evaluation_results)
        
        if verbose:
            print(f"Récompense moyenne: {evaluation_results['avg_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
            print(f"Récompense min/max: {evaluation_results['min_reward']:.2f} / {evaluation_results['max_reward']:.2f}")
            print(f"Longueur moyenne d'épisode: {evaluation_results['avg_episode_length']:.1f} ± {evaluation_results['std_episode_length']:.1f}")
            print(f"Temps d'évaluation: {evaluation_time:.2f}s")
            print(f"{'='*50}\n")
        
        return evaluation_results
    
    def demonstrate(self, 
                   num_episodes: int = 1,
                   render_mode: str = 'console',
                   step_by_step: bool = False,
                   delay_between_steps: float = 1.0) -> List[Dict[str, Any]]:
        """
        Démontre l'agent entraîné sur l'environnement.
        
        Args:
            num_episodes (int): Nombre d'épisodes à démontrer
            render_mode (str): Mode d'affichage ('console' ou 'pygame')
            step_by_step (bool): Pause entre chaque étape
            delay_between_steps (float): Délai entre les étapes (en secondes)
            
        Returns:
            List[Dict[str, Any]]: Historique détaillé des épisodes
        """
        if not self.algorithm.is_trained:
            raise ValueError("L'agent doit être entraîné avant la démonstration")
        
        print(f"\n{'='*50}")
        print(f"DÉMONSTRATION DE L'AGENT: {self.agent_name}")
        print(f"{'='*50}")
        
        episodes_history = []
        
        for episode_num in range(num_episodes):
            print(f"\n--- Épisode {episode_num + 1}/{num_episodes} ---")
            
            # Réinitialise l'environnement
            state = self.environment.reset()
            episode_history = []
            total_reward = 0.0
            step_count = 0
            
            # Affichage initial
            print(f"État initial: {state}")
            self.environment.render(mode=render_mode)
            
            if step_by_step:
                input("Appuyez sur Entrée pour continuer...")
            else:
                time.sleep(delay_between_steps)
            
            # Boucle de l'épisode
            max_steps = getattr(self.environment, 'max_steps', 1000)
            for step in range(max_steps):
                # Sélection d'action (mode exploitation)
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
                    "info": info
                }
                episode_history.append(step_info)
                
                total_reward += reward
                step_count += 1
                
                # Affichage de l'étape
                print(f"\nÉtape {step + 1}:")
                print(f"  Action: {action} -> Récompense: {reward}")
                print(f"  État: {state} -> {next_state}")
                if info:
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                
                self.environment.render(mode=render_mode)
                
                # Transition vers l'état suivant
                state = next_state
                
                if done:
                    print(f"\nÉpisode terminé à l'étape {step + 1}")
                    break
                
                # Pause entre les étapes
                if step_by_step:
                    input("Appuyez sur Entrée pour l'étape suivante...")
                else:
                    time.sleep(delay_between_steps)
            
            # Résumé de l'épisode
            episode_summary = {
                "episode": episode_num + 1,
                "total_reward": total_reward,
                "steps": step_count,
                "history": episode_history
            }
            episodes_history.append(episode_summary)
            
            print(f"\nRésumé de l'épisode {episode_num + 1}:")
            print(f"  Récompense totale: {total_reward}")
            print(f"  Nombre d'étapes: {step_count}")
        
        return episodes_history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des performances de l'agent.
        
        Returns:
            Dict[str, Any]: Résumé des performances
        """
        summary = {
            "agent_info": {
                "name": self.agent_name,
                "algorithm": self.algorithm.algo_name,
                "environment": self.environment.env_name,
                "is_trained": self.algorithm.is_trained
            },
            "session_info": self.session_info
        }
        
        # Informations d'entraînement
        if self.training_history:
            latest_training = self.training_history[-1]
            summary["training"] = {
                "sessions_completed": len(self.training_history),
                "latest_avg_reward": latest_training.get("avg_reward"),
                "latest_final_reward": latest_training.get("final_episode_reward"),
                "total_episodes": latest_training.get("episodes_trained"),
                "last_training_time": latest_training.get("training_time")
            }
        
        # Informations d'évaluation
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-1]
            summary["evaluation"] = {
                "evaluations_completed": len(self.evaluation_history),
                "latest_avg_reward": latest_eval.get("avg_reward"),
                "latest_std_reward": latest_eval.get("std_reward"),
                "latest_avg_episode_length": latest_eval.get("avg_episode_length")
            }
        
        # Hyperparamètres actuels
        summary["hyperparameters"] = self.algorithm.get_hyperparameters()
        
        return summary
    
    def plot_training_progress(self, save_path: str = None, show: bool = True):
        """
        Affiche les courbes de progression de l'entraînement.
        
        Args:
            save_path (str): Chemin pour sauvegarder le graphique
            show (bool): Afficher le graphique
        """
        if not self.algorithm.training_stats:
            print("Aucune donnée d'entraînement disponible pour le graphique")
            return
        
        stats = self.algorithm.training_stats
        episodes = [s.episode for s in stats]
        rewards = [s.total_reward for s in stats]
        steps = [s.steps for s in stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Graphique des récompenses
        ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Récompense par épisode')
        
        # Moyenne mobile
        if len(rewards) > 10:
            window_size = min(100, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = episodes[window_size-1:]
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, 
                    label=f'Moyenne mobile ({window_size} épisodes)')
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense totale')
        ax1.set_title(f'Progression de l\'entraînement - {self.agent_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique des étapes par épisode
        ax2.plot(episodes, steps, 'g-', alpha=0.7, label='Étapes par épisode')
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Nombre d\'étapes')
        ax2.set_title('Longueur des épisodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_agent(self, filepath: str) -> bool:
        """
        Sauvegarde l'agent complet (algorithme + métadonnées).
        
        Args:
            filepath (str): Chemin de sauvegarde (sans extension)
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Sauvegarde de l'algorithme
            algo_saved = self.algorithm.save_model(f"{filepath}_algorithm")
            
            # Sauvegarde des métadonnées de l'agent
            agent_metadata = {
                "agent_name": self.agent_name,
                "session_info": self.session_info,
                "training_history": self.training_history,
                "evaluation_history": self.evaluation_history,
                "performance_summary": self.get_performance_summary()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(agent_metadata, f, indent=2)
            
            print(f"Agent sauvegardé avec succès: {filepath}")
            return algo_saved
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent: {e}")
            return False
    
    def load_agent(self, filepath: str) -> bool:
        """
        Charge un agent pré-entraîné.
        
        Args:
            filepath (str): Chemin du modèle (sans extension)
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            # Chargement de l'algorithme
            algo_loaded = self.algorithm.load_model(f"{filepath}_algorithm")
            
            # Chargement des métadonnées
            try:
                with open(f"{filepath}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                self.training_history = metadata.get("training_history", [])
                self.evaluation_history = metadata.get("evaluation_history", [])
                
            except FileNotFoundError:
                print("Métadonnées non trouvées, seul l'algorithme a été chargé")
            
            print(f"Agent chargé avec succès: {filepath}")
            return algo_loaded
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent: {e}")
            return False
    
    def compare_algorithms(self, other_agents: List['Agent'], 
                          num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare les performances de cet agent avec d'autres agents.
        
        Args:
            other_agents (List[Agent]): Liste d'autres agents à comparer
            num_episodes (int): Nombre d'épisodes pour l'évaluation
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        all_agents = [self] + other_agents
        results = {}
        
        print(f"\n{'='*60}")
        print(f"COMPARAISON D'AGENTS SUR {self.environment.env_name}")
        print(f"{'='*60}")
        
        for agent in all_agents:
            if not agent.algorithm.is_trained:
                print(f"⚠️  Agent {agent.agent_name} non entraîné - ignoré")
                continue
            
            print(f"Évaluation de {agent.agent_name}...")
            eval_results = agent.evaluate(
                num_episodes=num_episodes,
                verbose=False
            )
            
            results[agent.agent_name] = {
                "algorithm": agent.algorithm.algo_name,
                "avg_reward": eval_results["avg_reward"],
                "std_reward": eval_results["std_reward"],
                "min_reward": eval_results["min_reward"],
                "max_reward": eval_results["max_reward"],
                "avg_episode_length": eval_results["avg_episode_length"],
                "hyperparameters": agent.algorithm.get_hyperparameters()
            }
        
        # Tri par récompense moyenne décroissante
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]["avg_reward"], 
                               reverse=True)
        
        # Affichage du classement
        print(f"\n{'='*60}")
        print("CLASSEMENT DES AGENTS")
        print(f"{'='*60}")
        for rank, (agent_name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {agent_name}")
            print(f"   Algorithme: {result['algorithm']}")
            print(f"   Récompense moyenne: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}")
            print(f"   Min/Max: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
            print(f"   Longueur d'épisode: {result['avg_episode_length']:.1f}")
            print()
        
        comparison_summary = {
            "environment": self.environment.env_name,
            "num_episodes": num_episodes,
            "agents_compared": len(results),
            "results": dict(sorted_results),
            "best_agent": sorted_results[0][0] if sorted_results else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison_summary
    
    def get_policy_visualization(self) -> str:
        """
        Retourne une visualisation textuelle de la politique apprise.
        
        Returns:
            str: Politique formatée
        """
        if not self.algorithm.is_trained:
            return "Agent non entraîné - politique non disponible"
        
        policy = self.algorithm.get_policy()
        value_function = self.algorithm.get_value_function()
        
        output = f"\n{'='*50}\n"
        output += f"POLITIQUE APPRISE - {self.agent_name}\n"
        output += f"{'='*50}\n"
        
        if hasattr(policy, 'shape'):  # NumPy array
            output += f"{'État':<8}{'Action':<8}{'Valeur':<12}{'Description':<20}\n"
            output += "-" * 50 + "\n"
            
            for state in range(len(policy)):
                action = policy[state]
                value = value_function[state] if value_function is not None else 0.0
                description = self.environment.get_state_description(state)
                output += f"{state:<8}{action:<8}{value:<12.3f}{description:<20}\n"
        else:  # Dictionnaire
            output += f"{'État':<8}{'Action':<8}{'Valeur':<12}{'Description':<20}\n"
            output += "-" * 50 + "\n"
            
            for state in sorted(policy.keys()):
                action = policy[state]
                value = value_function.get(state, 0.0) if value_function else 0.0
                description = self.environment.get_state_description(state)
                output += f"{state:<8}{action:<8}{value:<12.3f}{description:<20}\n"
        
        return output
    
    def __str__(self) -> str:
        """Représentation textuelle de l'agent."""
        status = "entraîné" if self.algorithm.is_trained else "non entraîné"
        return f"Agent({self.agent_name}, {status})"
    
    def __repr__(self) -> str:
        """Représentation détaillée de l'agent."""
        return (f"Agent(name='{self.agent_name}', "
                f"algorithm={self.algorithm.algo_name}, "
                f"environment={self.environment.env_name}, "
                f"trained={self.algorithm.is_trained})")


# Fonctions utilitaires pour créer des agents
def create_agent_from_configs(algorithm_class, algorithm_config: Dict[str, Any],
                             environment_class, environment_config: Dict[str, Any],
                             agent_name: str = None) -> Agent:
    """
    Crée un agent à partir de configurations.
    
    Args:
        algorithm_class: Classe de l'algorithme
        algorithm_config (Dict[str, Any]): Configuration de l'algorithme
        environment_class: Classe de l'environnement
        environment_config (Dict[str, Any]): Configuration de l'environnement
        agent_name (str): Nom de l'agent
        
    Returns:
        Agent: Agent configuré
    """
    # Création de l'environnement
    environment = environment_class(**environment_config)
    
    # Ajout des tailles d'espaces à la config de l'algorithme
    algorithm_config.update({
        'state_space_size': environment.state_space_size,
        'action_space_size': environment.action_space_size
    })
    
    # Création de l'algorithme
    algorithm = algorithm_class(**algorithm_config)
    
    # Création de l'agent
    return Agent(algorithm, environment, agent_name)


def quick_train_and_evaluate(agent: Agent, 
                            train_episodes: int = 1000,
                            eval_episodes: int = 100,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Fonction utilitaire pour entraîner et évaluer rapidement un agent.
    
    Args:
        agent (Agent): Agent à entraîner
        train_episodes (int): Nombre d'épisodes d'entraînement
        eval_episodes (int): Nombre d'épisodes d'évaluation
        verbose (bool): Affichage des détails
        
    Returns:
        Dict[str, Any]: Résultats d'entraînement et d'évaluation
    """
    # Entraînement
    train_results = agent.train(train_episodes, verbose=verbose)
    
    # Évaluation
    eval_results = agent.evaluate(eval_episodes, verbose=verbose)
    
    # Résumé combiné
    return {
        "agent_name": agent.agent_name,
        "training": train_results,
        "evaluation": eval_results,
        "summary": {
            "training_reward": train_results.get("final_episode_reward"),
            "evaluation_reward": eval_results.get("avg_reward"),
            "evaluation_std": eval_results.get("std_reward"),
            "improvement": "Success" if eval_results.get("avg_reward", 0) > 0 else "Needs work"
        }
    }


if __name__ == "__main__":
    # Test de la classe Agent
    print("Test de la classe Agent")
    
    # Métadonnées d'exemple
    example_session = {
        "created_at": datetime.now().isoformat(),
        "algorithm": "Q-Learning",
        "environment": "LineWorld",
        "state_space_size": 5,
        "action_space_size": 2
    }
    
    print("Structure de session d'exemple:")
    print(json.dumps(example_session, indent=2))
    
    print("\n✅ Classe Agent prête pour l'utilisation !")