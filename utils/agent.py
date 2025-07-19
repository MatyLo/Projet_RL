"""
Agent - Wrapper post-entraînement pour algorithmes RL.

Cette classe encapsule un algorithme entraîné et fournit des fonctionnalités
d'évaluation, de démonstration et de comparaison.

Workflow simple:
1. Algorithme s'entraîne de façon autonome
2. Agent wrapper pour évaluation/démonstration  
3. Mode humain et comparaisons
"""

import numpy as np
import time
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime


class Agent:
    """
    Agent wrapper pour la phase post-entraînement.
    
    Utilisation:
    >>> algorithm.train(env, episodes=1000)  # Entraînement autonome
    >>> agent = Agent(algorithm, env, "MonAgent")  # Wrapper
    >>> agent.evaluate_performance()  # Évaluation
    >>> agent.demonstrate_step_by_step()  # Démo soutenance
    """
    
    def __init__(self, 
                 algorithm,
                 environment,
                 agent_name: str = None):
        """
        Initialise l'agent wrapper.
        
        Args:
            algorithm: Algorithme d'apprentissage entraîné
            environment: Environnement de test
            agent_name: Nom de l'agent pour identification
        """
        if not algorithm.is_trained:
            raise ValueError("L'algorithme doit être entraîné avant de créer l'Agent")
        
        self.algorithm = algorithm
        self.environment = environment
        self.agent_name = agent_name or f"{algorithm.algo_name}_Agent"
        
        # Vérification de compatibilité
        self._validate_compatibility()
        
        # Historique des évaluations et démos
        self.evaluation_history = []
        self.demonstration_history = []
        
        print(f"✅ Agent créé: {self.agent_name}")
    
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
    
    def evaluate_performance(self, 
                           num_episodes: int = 100,
                           verbose: bool = True,
                           success_criterion: str = "auto") -> Dict[str, Any]:
        """
        Évalue les performances de l'algorithme entraîné.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            verbose: Affichage des informations détaillées
            success_criterion: Critère de succès ("auto", "target_reached", "positive_reward", "custom")
            
        Returns:
            Dict avec statistiques d'évaluation
        """
        if verbose:
            print(f"\n📊 ÉVALUATION: {self.agent_name}")
            print(f"Épisodes d'évaluation: {num_episodes}")
        
        start_time = time.time()
        
        # Métriques de performance
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            steps = 0
            max_steps = getattr(self.environment, 'max_steps', 1000)
            episode_successful = False
            
            for step in range(max_steps):
                # Action sans exploration (politique gloutonne)
                action = self.algorithm.select_action(state, training=False)
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    # Détermination du succès selon le critère choisi
                    episode_successful = self._is_episode_successful(
                        episode_reward, info, success_criterion
                    )
                    break
            
            if episode_successful:
                success_count += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            if verbose and (episode + 1) % (num_episodes // 10) == 0:
                progress = (episode + 1) / num_episodes * 100
                print(f"Progression: {progress:.0f}% - Récompense moyenne: {np.mean(episode_rewards):.2f}")
        
        evaluation_time = time.time() - start_time
        success_rate = success_count / num_episodes
        
        # Résultats
        results = {
            "agent_name": self.agent_name,
            "environment": self.environment.__class__.__name__,
            "num_episodes": num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "success_rate": success_rate,
            "success_criterion": success_criterion,
            "evaluation_time": evaluation_time,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_history.append(results)
        
        if verbose:
            print(f"\n✅ RÉSULTATS:")
            print(f"Environnement: {results['environment']}")
            print(f"Récompense moyenne: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Taux de succès: {success_rate:.1%} (critère: {success_criterion})")
            print(f"Longueur moyenne: {results['avg_episode_length']:.1f} étapes")
            print(f"Temps d'évaluation: {evaluation_time:.2f}s\n")
        
        return results

    def _is_episode_successful(self, episode_reward: float, info: Dict, criterion: str) -> bool:
        """
        Détermine si un épisode est considéré comme réussi selon le critère choisi.
        
        Args:
            episode_reward: Récompense totale de l'épisode
            info: Dictionnaire d'informations de l'environnement
            criterion: Critère de succès à utiliser
            
        Returns:
            True si l'épisode est réussi, False sinon
        """
        if criterion == "target_reached":
            # Pour GridWorld, LineWorld avec objectif explicite
            return info.get("target_reached", False)
        
        elif criterion == "positive_reward":
            # Pour RPS et autres jeux compétitifs
            return episode_reward > 0
        
        elif criterion == "auto":
            # Détection automatique selon l'environnement
            env_name = self.environment.__class__.__name__.lower()
            
            if "rps" in env_name or "rockpaperscissors" in env_name:
                return episode_reward > 0
            elif "grid" in env_name or "line" in env_name:
                return info.get("target_reached", False)
            else:
                # Par défaut, utiliser la récompense positive
                return episode_reward > 0
        
        elif criterion == "custom":
            # Permet d'override cette méthode dans des sous-classes
            return self._custom_success_criterion(episode_reward, info)
        
        else:
            raise ValueError(f"Critère de succès non reconnu: {criterion}")

    def _custom_success_criterion(self, episode_reward: float, info: Dict) -> bool:
        """
        Critère de succès personnalisé à override dans les sous-classes.
        Par défaut, utilise la récompense positive.
        """
        return episode_reward > 0
    
    def demonstrate_step_by_step(self, 
                                num_episodes: int = 1,
                                show_q_values: bool = True,
                                pause_between_steps: bool = True) -> List[Dict[str, Any]]:
        """
        Démonstration pas-à-pas pour la soutenance.
        
        Args:
            num_episodes: Nombre d'épisodes à démontrer
            show_q_values: Afficher les Q-values si disponible
            pause_between_steps: Pause entre chaque étape
            
        Returns:
            Liste des démonstrations
        """
        print(f"\n🎬 DÉMONSTRATION PAS-À-PAS: {self.agent_name}")
        print("=" * 60)
        
        demonstrations = []
        
        for episode_num in range(num_episodes):
            print(f"\n🎯 Épisode {episode_num + 1}/{num_episodes}")
            
            # Initialisation
            state = self.environment.reset()
            demo_data = {
                "episode": episode_num + 1,
                "steps": [],
                "total_reward": 0.0,
                "success": False
            }
            
            print(f"État initial: {state}")
            self.environment.render('console')
            
            if show_q_values and hasattr(self.algorithm, 'q_function'):
                self._show_q_values(state)
            
            if pause_between_steps:
                input("\n▶️ Appuyez sur Entrée pour commencer...")
            
            # Déroulement de l'épisode
            max_steps = getattr(self.environment, 'max_steps', 1000)
            for step in range(max_steps):
                # Action optimale
                action = self.algorithm.select_action(state, training=False)
                next_state, reward, done, info = self.environment.step(action)
                
                # Enregistrement
                step_info = {
                    "step": step + 1,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                }
                demo_data["steps"].append(step_info)
                demo_data["total_reward"] += reward
                
                # Affichage
                print(f"\n⏯️ Étape {step + 1}:")
                print(f"Action choisie: {action}")
                print(f"État: {state} → {next_state}")
                print(f"Récompense: {reward} | Total: {demo_data['total_reward']:.2f}")
                
                self.environment.render('console')
                
                if show_q_values and hasattr(self.algorithm, 'q_function'):
                    self._show_q_values(next_state)
                
                state = next_state
                
                if done:
                    if info.get("target_reached", False):
                        print("🎉 SUCCÈS - Cible atteinte!")
                        demo_data["success"] = True
                    else:
                        print("⏰ Épisode terminé")
                    break
                
                if pause_between_steps:
                    input("\n▶️ Appuyez sur Entrée pour l'étape suivante...")
                else:
                    time.sleep(1)
            
            demonstrations.append(demo_data)
            
            print(f"\n📋 Résumé épisode {episode_num + 1}:")
            print(f"Récompense totale: {demo_data['total_reward']:.2f}")
            print(f"Nombre d'étapes: {len(demo_data['steps'])}")
            print(f"Succès: {'✅' if demo_data['success'] else '❌'}")
        
        self.demonstration_history.append({
            "timestamp": datetime.now().isoformat(),
            "demonstrations": demonstrations
        })
        
        return demonstrations
    
    def _show_q_values(self, state: int):
        """Affiche les Q-values pour un état donné."""
        if hasattr(self.algorithm, 'q_function'):
            try:
                q_values = self.algorithm.q_function[state]
                print(f"Q-values état {state}: {q_values}")
                best_action = np.argmax(q_values)
                print(f"Meilleure action: {best_action} (Q={q_values[best_action]:.3f})")
            except Exception:
                print("Q-values non disponibles pour cet état")
    
    def compare_with_other_agents(self, other_agents: List['Agent'], 
                                num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare les performances avec d'autres agents.
        
        Args:
            other_agents: Autres agents à comparer
            num_episodes: Nombre d'épisodes pour la comparaison
            
        Returns:
            Résultats de la comparaison
        """
        all_agents = [self] + other_agents
        comparison_results = {}
        
        print(f"\n🏆 COMPARAISON D'AGENTS")
        print(f"Environnement: {self.environment.env_name}")
        print(f"Épisodes de test: {num_episodes}")
        print("=" * 50)
        
        for agent in all_agents:
            print(f"🔍 Évaluation de {agent.agent_name}...")
            eval_results = agent.evaluate_performance(
                num_episodes=num_episodes,
                verbose=False
            )
            
            comparison_results[agent.agent_name] = {
                "algorithm": agent.algorithm.algo_name,
                "avg_reward": eval_results["avg_reward"],
                "success_rate": eval_results["success_rate"],
                "avg_episode_length": eval_results["avg_episode_length"]
            }
        
        # Classement par performance
        if comparison_results:
            sorted_results = sorted(comparison_results.items(), 
                                   key=lambda x: x[1]["avg_reward"], 
                                   reverse=True)
            
            print(f"\n CLASSEMENT:")
            print(f"{'Rang':<5}{'Agent':<20}{'Algorithme':<15}{'Récompense':<12}{'Succès':<10}")
            print("-" * 62)
            
            for rank, (agent_name, result) in enumerate(sorted_results, 1):
                print(f"{rank:<5}{agent_name[:19]:<20}{result['algorithm']:<15}"
                      f"{result['avg_reward']:<12.2f}{result['success_rate']:<10.1%}")
        
        return {
            "environment": self.environment.env_name,
            "num_episodes": num_episodes,
            "results": comparison_results,
            "best_agent": sorted_results[0][0] if comparison_results else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances de l'agent."""
        summary = {
            "agent_name": self.agent_name,
            "algorithm": self.algorithm.algo_name,
            "environment": self.environment.env_name,
            "is_trained": self.algorithm.is_trained,
            "num_evaluations": len(self.evaluation_history),
            "num_demonstrations": len(self.demonstration_history)
        }
        
        # Dernière évaluation
        if self.evaluation_history:
            latest = self.evaluation_history[-1]
            summary["latest_performance"] = {
                "avg_reward": latest["avg_reward"],
                "success_rate": latest["success_rate"],
                "timestamp": latest["timestamp"]
            }
        
        return summary
    
    def save_results(self, filepath: str) -> bool:
        """
        Sauvegarde tous les résultats de l'agent.
        
        Args:
            filepath: Chemin de sauvegarde (sans extension)
            
        Returns:
            True si succès
        """
        try:
            # Simple sauvegarde pickle - pas de problème NumPy
            results_data = {
                "agent_info": {
                    "name": self.agent_name,
                    "algorithm": self.algorithm.algo_name,
                    "environment": self.environment.env_name
                },
                "evaluation_history": self.evaluation_history,
                "demonstration_history": self.demonstration_history,
                "performance_summary": self.get_performance_summary()
            }
            
            with open(f"{filepath}_agent_results.pkl", 'wb') as f:
                pickle.dump(results_data, f)
            
            print(f"✅ Résultats sauvegardés: {filepath}_agent_results.pkl")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def __str__(self):
        """Représentation textuelle de l'agent."""
        return f"Agent({self.agent_name}, {self.algorithm.algo_name})"


if __name__ == "__main__":
    print("🤖 Classe Agent prête pour utilisation avec algorithmes entraînés!")
    print("Workflow: Algorithm.train() → Agent(algorithm, env) → évaluation/démo")