"""
Mode Agent Humain - Interface pour qu'un humain puisse jouer sur les environnements.

Cette classe permet à un utilisateur humain de jouer manuellement sur les environnements
pour tester les règles et comprendre la dynamique du jeu.

Placement: utils/human_player.py
"""

import sys
from typing import Dict, List, Any, Optional
import time

from src.rl_environments.base_environment import BaseEnvironment


class HumanPlayer:
    """
    Agent humain pour jouer manuellement sur les environnements RL.
    
    Cette classe fournit une interface pour qu'un utilisateur puisse
    interagir manuellement avec les environnements d'apprentissage par renforcement.
    """
    
    def __init__(self, environment: BaseEnvironment, player_name: str = "Human"):
        """
        Initialise le joueur humain.
        
        Args:
            environment (BaseEnvironment): Environnement de jeu
            player_name (str): Nom du joueur humain
        """
        self.environment = environment
        self.player_name = player_name
        self.game_history = []
        self.total_games = 0
        self.total_wins = 0
        
        print(f"🎮 Joueur humain initialisé: {player_name}")
        print(f"Environnement: {environment.env_name}")
    
    def play_episode(self, 
                    show_instructions: bool = True,
                    show_rewards: bool = True,
                    show_q_values: bool = False,
                    trained_algorithm=None) -> Dict[str, Any]:
        """
        Lance un épisode de jeu humain.
        
        Args:
            show_instructions (bool): Afficher les instructions
            show_rewards (bool): Afficher les récompenses
            show_q_values (bool): Afficher les Q-values d'un algorithme entraîné
            trained_algorithm: Algorithme entraîné pour comparaison (optionnel)
            
        Returns:
            Dict[str, Any]: Résultats de l'épisode
        """
        print(f"\n🎯 NOUVEAU JEU - {self.player_name}")
        print(f"Environnement: {self.environment.env_name}")
        print("=" * 50)
        
        if show_instructions:
            self._show_instructions()
        
        # Initialisation de l'épisode
        state = self.environment.reset()
        episode_data = {
            "player": self.player_name,
            "environment": self.environment.env_name,
            "steps": [],
            "total_reward": 0.0,
            "success": False,
            "num_steps": 0
        }
        
        print(f"\n🏁 ÉTAT INITIAL:")
        print(f"Position: {state} - {self.environment.get_state_description(state)}")
        self.environment.render(mode='console')
        
        if show_q_values and trained_algorithm and hasattr(trained_algorithm, 'q_function'):
            self._show_algorithm_suggestion(state, trained_algorithm)
        
        # Boucle de jeu
        max_steps = getattr(self.environment, 'max_steps', 1000)
        for step in range(max_steps):
            # Obtenir l'action du joueur humain
            try:
                action = self._get_human_action(state)
            except KeyboardInterrupt:
                print("\n❌ Jeu interrompu par le joueur")
                break
            except EOFError:
                print("\n❌ Fin d'entrée détectée")
                break
            
            if action is None:
                print("❌ Action invalide ou abandon du jeu")
                break
            
            # Exécuter l'action
            prev_state = state
            next_state, reward, done, info = self.environment.step(action)
            
            # Enregistrer l'étape
            step_info = {
                "step": step + 1,
                "state": prev_state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "info": info
            }
            episode_data["steps"].append(step_info)
            episode_data["total_reward"] += reward
            episode_data["num_steps"] += 1
            
            # Affichage du résultat
            print(f"\n⏯️ ÉTAPE {step + 1}:")
            print(f"Action choisie: {action}")
            print(f"Nouvel état: {next_state} - {self.environment.get_state_description(next_state)}")
            
            if show_rewards:
                print(f"Récompense: {reward:.2f} | Total: {episode_data['total_reward']:.2f}")
            
            if info:
                for key, value in info.items():
                    if key in ['target_reached', 'boundary_hit', 'max_steps_reached']:
                        status = "✅" if value else "❌"
                        print(f"{key}: {status}")
            
            self.environment.render(mode='console')
            
            if show_q_values and trained_algorithm and hasattr(trained_algorithm, 'q_function'):
                self._show_algorithm_suggestion(next_state, trained_algorithm)
            
            state = next_state
            
            if done:
                if info.get("target_reached", False):
                    print("🎉 FÉLICITATIONS! Vous avez atteint la cible!")
                    episode_data["success"] = True
                    self.total_wins += 1
                elif info.get("max_steps_reached", False):
                    print("⏰ Temps écoulé! Nombre maximum d'étapes atteint.")
                else:
                    print("🏁 Épisode terminé.")
                break
        
        # Résumé de l'épisode
        self.total_games += 1
        self.game_history.append(episode_data)
        
        print(f"\n📊 RÉSUMÉ DE L'ÉPISODE:")
        print(f"Nombre d'étapes: {episode_data['num_steps']}")
        print(f"Récompense totale: {episode_data['total_reward']:.2f}")
        print(f"Succès: {'✅' if episode_data['success'] else '❌'}")
        print(f"Statistiques globales: {self.total_wins}/{self.total_games} victoires ({self.total_wins/self.total_games*100:.1f}%)")
        
        return episode_data
    
    def _get_human_action(self, state: int) -> Optional[int]:
        """
        Obtient l'action du joueur humain via l'interface console.
        
        Args:
            state (int): État actuel
            
        Returns:
            Optional[int]: Action choisie par le joueur (None si abandon)
        """
        valid_actions = self.environment.valid_actions
        
        print(f"\n🎮 À VOTRE TOUR!")
        print(f"État actuel: {state}")
        print(f"Actions disponibles: {valid_actions}")
        
        # Instructions spécifiques selon le type d'environnement
        if "lineworld" in self.environment.env_name.lower():
            print("0 = Gauche ← | 1 = Droite →")
        elif "gridworld" in self.environment.env_name.lower():
            print("0 = Haut ↑ | 1 = Droite → | 2 = Bas ↓ | 3 = Gauche ←")
        elif "monty" in self.environment.env_name.lower():
            print("Choisissez le numéro de la porte")
        
        while True:
            try:
                user_input = input("Votre action (ou 'q' pour quitter): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    return None
                
                action = int(user_input)
                
                if action in valid_actions:
                    return action
                else:
                    print(f"❌ Action invalide! Choisissez parmi: {valid_actions}")
                    
            except ValueError:
                print("❌ Veuillez entrer un nombre valide ou 'q' pour quitter")
            except (KeyboardInterrupt, EOFError):
                return None
    
    def _show_instructions(self):
        """Affiche les instructions de jeu selon l'environnement."""
        print(f"\n📖 INSTRUCTIONS DE JEU:")
        print(f"Environnement: {self.environment.env_name}")
        
        if "lineworld" in self.environment.env_name.lower():
            print("🎯 Objectif: Atteindre la position cible")
            print("🕹️ Actions: 0 (Gauche) ou 1 (Droite)")
            print("⚠️ Attention: Sortir des limites donne une pénalité")
            
        elif "gridworld" in self.environment.env_name.lower():
            print("🎯 Objectif: Atteindre la case cible")
            print("🕹️ Actions: 0 (Haut), 1 (Droite), 2 (Bas), 3 (Gauche)")
            print("⚠️ Attention: Évitez les obstacles")
            
        elif "monty" in self.environment.env_name.lower():
            print("🎯 Objectif: Choisir la porte gagnante")
            print("🕹️ Actions: Numéro de la porte à choisir")
            print("💡 Astuce: Changez ou gardez votre choix selon la stratégie")
            
        elif "rock" in self.environment.env_name.lower():
            print("🎯 Objectif: Gagner le maximum de rounds")
            print("🕹️ Actions: 0 (Pierre), 1 (Feuille), 2 (Ciseaux)")
            print("💡 Astuce: Analysez le comportement de l'adversaire")
        
        print("💾 Tapez 'q' à tout moment pour quitter")
        print("-" * 50)
    
    def _show_algorithm_suggestion(self, state: int, algorithm):
        """
        Affiche la suggestion d'un algorithme entraîné.
        
        Args:
            state (int): État actuel
            algorithm: Algorithme entraîné
        """
        try:
            if hasattr(algorithm, 'q_function'):
                q_values = algorithm.q_function[state]
                best_action = algorithm.select_action(state, training=False)
                
                print(f"\n🤖 SUGGESTION DE L'ALGORITHME {algorithm.algo_name}:")
                print(f"Q-values: {q_values}")
                print(f"Action recommandée: {best_action} (Q={q_values[best_action]:.3f})")
                
        except Exception as e:
            print(f"⚠️ Erreur lors de l'affichage des suggestions: {e}")
    
    def play_multiple_episodes(self, 
                             num_episodes: int = 5,
                             show_instructions: bool = True,
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Lance plusieurs épisodes consécutifs.
        
        Args:
            num_episodes (int): Nombre d'épisodes à jouer
            show_instructions (bool): Afficher les instructions au début
            **kwargs: Arguments pour play_episode
            
        Returns:
            List[Dict[str, Any]]: Résultats de tous les épisodes
        """
        print(f"\n🎮 SESSION DE JEU MULTIPLE")
        print(f"Joueur: {self.player_name}")
        print(f"Nombre d'épisodes: {num_episodes}")
        print("=" * 50)
        
        episodes_results = []
        
        for episode_num in range(num_episodes):
            print(f"\n🎯 ÉPISODE {episode_num + 1}/{num_episodes}")
            
            # Afficher les instructions seulement au premier épisode
            show_instr = show_instructions and episode_num == 0
            
            try:
                episode_result = self.play_episode(
                    show_instructions=show_instr,
                    **kwargs
                )
                episodes_results.append(episode_result)
                
                # Demander si continuer
                if episode_num < num_episodes - 1:
                    continue_input = input(f"\nContinuer vers l'épisode {episode_num + 2}? (y/n): ").strip().lower()
                    if continue_input in ['n', 'no', 'non']:
                        print("Session de jeu interrompue par le joueur")
                        break
                        
            except KeyboardInterrupt:
                print("\nSession de jeu interrompue")
                break
        
        # Affichage du résumé final
        self._show_session_summary(episodes_results)
        
        return episodes_results
    
    def _show_session_summary(self, episodes_results: List[Dict[str, Any]]):
        """
        Affiche un résumé de la session de jeu.
        
        Args:
            episodes_results (List): Résultats des épisodes
        """
        if not episodes_results:
            return
        
        print(f"\n📊 RÉSUMÉ DE LA SESSION")
        print("=" * 50)
        
        total_episodes = len(episodes_results)
        total_wins = sum(1 for episode in episodes_results if episode['success'])
        total_reward = sum(episode['total_reward'] for episode in episodes_results)
        avg_reward = total_reward / total_episodes
        avg_steps = sum(episode['num_steps'] for episode in episodes_results) / total_episodes
        
        print(f"Épisodes joués: {total_episodes}")
        print(f"Victoires: {total_wins}/{total_episodes} ({total_wins/total_episodes*100:.1f}%)")
        print(f"Récompense moyenne: {avg_reward:.2f}")
        print(f"Nombre d'étapes moyen: {avg_steps:.1f}")
        
        # Détail par épisode
        print(f"\n📋 DÉTAIL PAR ÉPISODE:")
        print(f"{'Épisode':<8}{'Succès':<8}{'Récompense':<12}{'Étapes':<8}")
        print("-" * 36)
        
        for i, episode in enumerate(episodes_results, 1):
            success_icon = "✅" if episode['success'] else "❌"
            print(f"{i:<8}{success_icon:<8}{episode['total_reward']:<12.2f}{episode['num_steps']:<8}")
    
    def compare_with_algorithm(self, 
                             algorithm,
                             num_episodes: int = 5,
                             show_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare les performances humaines avec un algorithme.
        
        Args:
            algorithm: Algorithme entraîné à comparer
            num_episodes (int): Nombre d'épisodes pour la comparaison
            show_comparison (bool): Afficher la comparaison détaillée
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        print(f"\n🏆 DÉFI HUMAIN VS ALGORITHME")
        print(f"Humain: {self.player_name}")
        print(f"Algorithme: {algorithm.algo_name}")
        print("=" * 50)
        
        # Jeu humain
        print(f"\n👤 TOUR DU JOUEUR HUMAIN")
        human_results = self.play_multiple_episodes(
            num_episodes=num_episodes,
            show_instructions=True,
            show_q_values=True,
            trained_algorithm=algorithm
        )
        
        # Évaluation de l'algorithme
        print(f"\n🤖 TOUR DE L'ALGORITHME")
        from utils.agent import Agent
        
        agent = Agent(algorithm, self.environment)
        algo_results = agent.evaluate_performance(
            num_episodes=num_episodes,
            verbose=True
        )
        
        # Comparaison
        if show_comparison:
            self._show_human_vs_algo_comparison(human_results, algo_results)
        
        # Résultats de la comparaison
        human_avg_reward = sum(ep['total_reward'] for ep in human_results) / len(human_results)
        human_win_rate = sum(1 for ep in human_results if ep['success']) / len(human_results)
        
        comparison = {
            "human_performance": {
                "avg_reward": human_avg_reward,
                "win_rate": human_win_rate,
                "episodes": len(human_results)
            },
            "algorithm_performance": {
                "avg_reward": algo_results["avg_reward"],
                "win_rate": algo_results["success_rate"],
                "episodes": algo_results["num_episodes"]
            },
            "winner": "Human" if human_avg_reward > algo_results["avg_reward"] else "Algorithm"
        }
        
        return comparison
    
    def _show_human_vs_algo_comparison(self, human_results: List[Dict], algo_results: Dict):
        """Affiche la comparaison détaillée humain vs algorithme."""
        print(f"\n🏆 RÉSULTATS DU DÉFI")
        print("=" * 50)
        
        human_avg_reward = sum(ep['total_reward'] for ep in human_results) / len(human_results)
        human_win_rate = sum(1 for ep in human_results if ep['success']) / len(human_results)
        
        print(f"{'Métrique':<20}{'Humain':<15}{'Algorithme':<15}{'Gagnant':<10}")
        print("-" * 60)
        
        # Récompense moyenne
        winner_reward = "Humain" if human_avg_reward > algo_results["avg_reward"] else "Algorithme"
        print(f"{'Récompense moy.':<20}{human_avg_reward:<15.2f}{algo_results['avg_reward']:<15.2f}{winner_reward:<10}")
        
        # Taux de succès
        winner_success = "Humain" if human_win_rate > algo_results["success_rate"] else "Algorithme"
        print(f"{'Taux de succès':<20}{human_win_rate:<15.2%}{algo_results['success_rate']:<15.2%}{winner_success:<10}")
        
        # Gagnant global
        overall_winner = "HUMAIN" if human_avg_reward > algo_results["avg_reward"] else "ALGORITHME"
        print(f"\n🏆 GAGNANT GLOBAL: {overall_winner}")
    
    def get_player_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes du joueur.
        
        Returns:
            Dict[str, Any]: Statistiques du joueur
        """
        if not self.game_history:
            return {"message": "Aucune partie jouée"}
        
        rewards = [game['total_reward'] for game in self.game_history]
        steps = [game['num_steps'] for game in self.game_history]
        
        return {
            "player_name": self.player_name,
            "environment": self.environment.env_name,
            "total_games": self.total_games,
            "total_wins": self.total_wins,
            "win_rate": self.total_wins / self.total_games if self.total_games > 0 else 0,
            "avg_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "avg_steps": sum(steps) / len(steps),
            "game_history": self.game_history
        }
    
    def save_player_data(self, filepath: str) -> bool:
        """
        Sauvegarde les données du joueur.
        
        Args:
            filepath (str): Chemin de sauvegarde
            
        Returns:
            bool: True si succès
        """
        try:
            import json
            
            player_data = self.get_player_statistics()
            
            with open(f"{filepath}_human_player.json", 'w') as f:
                json.dump(player_data, f, indent=2)
            
            print(f"✅ Données du joueur sauvegardées: {filepath}_human_player.json")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def __str__(self) -> str:
        """Représentation textuelle du joueur."""
        return f"HumanPlayer({self.player_name}, {self.total_wins}/{self.total_games} victoires)"


# Fonction utilitaire pour lancer rapidement le mode humain
def quick_human_game(environment, player_name: str = "Player1", num_episodes: int = 1):
    """
    Lance rapidement un jeu humain sur un environnement.
    
    Args:
        environment: Environnement de jeu
        player_name (str): Nom du joueur
        num_episodes (int): Nombre d'épisodes
    """
    human = HumanPlayer(environment, player_name)
    
    if num_episodes == 1:
        return human.play_episode()
    else:
        return human.play_multiple_episodes(num_episodes)


if __name__ == "__main__":
    print("🎮 Mode Agent Humain - Prêt pour les tests!")
    print("Utilisez quick_human_game(environment) pour tester rapidement")