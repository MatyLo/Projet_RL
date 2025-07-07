"""
Gestionnaire de configurations JSON simplifié pour le projet RL.

Version allégée sans héritage complexe - plus facile pour les étudiants.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigError(Exception):
    """Exception levée pour les erreurs de configuration."""
    pass


class ConfigLoader:
    """
    Gestionnaire de configurations JSON simplifié.
    
    Supprime la complexité de l'héritage pour faciliter la compréhension.
    """
    
    def __init__(self, base_config_dir: str = "experiments/configs"):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            base_config_dir (str): Répertoire de base pour les configurations
        """
        self.base_config_dir = Path(base_config_dir)
        self.loaded_configs = {}  #
        
    def load(self, config_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Charge une configuration depuis un fichier JSON.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration
            use_cache (bool): Utiliser le cache si disponible
            
        Returns:
            Dict[str, Any]: Configuration chargée et validée
            
        Raises:
            ConfigError: Si le fichier n'existe pas ou est invalide
        """
        # Normalisation du chemin
        if not config_path.endswith('.json'):
            config_path += '.json'
            
        full_path = self.base_config_dir / config_path
        
        # Vérification cache
        if use_cache and str(full_path) in self.loaded_configs:
            return self.loaded_configs[str(full_path)].copy()
        
        # Chargement du fichier
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Fichier de configuration non trouvé: {full_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Erreur de syntaxe JSON dans {full_path}: {e}")
        
        # Validation simple
        validated_config = self._validate_config(config)
        
        # Mise en cache
        if use_cache:
            self.loaded_configs[str(full_path)] = validated_config.copy()
        
        return validated_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide une configuration chargée.
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Dict[str, Any]: Configuration validée avec defaults
            
        Raises:
            ConfigError: Si la configuration est invalide
        """
        # Vérification des sections minimales
        if 'experiment' not in config:
            raise ConfigError("Section 'experiment' obligatoire manquante")
        
        if 'environment' not in config:
            raise ConfigError("Section 'environment' obligatoire manquante")
        
        # Ajout des defaults simples
        self._add_defaults(config)
        
        return config
    
    def _add_defaults(self, config: Dict[str, Any]):
        """Ajoute les valeurs par défaut manquantes."""
        
        # Defaults pour experiment
        if 'name' not in config['experiment']:
            config['experiment']['name'] = "unnamed_experiment"
        
        # Defaults pour environment
        env_config = config['environment']
        if env_config.get('type') == 'lineworld':
            defaults = {
                'max_steps': 100,
                'line_length': 5,
                'start_position': 2,
                'target_position': 4
            }
            for key, value in defaults.items():
                if key not in env_config:
                    env_config[key] = value
        
        # Defaults pour algorithms (si présent)
        if 'algorithm' in config:
            algo_config = config['algorithm']
            if algo_config.get('type') == 'q_learning':
                defaults = {
                    'learning_rate': 0.1,
                    'gamma': 0.9,
                    'epsilon': 0.1,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.01
                }
                for key, value in defaults.items():
                    if key not in algo_config:
                        algo_config[key] = value
        
        # Defaults pour training (si présent)
        if 'training' not in config:
            config['training'] = {}
        
        training_defaults = {
            'num_episodes': 1000,
            'verbose': True,
            'save_model': True
        }
        for key, value in training_defaults.items():
            if key not in config['training']:
                config['training'][key] = value
        
        # Defaults pour evaluation (si présent)
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        eval_defaults = {
            'num_episodes': 100,
            'verbose': True
        }
        for key, value in eval_defaults.items():
            if key not in config['evaluation']:
                config['evaluation'][key] = value
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """
        Sauvegarde une configuration dans un fichier JSON.
        
        Args:
            config (Dict): Configuration à sauvegarder
            config_path (str): Chemin de sauvegarde
        """
        if not config_path.endswith('.json'):
            config_path += '.json'
            
        full_path = self.base_config_dir / config_path
        
        # Création du répertoire si nécessaire
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigError(f"Erreur lors de la sauvegarde: {e}")
    
    def list_configs(self, pattern: str = "*.json") -> List[str]:
        """
        Liste les fichiers de configuration disponibles.
        
        Args:
            pattern (str): Pattern de recherche
            
        Returns:
            List[str]: Liste des fichiers de configuration
        """
        if not self.base_config_dir.exists():
            return []
        
        config_files = []
        for file_path in self.base_config_dir.rglob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.base_config_dir)
                config_files.append(str(relative_path))
        
        return sorted(config_files)
    
    def clear_cache(self):
        """Vide la liste des configurations enregistré dans init."""
        self.loaded_configs.clear()


def load_config(config_path: str, base_dir: str = "experiments/configs") -> Dict[str, Any]:
    """
    Fonction utilitaire pour charger rapidement une configuration.
    
    Args:
        config_path (str): Chemin vers la configuration
        base_dir (str): Répertoire de base
        
    Returns:
        Dict[str, Any]: Configuration chargée
    """
    loader = ConfigLoader(base_dir)
    return loader.load(config_path)


def create_simple_config(env_type: str, algorithm_type: str) -> Dict[str, Any]:
    """
    Crée une configuration simple pour un environnement et algorithme.
    
    Args:
        env_type (str): Type d'environnement ('lineworld', 'gridworld', etc.)
        algorithm_type (str): Type d'algorithme ('q_learning', 'sarsa', etc.)
        
    Returns:
        Dict[str, Any]: Configuration simple
    """
    config = {
        "experiment": {
            "name": f"simple_{env_type}_{algorithm_type}",
            "description": f"Configuration simple pour {env_type} avec {algorithm_type}"
        },
        "environment": {
            "type": env_type
        },
        "algorithm": {
            "type": algorithm_type
        },
        "training": {
            "num_episodes": 1000,
            "verbose": True
        },
        "evaluation": {
            "num_episodes": 100,
            "verbose": True
        }
    }
    
    # Validation automatique pour ajouter les defaults
    loader = ConfigLoader()
    validated_config = loader._validate_config(config)
    
    return validated_config


if __name__ == "__main__":
    # Test du système simplifié
    print("Test du système de configuration simplifié")
    
    # Création d'une configuration simple
    config = create_simple_config("lineworld", "q_learning")
    print("Configuration simple créée:")
    print(json.dumps(config, indent=2))
    
    # Test du chargement
    loader = ConfigLoader()
    print(f"\nFichiers de configuration disponibles: {loader.list_configs()}")
    
    print("\n✅ Système de configuration simplifié prêt !")