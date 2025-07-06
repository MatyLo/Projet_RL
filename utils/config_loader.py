"""
Gestionnaire de configurations JSON pour le projet d'apprentissage par renforcement.

Ce module fournit des utilitaires pour charger, valider et gérer les configurations
des environnements et algorithmes depuis des fichiers JSON.

Placement: utils/config_loader.py
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class ConfigError(Exception):
    """Exception levée pour les erreurs de configuration."""
    pass


class ConfigLoader:
    """
    Gestionnaire de configurations JSON pour le projet RL.
    
    Supporte le chargement de configurations hiérarchiques avec héritage
    et validation des paramètres.
    """
    
    def __init__(self, base_config_dir: str = "experiments/configs"):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            base_config_dir (str): Répertoire de base pour les configurations
        """
        self.base_config_dir = Path(base_config_dir)
        self.loaded_configs = {}  # Cache des configurations chargées
        
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
        
        # Traitement de l'héritage (si extends existe)
        if 'extends' in config:
            config = self._process_inheritance(config, full_path.parent)
        
        # Validation de base
        validated_config = self._validate_config(config)
        
        # Mise en cache
        if use_cache:
            self.loaded_configs[str(full_path)] = validated_config.copy()
        
        return validated_config
    
    def _process_inheritance(self, config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
        """
        Traite l'héritage de configuration (extends).
        
        Args:
            config (Dict): Configuration avec extends
            config_dir (Path): Répertoire de la configuration actuelle
            
        Returns:
            Dict[str, Any]: Configuration avec héritage résolu
        """
        extends_path = config.pop('extends')
        
        # Chargement de la configuration parent
        if not extends_path.endswith('.json'):
            extends_path += '.json'
            
        parent_path = config_dir / extends_path
        try:
            with open(parent_path, 'r', encoding='utf-8') as f:
                parent_config = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration parent non trouvée: {parent_path}")
        
        # Fusion des configurations (enfant override parent)
        merged_config = self._deep_merge(parent_config, config)
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusion profonde de deux dictionnaires.
        
        Args:
            base (Dict): Configuration de base
            override (Dict): Configuration qui override
            
        Returns:
            Dict[str, Any]: Configuration fusionnée
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide une configuration chargée.
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Dict[str, Any]: Configuration validée
            
        Raises:
            ConfigError: Si la configuration est invalide
        """
        required_sections = ['experiment', 'environment', 'algorithms']
        
        # Vérification des sections obligatoires
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Section obligatoire manquante: {section}")
        
        # Validation spécifique par section
        self._validate_experiment_config(config['experiment'])
        self._validate_environment_config(config['environment'])
        self._validate_algorithms_config(config['algorithms'])
        
        return config
    
    def _validate_experiment_config(self, experiment_config: Dict[str, Any]):
        """Valide la configuration d'expérience."""
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in experiment_config:
                raise ConfigError(f"Champ obligatoire manquant dans experiment: {field}")
    
    def _validate_environment_config(self, env_config: Dict[str, Any]):
        """Valide la configuration d'environnement."""
        if 'type' not in env_config:
            raise ConfigError("Le type d'environnement doit être spécifié")
            
        env_type = env_config['type']
        
        # Validation spécifique par type d'environnement
        if env_type == 'lineworld':
            self._validate_lineworld_config(env_config)
        elif env_type == 'gridworld':
            self._validate_gridworld_config(env_config)
        # Ajouter d'autres validations selon les environnements
    
    def _validate_lineworld_config(self, config: Dict[str, Any]):
        """Valide la configuration LineWorld."""
        defaults = {
            'line_length': 5,
            'start_position': 0,
            'target_position': 4,
            'reward_target': 10.0,
            'reward_step': -0.1,
            'reward_boundary': -1.0,
            'max_steps': 100
        }
        
        # Ajout des valeurs par défaut si manquantes
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
    
    def _validate_gridworld_config(self, config: Dict[str, Any]):
        """Valide la configuration GridWorld."""
        defaults = {
            'grid_size': [5, 5],
            'start_position': [0, 0],
            'target_position': [4, 4],
            'obstacles': [],
            'reward_target': 10.0,
            'reward_step': -0.1,
            'reward_obstacle': -1.0,
            'max_steps': 100
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
    
    def _validate_algorithms_config(self, algorithms_config: Dict[str, Any]):
        """Valide la configuration des algorithmes."""
        for algo_name, algo_config in algorithms_config.items():
            if not isinstance(algo_config, dict):
                raise ConfigError(f"Configuration invalide pour l'algorithme: {algo_name}")
            
            # Validation spécifique par algorithme
            if algo_name == 'q_learning':
                self._validate_qlearning_config(algo_config)
            elif algo_name == 'sarsa':
                self._validate_sarsa_config(algo_config)
    
    def _validate_qlearning_config(self, config: Dict[str, Any]):
        """Valide la configuration Q-Learning."""
        defaults = {
            'learning_rate': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'num_episodes': 1000,
            'initial_q_value': 0.0
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Validation des valeurs
        if not 0 <= config['learning_rate'] <= 1:
            raise ConfigError("learning_rate doit être entre 0 et 1")
        if not 0 <= config['gamma'] <= 1:
            raise ConfigError("gamma doit être entre 0 et 1")
        if not 0 <= config['epsilon'] <= 1:
            raise ConfigError("epsilon doit être entre 0 et 1")
    
    def _validate_sarsa_config(self, config: Dict[str, Any]):
        """Valide la configuration SARSA."""
        defaults = {
            'learning_rate': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'num_episodes': 1000
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
    
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
    
    def get_algorithm_config(self, config: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
        """
        Extrait la configuration d'un algorithme spécifique.
        
        Args:
            config (Dict): Configuration complète
            algorithm_name (str): Nom de l'algorithme
            
        Returns:
            Dict[str, Any]: Configuration de l'algorithme
            
        Raises:
            ConfigError: Si l'algorithme n'est pas trouvé
        """
        if 'algorithms' not in config:
            raise ConfigError("Section 'algorithms' manquante dans la configuration")
        
        if algorithm_name not in config['algorithms']:
            available = list(config['algorithms'].keys())
            raise ConfigError(f"Algorithme '{algorithm_name}' non trouvé. Disponibles: {available}")
        
        return config['algorithms'][algorithm_name].copy()
    
    def get_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrait la configuration d'environnement.
        
        Args:
            config (Dict): Configuration complète
            
        Returns:
            Dict[str, Any]: Configuration de l'environnement
        """
        if 'environment' not in config:
            raise ConfigError("Section 'environment' manquante dans la configuration")
        
        return config['environment'].copy()
    
    def clear_cache(self):
        """Vide le cache des configurations."""
        self.loaded_configs.clear()


# Fonctions utilitaires globales
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


def create_default_config(env_type: str, algorithm_name: str) -> Dict[str, Any]:
    """
    Crée une configuration par défaut pour un environnement et algorithme.
    
    Args:
        env_type (str): Type d'environnement ('lineworld', 'gridworld', etc.)
        algorithm_name (str): Nom de l'algorithme ('q_learning', 'sarsa', etc.)
        
    Returns:
        Dict[str, Any]: Configuration par défaut
    """
    config = {
        "experiment": {
            "name": f"default_{env_type}_{algorithm_name}",
            "description": f"Configuration par défaut pour {env_type} avec {algorithm_name}",
            "tags": ["default", env_type, algorithm_name]
        },
        "environment": {
            "type": env_type
        },
        "algorithms": {
            algorithm_name: {}
        },
        "training": {
            "save_frequency": 100,
            "evaluation_frequency": 50,
            "verbose": True
        }
    }
    
    # Validation pour ajouter les defaults
    loader = ConfigLoader()
    validated_config = loader._validate_config(config)
    
    return validated_config


if __name__ == "__main__":
    # Test du système de configuration
    print("Test du système de configuration")
    
    # Création d'une configuration par défaut
    config = create_default_config("lineworld", "q_learning")
    print("Configuration par défaut créée:")
    print(json.dumps(config, indent=2))
    
    # Test du chargement
    loader = ConfigLoader()
    print(f"\nFichiers de configuration disponibles: {loader.list_configs()}")
    
    print("\n✅ Système de configuration prêt !")