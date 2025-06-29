from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class BaseAlgorithm(ABC):
    """Classe de base pour tous les algorithmes de renforcement learning."""
    
    def __init__(self, environment: Any, **kwargs):
        """
        Initialise l'algorithme.
        
        Args:
            environment: L'environnement sur lequel l'algorithme va s'entraîner
            **kwargs: Paramètres additionnels spécifiques à l'algorithme
        """
        self.environment = environment
        self.params = kwargs
        
    @abstractmethod
    def train(self, n_episodes: int) -> Dict[str, Any]:
        """
        Entraîne l'algorithme sur l'environnement.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dict contenant les métriques d'entraînement
        """
        pass
    
    @abstractmethod
    def get_action(self, state: Any) -> Any:
        """
        Retourne l'action à prendre dans un état donné.
        
        Args:
            state: L'état actuel
            
        Returns:
            L'action à prendre
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Charge le modèle.
        
        Args:
            path: Chemin du modèle à charger
        """
        pass 