from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict
import numpy as np

class BaseEnvironment(ABC):
    """Classe de base pour tous les environnements de renforcement learning."""
    
    def __init__(self, **kwargs):
        """
        Initialise l'environnement.
        
        Args:
            **kwargs: Paramètres spécifiques à l'environnement
        """
        self.params = kwargs
        self.reset()
    
    @abstractmethod
    def reset(self) -> Any:
        """
        Réinitialise l'environnement à son état initial.
        
        Returns:
            L'état initial
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: L'action à exécuter
            
        Returns:
            Tuple contenant:
            - next_state: Le prochain état
            - reward: La récompense obtenue
            - done: Si l'épisode est terminé
            - info: Informations additionnelles
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """
        Affiche l'état actuel de l'environnement.
        """
        pass
    
    @abstractmethod
    def get_state_space(self) -> Any:
        """
        Retourne l'espace des états.
        
        Returns:
            Description de l'espace des états
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> Any:
        """
        Retourne l'espace des actions.
        
        Returns:
            Description de l'espace des actions
        """
        pass 