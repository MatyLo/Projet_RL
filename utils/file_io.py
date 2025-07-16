"""
Utilitaires pour la gestion des fichiers (sauvegarde/chargement) dans le projet RL.

Ce module fournit des fonctions pour sauvegarder et charger les modèles,
résultats d'expériences, et autres données du projet.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import shutil


class RLFileManager:
    """Gestionnaire de fichiers pour le projet d'apprentissage par renforcement."""
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialise le gestionnaire de fichiers.
        
        Args:
            base_output_dir (str): Répertoire de base pour les sorties
        """
        self.base_dir = base_output_dir
        self.models_dir = os.path.join(base_output_dir, "models")
        self.results_dir = os.path.join(base_output_dir, "results")
        self.plots_dir = os.path.join(base_output_dir, "plots")
        
        # Sous-répertoires pour les modèles
        self.policies_dir = os.path.join(self.models_dir, "policies")
        self.value_functions_dir = os.path.join(self.models_dir, "value_functions")
        self.q_tables_dir = os.path.join(self.models_dir, "q_tables")
        
        # Sous-répertoires pour les résultats
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        self.comparisons_dir = os.path.join(self.results_dir, "comparisons")
        
        # Création des répertoires
        self._create_directories()
    
    def _create_directories(self):
        """Crée tous les répertoires nécessaires."""
        directories = [
            self.base_dir, self.models_dir, self.results_dir, self.plots_dir,
            self.policies_dir, self.value_functions_dir, self.q_tables_dir,
            self.logs_dir, self.metrics_dir, self.comparisons_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_filename(self, prefix: str, algorithm: str = None, 
                         environment: str = None, timestamp: bool = True,
                         extension: str = "json") -> str:
        """
        Génère un nom de fichier standardisé.
        
        Args:
            prefix (str): Préfixe du fichier
            algorithm (str): Nom de l'algorithme
            environment (str): Nom de l'environnement
            timestamp (bool): Inclure un timestamp
            extension (str): Extension du fichier
            
        Returns:
            str: Nom de fichier généré
        """
        components = [prefix]
        
        if algorithm:
            components.append(algorithm.replace(" ", "_").replace("-", "_"))
        
        if environment:
            components.append(environment.replace(" ", "_").replace("-", "_"))
        
        if timestamp:
            components.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        filename = "_".join(components)
        return f"{filename}.{extension}"
    
    def save_policy(self, policy: Union[np.ndarray, Dict[int, int]], 
                   algorithm_name: str, environment_name: str,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Sauvegarde une politique.
        
        Args:
            policy: Politique à sauvegarder
            algorithm_name (str): Nom de l'algorithme
            environment_name (str): Nom de l'environnement
            metadata (Dict): Métadonnées supplémentaires
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        filename = self.generate_filename(
            "policy", algorithm_name, environment_name
        )
        filepath = os.path.join(self.policies_dir, filename)
        
        # Préparation des données
        data = {
            "policy": policy.tolist() if isinstance(policy, np.ndarray) else policy,
            "algorithm": algorithm_name,
            "environment": environment_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Sauvegarde JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Sauvegarde pickle pour NumPy
        if isinstance(policy, np.ndarray):
            pickle_path = filepath.replace('.json', '.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(policy, f)
        
        return filepath
    
    def load_policy(self, filepath: str) -> Union[np.ndarray, Dict[int, int]]:
        """
        Charge une politique sauvegardée.
        
        Args:
            filepath (str): Chemin du fichier
            
        Returns:
            Union[np.ndarray, Dict[int, int]]: Politique chargée
        """
        try:
            # Essaie d'abord le fichier pickle
            pickle_path = filepath.replace('.json', '.pkl')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            
            # Sinon, charge le JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            policy = data['policy']
            if isinstance(policy, list):
                return np.array(policy)
            else:
                return policy
                
        except Exception as e:
            raise IOError(f"Erreur lors du chargement de la politique: {e}")
    
    def save_q_table(self, q_table: np.ndarray, algorithm_name: str, 
                    environment_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Sauvegarde une Q-table.
        
        Args:
            q_table (np.ndarray): Q-table à sauvegarder
            algorithm_name (str): Nom de l'algorithme
            environment_name (str): Nom de l'environnement
            metadata (Dict): Métadonnées supplémentaires
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        filename = self.generate_filename(
            "qtable", algorithm_name, environment_name
        )
        filepath = os.path.join(self.q_tables_dir, filename)
        
        # Sauvegarde NumPy
        np.save(filepath.replace('.json', '.npy'), q_table)
        
        # Sauvegarde métadonnées JSON
        data = {
            "shape": q_table.shape,
            "algorithm": algorithm_name,
            "environment": environment_name,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "min_value": float(np.min(q_table)),
                "max_value": float(np.max(q_table)),
                "mean_value": float(np.mean(q_table)),
                "std_value": float(np.std(q_table))
            },
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_q_table(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Charge une Q-table sauvegardée.
        
        Args:
            filepath (str): Chemin du fichier JSON
            
        Returns:
            Tuple[np.ndarray, Dict]: Q-table et métadonnées
        """
        try:
            # Charge les métadonnées
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            
            # Charge la Q-table
            npy_path = filepath.replace('.json', '.npy')
            q_table = np.load(npy_path)
            
            return q_table, metadata
            
        except Exception as e:
            raise IOError(f"Erreur lors du chargement de la Q-table: {e}")
    
    def save_value_function(self, value_function: Union[np.ndarray, Dict[int, float]],
                           algorithm_name: str, environment_name: str,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Sauvegarde une fonction de valeur.
        
        Args:
            value_function: Fonction de valeur à sauvegarder
            algorithm_name (str): Nom de l'algorithme
            environment_name (str): Nom de l'environnement
            metadata (Dict): Métadonnées supplémentaires
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        filename = self.generate_filename(
            "value_function", algorithm_name, environment_name
        )
        filepath = os.path.join(self.value_functions_dir, filename)
        
        # Préparation des données
        data = {
            "value_function": value_function.tolist() if isinstance(value_function, np.ndarray) else value_function,
            "algorithm": algorithm_name,
            "environment": environment_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Sauvegarde JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def save_training_results(self, results: Dict[str, Any], 
                            experiment_name: str = None) -> str:
        """
        Sauvegarde les résultats d'entraînement.
        
        Args:
            results (Dict): Résultats d'entraînement
            experiment_name (str): Nom de l'expérience
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        prefix = experiment_name or "training_results"
        filename = self.generate_filename(prefix)
        filepath = os.path.join(self.results_dir, filename)
        
        # Ajout de métadonnées
        results_with_meta = {
            "experiment_name": experiment_name,
            "saved_at": datetime.now().isoformat(),
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        
        return filepath
    
    def save_comparison_results(self, comparison_data: Dict[str, Any],
                              comparison_name: str) -> str:
        """
        Sauvegarde les résultats de comparaison d'algorithmes.
        
        Args:
            comparison_data (Dict): Données de comparaison
            comparison_name (str): Nom de la comparaison
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        filename = self.generate_filename(f"comparison_{comparison_name}")
        filepath = os.path.join(self.comparisons_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        return filepath
    
    def save_experiment_log(self, log_data: Dict[str, Any], 
                           experiment_name: str) -> str:
        """
        Sauvegarde un log d'expérience.
        
        Args:
            log_data (Dict): Données de log
            experiment_name (str): Nom de l'expérience
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        filename = self.generate_filename(f"log_{experiment_name}")
        filepath = os.path.join(self.logs_dir, filename)
        
        # Formatage du log
        log_entry = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "data": log_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return filepath
    
    def save_metrics_csv(self, metrics_data: List[Dict[str, Any]], 
                        filename: str) -> str:
        """
        Sauvegarde des métriques au format CSV.
        
        Args:
            metrics_data (List[Dict]): Données de métriques
            filename (str): Nom du fichier
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Conversion en DataFrame et sauvegarde
        df = pd.DataFrame(metrics_data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Charge des résultats sauvegardés.
        
        Args:
            filepath (str): Chemin du fichier
            
        Returns:
            Dict[str, Any]: Résultats chargés
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Erreur lors du chargement des résultats: {e}")
    
    def list_saved_models(self, algorithm_filter: str = None, 
                         environment_filter: str = None) -> List[Dict[str, str]]:
        """
        Liste les modèles sauvegardés avec filtres optionnels.
        
        Args:
            algorithm_filter (str): Filtre par algorithme
            environment_filter (str): Filtre par environnement
            
        Returns:
            List[Dict[str, str]]: Liste des modèles trouvés
        """
        models = []
        
        # Recherche dans tous les sous-répertoires de modèles
        model_dirs = [
            ("policies", self.policies_dir),
            ("q_tables", self.q_tables_dir),
            ("value_functions", self.value_functions_dir)
        ]
        
        for model_type, directory in model_dirs:
            if not os.path.exists(directory):
                continue
            
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            metadata = json.load(f)
                        
                        algorithm = metadata.get('algorithm', 'unknown')
                        environment = metadata.get('environment', 'unknown')
                        
                        # Application des filtres
                        if algorithm_filter and algorithm_filter.lower() not in algorithm.lower():
                            continue
                        if environment_filter and environment_filter.lower() not in environment.lower():
                            continue
                        
                        models.append({
                            "type": model_type,
                            "filename": filename,
                            "filepath": filepath,
                            "algorithm": algorithm,
                            "environment": environment,
                            "timestamp": metadata.get('timestamp', 'unknown')
                        })
                        
                    except Exception as e:
                        print(f"Erreur lors de la lecture de {filepath}: {e}")
        
        # Tri par timestamp (plus récent en premier)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def cleanup_old_files(self, days_old: int = 30, dry_run: bool = True) -> List[str]:
        """
        Nettoie les anciens fichiers.
        
        Args:
            days_old (int): Nombre de jours pour considérer un fichier comme ancien
            dry_run (bool): Mode test (n'efface pas réellement)
            
        Returns:
            List[str]: Liste des fichiers qui seraient/ont été supprimés
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        files_to_delete = []
        
        # Parcours récursif de tous les fichiers
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                # Vérification de la date de modification
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if mod_time < cutoff_date:
                    files_to_delete.append(filepath)
                    
                    if not dry_run:
                        try:
                            os.remove(filepath)
                            print(f"Supprimé: {filepath}")
                        except Exception as e:
                            print(f"Erreur lors de la suppression de {filepath}: {e}")
        
        if dry_run:
            print(f"Mode test: {len(files_to_delete)} fichiers seraient supprimés")
        else:
            print(f"{len(files_to_delete)} fichiers supprimés")
        
        return files_to_delete
    
    def backup_results(self, backup_dir: str) -> bool:
        """
        Crée une sauvegarde complète des résultats.
        
        Args:
            backup_dir (str): Répertoire de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"rl_backup_{timestamp}")
            
            # Copie récursive
            shutil.copytree(self.base_dir, backup_path)
            
            print(f"Sauvegarde créée: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'utilisation du stockage.
        
        Returns:
            Dict[str, Any]: Statistiques de stockage
        """
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "directories": {}
        }
        
        # Parcours de tous les répertoires
        for root, dirs, files in os.walk(self.base_dir):
            dir_stats = {
                "files": len(files),
                "size_mb": 0
            }
            
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    size = os.path.getsize(filepath)
                    dir_stats["size_mb"] += size / (1024 * 1024)  # Conversion en MB
                    stats["total_files"] += 1
                except OSError:
                    pass
            
            stats["total_size_mb"] += dir_stats["size_mb"]
            relative_path = os.path.relpath(root, self.base_dir)
            stats["directories"][relative_path] = dir_stats
        
        return stats


# Fonctions utilitaires globales
def create_file_manager(output_dir: str = "outputs") -> RLFileManager:
    """Crée un gestionnaire de fichiers pour le projet."""
    return RLFileManager(output_dir)


def quick_save_results(results: Dict[str, Any], name: str, 
                      output_dir: str = "outputs") -> str:
    """Sauvegarde rapide de résultats."""
    manager = create_file_manager(output_dir)
    return manager.save_training_results(results, name)


def quick_load_results(filepath: str) -> Dict[str, Any]:
    """Chargement rapide de résultats."""
    manager = create_file_manager()
    return manager.load_results(filepath)


def to_serializable(val):
    import numpy as np
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    return val


def save_model_with_description(model, environment_name: str, algorithm_name: str, hyperparameters: dict, metrics: dict, base_output_dir: str = "outputs"):
    """
    Sauvegarde un modèle RL et sa description dans un dossier structuré.
    - Crée outputs/<env>/<algo>_<timestamp>/
    - Sauvegarde le modèle (model.save_model)
    - Sauvegarde description.json (algo, env, date, hyperparams, metrics)
    """
    import os
    from datetime import datetime
    import json
    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{algorithm_name.lower().replace(' ', '_')}_{timestamp}"
    output_dir = os.path.join(base_output_dir, environment_name, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde du modèle
    model_path = os.path.join(output_dir, "model")
    model.save_model(model_path)

    # Description
    description = {
        "algorithm": algorithm_name,
        "environment": environment_name,
        "date": timestamp,
        "hyperparameters": hyperparameters,
        "metrics": metrics
    }
    desc_path = os.path.join(output_dir, "description.json")
    # Conversion pour JSON
    description = to_serializable(description)
    with open(desc_path, "w") as f:
        json.dump(description, f, indent=2)

    return output_dir


if __name__ == "__main__":
    # Test du gestionnaire de fichiers
    print("Test du gestionnaire de fichiers RL")
    
    # Création du gestionnaire
    manager = create_file_manager("test_outputs")
    print(f"Gestionnaire créé avec répertoire de base: {manager.base_dir}")
    
    # Test de sauvegarde de résultats
    test_results = {
        "algorithm": "Q-Learning",
        "environment": "LineWorld",
        "avg_reward": 8.5,
        "episodes": 1000
    }
    
    saved_path = manager.save_training_results(test_results, "test_experiment")
    print(f"Résultats sauvegardés: {saved_path}")
    
    # Test de chargement
    loaded_results = manager.load_results(saved_path)
    print(f"Résultats chargés: {loaded_results['results']}")
    
    # Statistiques de stockage
    stats = manager.get_storage_stats()
    print(f"Statistiques: {stats['total_files']} fichiers, {stats['total_size_mb']:.2f} MB")
    
    print("Test terminé !")