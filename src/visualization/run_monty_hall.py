import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
from rl_environments.monty_hall_interactive import MontyHallInteractive
from rl_environments.monty_hall2_stepbystep import MontyHall2StepByStep
from rl_algorithms import QLearning, ValueIteration, SARSA
from visualization.monty_hall_visualizer import MontyHallVisualizer
import time

def get_algorithm_class(algorithm_name: str):
    """Convertit le nom de l'algorithme en classe correspondante."""
    algorithm_map = {
        'Q-Learning': QLearning,
        'SARSA': SARSA,
        'Value Iteration': ValueIteration
    }
    return algorithm_map.get(algorithm_name, ValueIteration)

def wait_for_space(vis):
    """Attend que l'utilisateur appuie sur ESPACE ou ferme la fenêtre."""
    waiting = True
    while waiting and vis.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vis.running = False
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
        vis.clock.tick(vis.fps)

def run_monty_hall1():
    """Lance Monty Hall 1 avec sélection interactive du mode et de l'algorithme."""
    episodes = 3  # Nombre d'épisodes fixé ici
    env = MontyHallInteractive()
    vis = MontyHallVisualizer()
    
    # Afficher le menu principal
    mode, algorithm = vis.render_main_menu()
    
    if mode is None or not vis.running:
        vis.quit()
        return
    
    # Si mode agent mais pas d'algorithme sélectionné, quitter
    if mode == "agent" and algorithm is None:
        vis.quit()
        return
    
    win_count = 0
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        reward = 0
        info = {}
        agent = None
        if mode == "agent":
            agent_class = get_algorithm_class(algorithm)
            if algorithm == "Q-Learning":
                agent = agent_class(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=1000)
            elif algorithm == "SARSA":
                agent = agent_class(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=1000)
            else:  # Value Iteration
                agent = agent_class(env.state_space_size, env.action_space_size, gamma=0.9)
                agent.train(env)
        
        while not done and vis.running:
            vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count, "Mode": mode, "Algorithme": algorithm if mode == "agent" else "Humain"})
            if mode == "human":
                action = vis.get_human_action_mh2(env)
                if action is None:
                    vis.quit()
                    return
            else:
                # Vérifier s'il y a des actions valides
                if len(env.valid_actions) == 0:
                    break
                # Sélectionner une action valide
                valid_actions = env.valid_actions
                if len(valid_actions) > 0:
                    action = agent.select_action(state)
                    # S'assurer que l'action est valide
                    if action not in valid_actions:
                        action = valid_actions[0]  # Fallback
                else:
                    break
                # Afficher message et attendre ESPACE
                vis.draw_text("Appuyez sur ESPACE pour voir l'action suivante de l'IA", 120, 500, vis.RED, vis.font)
                vis.update_display()
                wait_for_space(vis)
            state, reward, done, info = env.step(action)
        
        if reward > 0:
            win_count += 1
        
        vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count, "Mode": mode, "Algorithme": algorithm if mode == "agent" else "Humain"})
        pygame.time.wait(1200)
        if not vis.running:
            break
    
    vis.quit()

def run_monty_hall2():
    """Lance Monty Hall 2 avec sélection interactive du mode et de l'algorithme."""
    episodes = 3  # Nombre d'épisodes fixé ici
    env = MontyHall2StepByStep()
    vis = MontyHallVisualizer()
    
    # Afficher le menu principal
    mode, algorithm = vis.render_main_menu()
    
    if mode is None or not vis.running:
        vis.quit()
        return
    
    # Si mode agent mais pas d'algorithme sélectionné, quitter
    if mode == "agent" and algorithm is None:
        vis.quit()
        return
    
    win_count = 0
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        reward = 0
        info = {}
        agent = None
        if mode == "agent":
            agent_class = get_algorithm_class(algorithm)
            if algorithm == "Q-Learning":
                agent = agent_class(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=2000)
            elif algorithm == "SARSA":
                agent = agent_class(env.state_space_size, env.action_space_size, learning_rate=0.1, gamma=0.9, epsilon=0.1)
                agent.train(env, num_episodes=2000)
            else:  # Value Iteration
                agent = agent_class(env.state_space_size, env.action_space_size, gamma=0.9)
                agent.train(env)
        
        while not done and vis.running:
            vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count, "Mode": mode, "Algorithme": algorithm if mode == "agent" else "Humain"})
            if mode == "human":
                action = vis.get_human_action_mh2(env)
                if action is None:
                    vis.quit()
                    return
            else:
                # Vérifier s'il y a des actions valides
                if len(env.valid_actions) == 0:
                    break
                action = agent.select_action(state)
                # Afficher message et attendre ESPACE
                vis.draw_text("Appuyez sur ESPACE pour voir l'action suivante de l'IA", 120, 500, vis.RED, vis.font)
                vis.update_display()
                wait_for_space(vis)
            state, reward, done, info = env.step(action)
        
        if reward > 0:
            win_count += 1
        
        vis.render_monty_hall2_state(env, episode_info={"Épisode": episode, "Victoires": win_count, "Mode": mode, "Algorithme": algorithm if mode == "agent" else "Humain"})
        pygame.time.wait(1200)
        if not vis.running:
            break
    
    vis.quit()

def run_with_environment_selection():
    """Lance l'application avec sélection de l'environnement."""
    print("🚀 Lancement de l'interface Monty Hall...")
    vis = MontyHallVisualizer()
    print("✅ Visualizer créé")
    
    # Menu de sélection d'environnement
    env_selected = None
    print("🔄 Affichage du menu de sélection d'environnement...")
    while env_selected is None and vis.running:
        vis.clear_screen()
        
        # Fond dégradé
        for i in range(vis.height):
            color = (
                int(230 - i * 80 / vis.height),
                int(245 - i * 60 / vis.height),
                int(255 - i * 40 / vis.height)
            )
            pygame.draw.line(vis.screen, color, (0, i), (vis.width, i))
        
        # Titre
        pygame.draw.rect(vis.screen, (33, 150, 243), (0, 0, vis.width, 100), border_radius=0)
        vis.draw_text("MONTY HALL PROBLEM", vis.width//2 - 200, 20, vis.WHITE, vis.large_font)
        vis.draw_text("Choisissez l'environnement", vis.width//2 - 180, 60, vis.WHITE, vis.font)
        
        # Boutons d'environnement
        btn_width = 300
        btn_height = 100
        btn_y = 200
        
        # Bouton Monty Hall 1
        mh1_rect = pygame.Rect(vis.width//2 - btn_width - 50, btn_y, btn_width, btn_height)
        mh1_color = (76, 175, 80) if pygame.mouse.get_pos()[1] >= btn_y and pygame.mouse.get_pos()[1] <= btn_y + btn_height and pygame.mouse.get_pos()[0] >= mh1_rect.x and pygame.mouse.get_pos()[0] <= mh1_rect.x + btn_width else (56, 142, 60)
        pygame.draw.rect(vis.screen, mh1_color, mh1_rect, border_radius=20)
        vis.draw_text("MONTY HALL 1", mh1_rect.x + 80, mh1_rect.y + 25, vis.WHITE, vis.font)
        vis.draw_text("3 portes, 1 voiture", mh1_rect.x + 90, mh1_rect.y + 55, vis.WHITE, vis.small_font)
        
        # Bouton Monty Hall 2
        mh2_rect = pygame.Rect(vis.width//2 + 50, btn_y, btn_width, btn_height)
        mh2_color = (255, 152, 0) if pygame.mouse.get_pos()[1] >= btn_y and pygame.mouse.get_pos()[1] <= btn_y + btn_height and pygame.mouse.get_pos()[0] >= mh2_rect.x and pygame.mouse.get_pos()[0] <= mh2_rect.x + btn_width else (245, 124, 0)
        pygame.draw.rect(vis.screen, mh2_color, mh2_rect, border_radius=20)
        vis.draw_text("MONTY HALL 2", mh2_rect.x + 80, mh2_rect.y + 25, vis.WHITE, vis.font)
        vis.draw_text("5 portes, étapes multiples", mh2_rect.x + 70, mh2_rect.y + 55, vis.WHITE, vis.small_font)
        
        # Instructions
        vis.draw_text("Cliquez sur un environnement pour commencer", vis.width//2 - 250, 350, vis.BLACK, vis.font)
        
        vis.update_display()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vis.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    vis.running = False
                    return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                if mh1_rect.collidepoint(mouse_x, mouse_y):
                    env_selected = "mh1"
                elif mh2_rect.collidepoint(mouse_x, mouse_y):
                    env_selected = "mh2"
        
        vis.clock.tick(vis.fps)
    
    print(f"🎯 Environnement sélectionné : {env_selected}")
    
    # Lancer l'environnement sélectionné
    if env_selected == "mh1":
        print("🎮 Lancement de Monty Hall 1...")
        run_monty_hall1()
    elif env_selected == "mh2":
        print("🎮 Lancement de Monty Hall 2...")
        run_monty_hall2()
    else:
        print("❌ Aucun environnement sélectionné")

if __name__ == "__main__":
    # Lancer avec sélection d'environnement
    run_with_environment_selection() 