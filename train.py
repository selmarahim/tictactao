import numpy as np
import json
from agent import Agent
from environment import TicTacToeEnv

# Paramètres d'entraînement
EPISODES = 5000  
BATCH_SIZE = 32  # taille du minibatch utilisé par l'agent utilisé pour apprentissage par renforcement 

env = TicTacToeEnv()
state_size = 9 # nbr delements dans letat du jeu 3x3
action_size = 9 # nbr dactions possibles
agent = Agent(state_size, action_size) # création de lagent qui utilise DQN POUR JOUER 


# Liste pour stocker la récompense finale de chaque épisode
stats = []

for e in range(EPISODES): # excecute lentrainement sur EPISODES
    state = env.reset() # reinstaliser lenv avant le debut de lepisode
    done = False # initialiser a false, et mise a true lorsque partie terminée
    total_reward = 0 #  Initialisation de la variable qui accumulera la récompense totale de l'épisode
    while not done: 
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.replay(BATCH_SIZE)
    agent.update_target_model()

 # Stocker la récompense finale (ex: 1, -1 ou 0)
    stats.append(total_reward)

    if e % 100 == 0:
        print(f"Épisode {e}/{EPISODES}, Récompense : {total_reward}")

agent.save("tic_tac_toe_dqn.pth")
# Sauvegarder les statistiques dans un fichier JSON
with open("stats.json", "w") as f:
    json.dump(stats, f)
    
print(" Entraînement terminé, modèle sauvegardé.")
