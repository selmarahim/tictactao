import numpy as np
import json
from agent import Agent
from environment import TicTacToeEnv

# Paramètres d'entraînement
EPISODES = 5000  
BATCH_SIZE = 32  

env = TicTacToeEnv()
state_size = 9
action_size = 9
agent = Agent(state_size, action_size)


# Liste pour stocker la récompense finale de chaque épisode
stats = []

for e in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
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
    
print("✅ Entraînement terminé, modèle sauvegardé.")
