import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Réseau de neurones pour le DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent DQN
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95      # Facteur de réduction
        self.epsilon = 1.0     # Exploration initiale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def check_winner(self, player, state):
        """Vérifie si le joueur `player` a gagné."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
            [0, 4, 8], [2, 4, 6]              # Diagonales
        ]
        for combo in winning_combinations:
            if all(state[i] == player for i in combo):
                return True
        return False

    def act(self, state):
        """Choisit une action en tenant compte des règles heuristiques."""
        available_actions = [i for i in range(9) if state[i] == 0]
        
        if not available_actions:
            return None  # Aucune action possible (plateau plein)

        # Règle 1 : Si l'agent peut gagner en un coup, jouer ce coup
        for action in available_actions:
            next_state = state.copy()
            next_state[action] = 1  # L'agent est X
            if self.check_winner(1, next_state):
                return action

        # Règle 2 : Si l'adversaire peut gagner au prochain coup, bloquer ce coup
        for action in available_actions:
            next_state = state.copy()
            next_state[action] = -1  # L'adversaire est O
            if self.check_winner(-1, next_state):
                return action

        # Règle 3 : Priorité au centre (case 4) si disponible
        if 4 in available_actions:
            return 4

        # Sinon, utiliser le modèle DQN
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).numpy().flatten()

        # Sélectionner la meilleure action parmi celles disponibles
        valid_q_values = {i: q_values[i] for i in available_actions}
        best_action = max(valid_q_values, key=valid_q_values.get)

        return best_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            target_f = self.model(state).clone().detach()
            target_f[action] = target

            output = self.model(state)
            loss = self.loss_fn(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name, weights_only=True))

        self.model.eval()