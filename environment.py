import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.state = [0] * 9  # Plateau vide (0 = vide, 1 = X, -1 = O)
        self.current_player = 1  # L'agent (X) commence par défaut

    def reset(self):
        """Réinitialiser l'environnement."""
        self.state = [0] * 9
        self.current_player = 1
        return self.state

    def check_winner(self, player):
        """Vérifie si le joueur `player` a gagné."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
            [0, 4, 8], [2, 4, 6]              # Diagonales
        ]
        for combo in winning_combinations:
            if all(self.state[i] == player for i in combo):
                return True
        return False

    def step(self, action):
        """Joue un coup et retourne le nouvel état, la récompense et si la partie est terminée."""
        if self.state[action] != 0:
            raise ValueError("Mouvement invalide.")

        # Appliquer l'action
        self.state[action] = self.current_player

        # Vérifier si le joueur actuel a gagné
        if self.check_winner(self.current_player):
            reward = 1 if self.current_player == 1 else -1  # 1 pour X (agent), -1 pour O (humain)
            return self.state, reward, True  # La partie est terminée

        # Vérifier si le plateau est plein (match nul)
        if all(cell != 0 for cell in self.state):
            return self.state, 0, True  # Égalité

        # Changer de joueur
        self.current_player *= -1
        return self.state, 0, False  # Partie continue
