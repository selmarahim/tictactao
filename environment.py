import numpy as np
#classe qui gere letat du jeu, les actions possibles, la logique des regles
class TicTacToeEnv:
    def __init__(self): # on initialise le plateau vide
        self.state = [0] * 9  # Plateau vide (0 = vide, 1 = X agent , -1 = O user )
        self.current_player = 1  # L'agent (X) commence par défaut
    # reinstallise l'environnement 
    def reset(self):
        """Réinitialiser l'environnement."""
        self.state = [0] * 9
        self.current_player = 1
        return self.state # retourne l'état du plateau après la reinitialisation 0
    # verifie si un joueur donné a gagné
    def check_winner(self, player):
        """Vérifie si le joueur `player` a gagné."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
            [0, 4, 8], [2, 4, 6]              # Diagonales
        ] # on cherche dans les combinaisons gagnantes
        for combo in winning_combinations:
            if all(self.state[i] == player for i in combo):
                return True
        return False 
    # methode step excecute un coup sur le plateau
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
