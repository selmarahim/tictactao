import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.full((3, 3), "", dtype=str)  # Grille vide
        self.current_player = "X"  # Le joueur commence

    def reset(self):
        """Réinitialise la grille"""
        self.board = np.full((3, 3), "", dtype=str)
        self.current_player = "X"

    def play_move(self, row, col):
        """Joue un coup et retourne le nouvel état"""
        if self.board[row, col] == "":  # Vérifier si la case est libre
            self.board[row, col] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def check_winner(self):
        """Vérifie s'il y a un gagnant"""
        for i in range(3):
            if all(self.board[i, :] == self.board[i, 0]) and self.board[i, 0] != "":
                return self.board[i, 0]  # Ligne gagnante
            if all(self.board[:, i] == self.board[0, i]) and self.board[0, i] != "":
                return self.board[0, i]  # Colonne gagnante
        if all(self.board.diagonal() == self.board[0, 0]) and self.board[0, 0] != "":
            return self.board[0, 0]  # Diagonale principale gagnante
        if all(np.fliplr(self.board).diagonal() == self.board[0, 2]) and self.board[0, 2] != "":
            return self.board[0, 2]  # Diagonale secondaire gagnante
        if "" not in self.board:
            return "Draw"  # Match nul
        return None  # Pas encore de gagnant
