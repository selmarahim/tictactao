from agent import Agent
from environment import TicTacToeEnv

# Fonction pour afficher le plateau proprement
def render_board(state):
    symbols = {1: "X", -1: "O", 0: " "}
    board = [symbols[state[i]] for i in range(9)]
    print(f"""
     {board[0]} | {board[1]} | {board[2]}
    ---+---+---
     {board[3]} | {board[4]} | {board[5]}
    ---+---+---
     {board[6]} | {board[7]} | {board[8]}
    """)

# Jouer contre l'agent
def play_against_agent(agent, env):
    print("🎮 Tic-Tac-Toe : Vous êtes O (humain), l'agent est X.")
    print("Positions des cases :")
    print("""
     0 | 1 | 2
    ---+---+---
     3 | 4 | 5
    ---+---+---
     6 | 7 | 8
    """)

    state = env.reset()
    human_first = int(input("Voulez-vous commencer ? (1 = Oui, 0 = Non) : "))
    
    # Définir qui commence
    if human_first == 1:
        env.current_player = -1  # Le joueur humain (O) commence
    else:
        env.current_player = 1  # L'agent (X) commence

    done = False
    while not done:
        render_board(state)

        if env.current_player == -1:  # Tour du joueur (O)
            valid_move = False
            while not valid_move:
                try:
                    action = int(input("🧑‍💻 Votre tour (O). Choisissez une case (0-8) : "))
                    if 0 <= action <= 8 and state[action] == 0:
                        valid_move = True
                    else:
                        print("⛔ Case invalide, réessayez.")
                except ValueError:
                    print("⛔ Entrez un nombre entre 0 et 8.")
        else:  # Tour de l'agent (X)
            print("🤖 Tour de l'agent (X)...")
            action = agent.act(state)

        # Appliquer le coup
        state, reward, done = env.step(action)

        # Vérifier immédiatement si la partie est terminée
        if done:
            break

    # Affichage final
    render_board(state)
    if reward == 1:
        print("❌ L'agent a gagné !")
    elif reward == -1:
        print("✅ Félicitations, vous avez gagné !")
    else:
        print("🤝 Match nul !")

# Initialisation de l'agent et de l'environnement
env = TicTacToeEnv()
state_size = 9
action_size = 9
agent = Agent(state_size, action_size)

# Charger le modèle entraîné
agent.load("tic_tac_toe_dqn.pth")

# Jouer contre l'agent
play_against_agent(agent, env)