from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from agent import Agent

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes depuis n'importe quel domaine

# Initialiser l'agent
state_size = 9
action_size = 9
agent = Agent(state_size, action_size)
agent.load("tic_tac_toe_dqn.pth")

@app.route('/play', methods=['POST'])
def play():
    try:
        data = request.json
        state = data['state']
        action = agent.act(state)
        return jsonify({'action': action})
    except Exception as e:
        print("Erreur dans l'API :", e)
        return jsonify({'error': str(e)}), 500

# Nouvel endpoint pour les statistiques d'entraînement
@app.route('/stats', methods=['GET'])
def stats():
    try:
        with open("stats.json", "r") as f:
            stats_data = json.load(f)
        return jsonify(stats_data)
    except Exception as e:
        print("Erreur lors de la récupération des statistiques :", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
