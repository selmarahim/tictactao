from flask import Flask, jsonify, request
from flask_cors import CORS  # Importer Flask-CORS

app = Flask(__name__)
CORS(app)  # Autoriser toutes les requêtes CORS


# Route pour jouer
@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    
    # Vérifier si des données sont envoyées
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Ici, on simule une réponse de l'API 
    response = {
        "board": [["", "", ""], ["", "X", ""], ["", "", ""]],
        "next_player": "O",
        "winner": None
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
