Lien vers le lien GITHUB de notre projet : https://github.com/selmarahim/tictactao.git

Instructions d'exécution
=========================

1. **Installation des bibliothèques**

   Pour installer les bibliothèques nécessaires, exécutez la commande suivante dans votre terminal :

 
      pip install -r requirements.txt

2. **Entraînement de l'agent**

   Si vous souhaitez entraîner l'agent, vous pouvez le faire avec la commande suivante :

      python train.py

   **Note** : Ce processus n'est pas nécessaire si vous souhaitez simplement jouer, car une version pré-entraînée de l'agent est déjà enregistrée et sera utilisée pour jouer. ( tic_tac_tao_dqn.pth)

3. **Lancer le serveur Flask**

   Une fois l'agent prêt, lancez le serveur Flask avec la commande suivante :

 
      python api.py

4. **Lancer l'interface web**

   Enfin, ouvrez l'interface web en lançant le fichier `index.html` avec l'extension **Live Server** dans votre éditeur de code (Visual Studio Code).
