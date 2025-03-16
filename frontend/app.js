document.getElementById("game-board").addEventListener("click", function (e) {
    let cell = e.target;
    if (!cell.textContent) {
        cell.textContent = 'X';  // L'agent joue (X)

        // Envoyer l'action au backend pour que l'IA joue
        fetch('/play', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action: cell.id })
        }).then(response => response.json())
          .then(data => {
              if (data.winner) {
                  alert(data.winner + " wins!");
              }
              // Mettre à jour l'interface en fonction du jeu
          });
    }
});