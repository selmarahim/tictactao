// État initial du plateau
let board = [0, 0, 0, 0, 0, 0, 0, 0, 0]; // 0 = vide, 1 = X (IA), -1 = O (joueur)
let currentPlayer = null; // Aucun tour avant que le choix ne soit fait
let gameOver = false;

// Éléments du DOM
const cells = document.querySelectorAll('.cell');
const statusText = document.getElementById('status');
const resetButton = document.getElementById('reset-button');

// Éléments pour le choix initial
const choiceDiv = document.getElementById('choice');
const playerFirstBtn = document.getElementById('player-first');
const aiFirstBtn = document.getElementById('ai-first');

// Ajout des écouteurs d'événements sur les cases
cells.forEach(cell => {
  cell.addEventListener('click', handleCellClick);
});

// Boutons pour choisir qui commence
playerFirstBtn.addEventListener('click', () => {
  currentPlayer = -1; // Le joueur (O) commence
  statusText.textContent = "Votre tour (O).";
  choiceDiv.style.display = 'none';
});

aiFirstBtn.addEventListener('click', () => {
  currentPlayer = 1; // L'IA (X) commence
  statusText.textContent = "Tour de l'IA...";
  choiceDiv.style.display = 'none';
  playAI(); // Lancer le coup de l'IA dès le départ
});

// Bouton de réinitialisation
resetButton.addEventListener('click', resetGame);

function handleCellClick(event) {
  // Si aucun choix n'a été fait ou si le jeu est terminé, ignorer le clic
  if (currentPlayer === null || gameOver) return;

  const cell = event.target;
  const index = cell.getAttribute('data-index');

  // Vérifier si la case est déjà occupée
  if (board[index] !== 0) {
    statusText.textContent = "Case déjà occupée, réessayez.";
    return;
  }

  // Jouer le coup du joueur
  board[index] = currentPlayer;
  cell.textContent = currentPlayer === -1 ? 'O' : 'X';
  cell.style.pointerEvents = 'none';

  // Vérifier si le joueur actuel a gagné
  if (checkWinner(currentPlayer)) {
    statusText.textContent = currentPlayer === -1 ? "Vous avez gagné !" : "L'IA a gagné !";
    gameOver = true;
    return;
  }

  // Vérifier si le plateau est plein (match nul)
  if (board.every(cell => cell !== 0)) {
    statusText.textContent = "Match nul !";
    gameOver = true;
    return;
  }

  // Changer de tour
  currentPlayer *= -1;
  if (currentPlayer === 1) {
    statusText.textContent = "Tour de l'IA...";
    playAI();
  } else {
    statusText.textContent = "Votre tour (O).";
  }
}

// Fonction asynchrone pour le coup de l'IA en appelant l'API Flask
async function playAI() {
  if (gameOver) return;

  try {
    const response = await fetch('http://127.0.0.1:5000/play', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: board })
    });
    const data = await response.json();
    const action = data.action;

    // Jouer le coup de l'IA
    board[action] = currentPlayer;
    cells[action].textContent = 'X';
    cells[action].style.pointerEvents = 'none';

    // Vérifier si l'IA a gagné
    if (checkWinner(currentPlayer)) {
      statusText.textContent = "L'IA a gagné !";
      gameOver = true;
      return;
    }

    // Vérifier si le plateau est plein (match nul)
    if (board.every(cell => cell !== 0)) {
      statusText.textContent = "Match nul !";
      gameOver = true;
      return;
    }

    // Changer de tour pour le joueur
    currentPlayer *= -1;
    statusText.textContent = "Votre tour (O).";
  } catch (error) {
    console.error("Erreur lors du coup de l'IA :", error);
    statusText.textContent = "Erreur lors du coup de l'IA.";
  }
}

// Fonction pour vérifier si un joueur a gagné
function checkWinner(player) {
  const winningCombinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Lignes
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Colonnes
    [0, 4, 8], [2, 4, 6]             // Diagonales
  ];
  return winningCombinations.some(combo =>
    combo.every(index => board[index] === player)
  );
}

// Fonction pour réinitialiser le jeu
function resetGame() {
  board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  currentPlayer = null;
  gameOver = false;
  statusText.textContent = "Choisissez qui commence.";
  choiceDiv.style.display = 'block';
  cells.forEach(cell => {
    cell.textContent = '';
    cell.style.pointerEvents = 'auto';
  });
}

// --- Statistiques d'apprentissage ---
// Récupérer et afficher le graphique des statistiques avec Chart.js
async function fetchStats() {
  try {
    const response = await fetch('http://127.0.0.1:5000/stats');
    const statsData = await response.json();
    renderChart(statsData);
  } catch (error) {
    console.error("Erreur lors de la récupération des statistiques :", error);
  }
}

function renderChart(data) {
  const ctx = document.getElementById('statsChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map((_, index) => index + 1),
      datasets: [{
        label: 'Récompense par épisode',
        data: data,
        backgroundColor: 'rgba(116, 185, 255, 0.2)',
        borderColor: 'rgba(116, 185, 255, 1)',
        borderWidth: 2,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Épisode' }
        },
        y: {
          display: true,
          title: { display: true, text: 'Récompense' }
        }
      }
    }
  });
}

// Récupérer les statistiques dès le chargement de la page
window.addEventListener('load', fetchStats);