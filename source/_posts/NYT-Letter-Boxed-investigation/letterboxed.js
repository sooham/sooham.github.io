class LetterBoxed {
    constructor(container) {
        this.container = container;
        this.letters = {
            top: ['A', 'I', 'E'],
            right: ['R', 'T', 'K'],
            bottom: ['L', 'U', 'M'],
            left: ['B', 'O', 'H']
        };
        this.currentWord = '';
        this.usedWords = [];
        this.visitedLetters = new Set();
        this.lastLetter = null;
        this.canvas = null;
        this.ctx = null;
        this.points = [];
        this.currentPath = [];
        this.isDrawing = false;

        this.init();
    }

    init() {
        // Create game container
        this.container.style.position = 'relative';
        this.container.style.width = '400px';
        this.container.style.height = '500px';
        this.container.style.margin = '0 auto';

        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 400;
        this.canvas.height = 400;
        this.canvas.style.border = '2px solid #333';
        this.ctx = this.canvas.getContext('2d');
        
        // Create input and buttons container
        const controls = document.createElement('div');
        controls.style.marginTop = '20px';
        controls.style.textAlign = 'center';
        
        // Create word input
        this.wordInput = document.createElement('input');
        this.wordInput.type = 'text';
        this.wordInput.style.fontSize = '20px';
        this.wordInput.style.padding = '5px';
        this.wordInput.style.marginRight = '10px';
        this.wordInput.style.textTransform = 'uppercase';
        
        // Create buttons
        const enterBtn = document.createElement('button');
        enterBtn.textContent = 'Enter';
        enterBtn.style.fontSize = '18px';
        enterBtn.style.padding = '5px 15px';
        enterBtn.style.marginRight = '10px';
        
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.style.fontSize = '18px';
        deleteBtn.style.padding = '5px 15px';
        deleteBtn.style.marginRight = '10px';
        
        const restartBtn = document.createElement('button');
        restartBtn.textContent = 'Restart';
        restartBtn.style.fontSize = '18px';
        restartBtn.style.padding = '5px 15px';

        // Add event listeners
        enterBtn.addEventListener('click', () => this.submitWord());
        deleteBtn.addEventListener('click', () => this.deleteLastLetter());
        restartBtn.addEventListener('click', () => this.restart());
        this.wordInput.addEventListener('input', (e) => this.handleInput(e));
        
        // Append elements
        controls.appendChild(this.wordInput);
        controls.appendChild(enterBtn);
        controls.appendChild(deleteBtn);
        controls.appendChild(restartBtn);
        
        this.container.appendChild(this.canvas);
        this.container.appendChild(controls);

        // Calculate letter positions
        this.calculateLetterPositions();
        this.draw();

        // Add canvas event listeners
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseleave', () => this.stopDrawing());
    }

    calculateLetterPositions() {
        const margin = 50;
        const size = this.canvas.width - (2 * margin);
        
        // Calculate positions for each side
        this.letterPositions = {
            top: [],
            right: [],
            bottom: [],
            left: []
        };

        // Top side
        for (let i = 0; i < 3; i++) {
            this.letterPositions.top.push({
                x: margin + (i * size/2),
                y: margin,
                letter: this.letters.top[i]
            });
        }

        // Right side
        for (let i = 0; i < 3; i++) {
            this.letterPositions.right.push({
                x: this.canvas.width - margin,
                y: margin + (i * size/2),
                letter: this.letters.right[i]
            });
        }

        // Bottom side (reverse order)
        for (let i = 2; i >= 0; i--) {
            this.letterPositions.bottom.push({
                x: margin + (i * size/2),
                y: this.canvas.height - margin,
                letter: this.letters.bottom[2-i]
            });
        }

        // Left side (reverse order)
        for (let i = 2; i >= 0; i--) {
            this.letterPositions.left.push({
                x: margin,
                y: margin + (i * size/2),
                letter: this.letters.left[2-i]
            });
        }

        // Create flat array of all points
        this.points = [
            ...this.letterPositions.top,
            ...this.letterPositions.right,
            ...this.letterPositions.bottom,
            ...this.letterPositions.left
        ];
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw the square
        this.ctx.beginPath();
        this.ctx.moveTo(this.letterPositions.top[0].x, this.letterPositions.top[0].y);
        this.ctx.lineTo(this.letterPositions.top[2].x, this.letterPositions.top[2].y);
        this.ctx.lineTo(this.letterPositions.right[0].x, this.letterPositions.right[0].y);
        this.ctx.lineTo(this.letterPositions.right[2].x, this.letterPositions.right[2].y);
        this.ctx.lineTo(this.letterPositions.bottom[2].x, this.letterPositions.bottom[2].y);
        this.ctx.lineTo(this.letterPositions.bottom[0].x, this.letterPositions.bottom[0].y);
        this.ctx.lineTo(this.letterPositions.left[2].x, this.letterPositions.left[2].y);
        this.ctx.lineTo(this.letterPositions.left[0].x, this.letterPositions.left[0].y);
        this.ctx.closePath();
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw letters
        this.ctx.font = '24px Arial';
        this.ctx.fillStyle = '#000';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        this.points.forEach(point => {
            // Draw circle
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 20, 0, Math.PI * 2);
            this.ctx.fillStyle = this.visitedLetters.has(point.letter) ? '#e0e0e0' : '#fff';
            this.ctx.fill();
            this.ctx.stroke();
            
            // Draw letter
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(point.letter, point.x, point.y);
        });

        // Draw current path
        if (this.currentPath.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.currentPath[0].x, this.currentPath[0].y);
            for (let i = 1; i < this.currentPath.length; i++) {
                this.ctx.lineTo(this.currentPath[i].x, this.currentPath[i].y);
            }
            this.ctx.strokeStyle = '#007AFF';
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
        }
    }

    handleInput(e) {
        const input = e.target.value.toUpperCase();
        if (input !== this.currentWord) {
            this.currentWord = input;
            this.updatePath();
        }
    }

    updatePath() {
        this.currentPath = [];
        let lastSide = null;
        
        for (let i = 0; i < this.currentWord.length; i++) {
            const letter = this.currentWord[i];
            const point = this.points.find(p => p.letter === letter);
            
            if (point) {
                // Determine current side
                let currentSide;
                if (this.letterPositions.top.includes(point)) currentSide = 'top';
                else if (this.letterPositions.right.includes(point)) currentSide = 'right';
                else if (this.letterPositions.bottom.includes(point)) currentSide = 'bottom';
                else if (this.letterPositions.left.includes(point)) currentSide = 'left';
                
                // Check if letter is from same side
                if (currentSide === lastSide) {
                    this.wordInput.style.color = 'red';
                    return;
                }
                
                this.currentPath.push(point);
                lastSide = currentSide;
                this.wordInput.style.color = 'black';
            }
        }
        
        this.draw();
    }

    submitWord() {
        if (this.currentWord.length < 3) {
            alert('Words must be at least 3 letters long');
            return;
        }
        
        // Check if word starts with last letter of previous word
        if (this.lastLetter && this.currentWord[0] !== this.lastLetter) {
            alert('Word must start with the last letter of the previous word');
            return;
        }
        
        // Add letters to visited set
        for (const letter of this.currentWord) {
            this.visitedLetters.add(letter);
        }
        
        this.usedWords.push(this.currentWord);
        this.lastLetter = this.currentWord[this.currentWord.length - 1];
        this.currentWord = '';
        this.wordInput.value = '';
        this.currentPath = [];
        this.draw();
        
        // Check if all letters have been used
        if (this.visitedLetters.size === 12) {
            alert("Congratulations! You've used all letters!");
        }
    }

    deleteLastLetter() {
        this.wordInput.value = this.wordInput.value.slice(0, -1);
        this.currentWord = this.wordInput.value;
        this.updatePath();
    }

    restart() {
        this.currentWord = '';
        this.usedWords = [];
        this.visitedLetters = new Set();
        this.lastLetter = null;
        this.currentPath = [];
        this.wordInput.value = '';
        this.draw();
    }

    startDrawing(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Find if click is near any letter
        const clickedPoint = this.points.find(point => {
            const distance = Math.sqrt(
                Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)
            );
            return distance < 20;
        });
        
        if (clickedPoint) {
            this.isDrawing = true;
            this.currentPath = [clickedPoint];
            this.currentWord = clickedPoint.letter;
            this.wordInput.value = this.currentWord;
            this.draw();
        }
    }

    stopDrawing() {
        this.isDrawing = false;
    }
}

// Initialize game when the script loads
document.addEventListener('DOMContentLoaded', () => {
    const gameContainer = document.getElementById('letter-boxed-game');
    console.log("Loading Letter Boxed Game")
    if (gameContainer) {
        new LetterBoxed(gameContainer);
    }
}); 