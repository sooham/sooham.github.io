/**
 * Function to load CSS file dynamically
 * @param {string} href - The path to the CSS file to load
 */
function loadCSS(href) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = href;
    document.head.appendChild(link);
}

/**
 * Loads a JSON file from the given URL and returns a Promise that resolves to the parsed JSON data.
 * @param {string} url - The path to the JSON file to load
 * @returns {Promise<any>} - Promise resolving to the parsed JSON data
 */
function loadJSON(url) {
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load JSON from ${url}: ${response.statusText}`);
            }
            return response.json();
        });
}

/**
 * @typedef {Object} LettersSides
 * @property {string[]} top - Array of 3 + letters for the top side
 * @property {string[]} right - Array of 3 + letters for the right side
 * @property {string[]} bottom - Array of 3 + letters for the bottom side
 * @property {string[]} left - Array of 3 + letters for the left side
 */

/**
 * @typedef {Object} DisplayConfig
 * @property {number} [gameSize=500] - The size of the game board in pixels
 * @property {number} [margin=50] - The margin around the game square in pixels
 * @property {number} [circleRadius=12] - The radius of the letter circles in pixels
 * @property {number} [wordCircleGap=5] - The gap between the word circles and the game square in pixels
 * @property {number} [borderThickness=4] - The thickness of the border around the game square in pixels
 */

const DEFAULT_GAME_SIZE = 450;
const DEFAULT_MARGIN = 55;
const DEFAULT_CIRCLE_RADIUS = 12;
const DEFAULT_WORD_CIRCLE_GAP = 5;
const DEFAULT_BORDER_THICKNESS = 5;
const DEFAULT_WORDS_FILE = 'filtered_words_dictionary.json';

class LetterBoxed {
    /**
     * Creates a new LetterBoxed game instance
     * @param {HTMLElement} container - The container element for the game
     * @param {LettersSides} letters - The letters to be used in the game, organized by side
     * @param {DisplayConfig} [displayConfig={}] - The configuration for the display
     */
    constructor(container, letters, displayConfig = {}, wordsFile = DEFAULT_WORDS_FILE) {
        this.validateLetters(letters)
        this.letters = letters;
        this.totalLetters = this.letters.top.length + this.letters.right.length + this.letters.bottom.length + this.letters.left.length;
        this.wordsFile = wordsFile;
        this.loadWords();

        this.container = container;
        this.currentWord = '';
        this.usedWords = [];
        this.visitedLetters = new Set();
        this.letterElements = new Map(); // Map letters to their DOM elements
        
        // Display configuration with defaults
        this.config = {
            gameSize: displayConfig.gameSize || DEFAULT_GAME_SIZE,
            margin: displayConfig.margin || DEFAULT_MARGIN,
            circleRadius: displayConfig.circleRadius || DEFAULT_CIRCLE_RADIUS, // Half of 24px width
            wordCircleGap: displayConfig.wordCircleGap || DEFAULT_WORD_CIRCLE_GAP,
            borderThickness: displayConfig.borderThickness || DEFAULT_BORDER_THICKNESS,

            ...displayConfig
        };
        
        this.init();
    }

    loadWords() {
        // Ensure the path is relative to the current JS file location
        loadJSON('./' + this.wordsFile)
            .then(data => {
                // Get the set of allowed letters for this game
                const allowedLetters = new Set([
                    ...this.letters.top,
                    ...this.letters.right,
                    ...this.letters.bottom,
                    ...this.letters.left
                ].map(l => l.toUpperCase()));

                // Get all words from the dictionary
                const allWords = Object.keys(data);
                const chunkSize = 2000; // Tune for best performance
                const chunks = [];
                for (let i = 0; i < allWords.length; i += chunkSize) {
                    chunks.push(allWords.slice(i, i + chunkSize));
                }

                // Each chunk is filtered in a microtask (not true thread parallelism, but parallelizable in browser event loop)
                return Promise.all(
                    chunks.map(chunk => {
                        return new Promise(resolve => {
                            const filtered = {};
                            for (const word of chunk) {
                                // word is a string, so split into array of letters
                                const wordLetters = word.toUpperCase().split('');
                                // Only keep words that use only allowed letters
                                if (wordLetters.every(letter => allowedLetters.has(letter))) {
                                    filtered[word.toUpperCase()] = data[word];
                                }
                            }
                            resolve(filtered);
                        });
                    })
                ).then(filteredChunks => {
                    // Merge all filtered chunks
                    this.words = Object.assign({}, ...filteredChunks);
                });
            })
            .catch(err => {
                console.error('Failed to load words file:', this.wordsFile, err);
                this.words = {};
            });
    }

    validateLetters(letters) {
        // Check that all sides exist
        if (!letters.top || !letters.right || !letters.bottom || !letters.left) {
            throw new Error('All sides (top, right, bottom, left) must be defined');
        }

        // Combine all letters into a single array
        const allLetters = [
            ...letters.top,
            ...letters.right, 
            ...letters.bottom,
            ...letters.left
        ];

        // Check that all letters are uppercase
        for (const letter of allLetters) {
            if (typeof letter !== 'string' || letter.length !== 1 || letter !== letter.toUpperCase()) {
                throw new Error('All letters must be uppercase single characters');
            }
        }

        // Check for uniqueness by converting to Set
        const uniqueLetters = new Set(allLetters);
        if (uniqueLetters.size !== allLetters.length) {
            throw new Error('All letters must be unique');
        }
    }

    /**
     * Sets up the main game DOM structure
     * @private
     */
    setupGameDOM() {
        this.container.className = 'letter-boxed-container';
        this.container.style.width = this.config.gameSize + 'px';
        this.container.style.height = (this.config.gameSize + 300) + 'px'; // Extra height for controls and settings

        const letterBoxedContainer = document.createElement('div');
        letterBoxedContainer.className = 'letter-boxed-container';

        // Create current word display
        const currentWordDisplay = document.createElement('div');
        currentWordDisplay.className = 'current-word';
        this.currentWordDisplay = currentWordDisplay;

        // Create game board
        const gameBoard = document.createElement('div');
        gameBoard.className = 'game-board';
        this.gameBoard = gameBoard;

        // Create square
        const square = document.createElement('div');
        square.className = 'game-square';
        square.style.top = this.config.margin + 'px';
        square.style.left = this.config.margin + 'px';
        square.style.width = (this.config.gameSize - this.config.margin * 2) + 'px';
        square.style.height = (this.config.gameSize - this.config.margin * 2) + 'px';
        square.style.borderWidth = Math.min(this.config.borderThickness, this.config.circleRadius) + 'px';

        // Create container for lines
        this.linesContainer = document.createElement('div');
        this.linesContainer.className = 'lines-container';
        this.linesContainer.style.position = 'absolute';
        this.linesContainer.style.top = '0';
        this.linesContainer.style.left = '0';
        this.linesContainer.style.width = '100%';
        this.linesContainer.style.height = '100%';

        // Add letters and circles
        this.createLetters(gameBoard);

        // Create controls
        const controls = document.createElement('div');
        controls.className = 'controls';

        // Create buttons
        const restartBtn = document.createElement('button');
        restartBtn.textContent = 'Restart';
        restartBtn.className = 'game-button';

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.className = 'game-button';

        const enterBtn = document.createElement('button');
        enterBtn.textContent = 'Enter';
        enterBtn.className = 'game-button';

        // Add event listeners
        enterBtn.addEventListener('click', () => this.submitWord());
        deleteBtn.addEventListener('click', () => this.deleteLastLetter());
        restartBtn.addEventListener('click', () => this.restart());

        // Append elements
        controls.appendChild(restartBtn);
        controls.appendChild(deleteBtn);
        controls.appendChild(enterBtn);

        gameBoard.appendChild(square);
        gameBoard.appendChild(this.linesContainer);


        this.container.appendChild(currentWordDisplay);
        this.container.appendChild(gameBoard);
        this.container.appendChild(controls);

        // Create settings panel
        if (this.config.showSettings) {
            const settingsPanel = this.createSettingsPanel();
            this.container.appendChild(settingsPanel);
        }
    }

    /**
     * Creates the settings panel with sliders for game configuration
     * @private
     * @returns {HTMLElement} The settings panel element
     */
    createSettingsPanel() {
        const settingsPanel = document.createElement('div');
        settingsPanel.className = 'settings-panel';
        settingsPanel.style.marginTop = '20px';
        settingsPanel.style.padding = '15px';
        settingsPanel.style.border = '1px solid #ccc';
        settingsPanel.style.circleRadius = '8px';
        settingsPanel.style.backgroundColor = '#f9f9f9';

        const title = document.createElement('h3');
        title.textContent = 'Settings';
        title.style.margin = '0 0 15px 0';
        title.style.fontSize = '16px';
        settingsPanel.appendChild(title);

        // Create sliders for each setting
        const settings = [
            { key: 'gameSize', label: 'Game Size', min: 300, max: 800, step: 50 },
            { key: 'margin', label: 'Margin', min: 20, max: 100, step: 5 },
            { key: 'circleRadius', label: 'Circle Radius', min: 8, max: 40, step: 2 },
            { key: 'wordCircleGap', label: 'Word Circle Gap', min: 0, max: 20, step: 5 },
            { key: 'borderThickness', label: 'Border Thickness', min: 0, max: 20, step: 1 }
        ];

        settings.forEach(setting => {
            const sliderContainer = document.createElement('div');
            sliderContainer.style.marginBottom = '10px';
            sliderContainer.style.display = 'flex';
            sliderContainer.style.alignItems = 'center';
            sliderContainer.style.gap = '10px';

            const label = document.createElement('label');
            label.textContent = setting.label + ':';
            label.style.minWidth = '100px';
            label.style.fontSize = '14px';

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = setting.min;
            slider.max = setting.max;
            slider.step = setting.step;
            slider.value = this.config[setting.key];
            slider.style.flex = '1';

            const valueDisplay = document.createElement('span');
            valueDisplay.textContent = this.config[setting.key];
            valueDisplay.style.minWidth = '40px';
            valueDisplay.style.fontSize = '14px';
            valueDisplay.style.fontWeight = 'bold';

            slider.addEventListener('input', (e) => {
                const newValue = parseInt(e.target.value);
                valueDisplay.textContent = newValue;
                this.updateSetting(setting.key, newValue);
            });

            sliderContainer.appendChild(label);
            sliderContainer.appendChild(slider);
            sliderContainer.appendChild(valueDisplay);
            settingsPanel.appendChild(sliderContainer);
        });

        return settingsPanel;
    }

    /**
     * Updates a configuration setting and rebuilds the game
     * @private
     * @param {string} key - The configuration key to update
     * @param {number} value - The new value for the setting
     */
    updateSetting(key, value) {
        this.config[key] = value;
        this.rebuildGame();
    }

    /**
     * Rebuilds the game with current configuration while preserving game state
     * @private
     */
    rebuildGame() {
        // Store current game state
        const currentState = {
            currentWord: this.currentWord,
            usedWords: [...this.usedWords],
            visitedLetters: new Set(this.visitedLetters),
        };

        // Update container size
        this.container.style.width = this.config.gameSize + 'px';
        this.container.style.height = (this.config.gameSize + 300) + 'px';


        // Clear existing letters and lines
        this.letterElements.clear();
        this.linesContainer.innerHTML = '';

        // Remove old letters and circles
        const oldLetters = this.gameBoard.querySelectorAll('.letter-circle, .letter-text');
        oldLetters.forEach(el => el.remove());

        // Update square size and position
        const square = this.gameBoard.querySelector('.game-square');
        square.style.top = this.config.margin + 'px';
        square.style.left = this.config.margin + 'px';
        square.style.width = (this.config.gameSize - this.config.margin * 2) + 'px';
        square.style.height = (this.config.gameSize - this.config.margin * 2) + 'px';
        square.style.borderWidth = Math.min(this.config.borderThickness, this.config.circleRadius) + 'px';

        // Recreate letters with new positions
        this.createLetters(this.gameBoard);

        // Restore game state
        this.currentWord = currentState.currentWord;
        this.usedWords = currentState.usedWords;
        this.visitedLetters = currentState.visitedLetters;

        // Update display
        this.currentWordDisplay.textContent = [...this.usedWords, this.currentWord].join(', ');

        // Restore visited letter styles
        this.visitedLetters.forEach(letter => {
            const letterElement = this.letterElements.get(letter);
            if (letterElement) {
                letterElement.element.classList.add('visited');
            }
        });


        // Rebuild the path for used words
        this.drawConnections();
    }

    /**
     * Creates letter elements and positions them on the game board
     * @private
     * @param {HTMLElement} gameBoard - The game board element
     */
    createLetters(gameBoard) {
        const margin = this.config.margin;
        const size = this.config.gameSize - this.config.margin * 2;
        const radius = this.config.circleRadius;
        const textCirclePadding = this.config.wordCircleGap;

        const topSpacing = size / (this.letters.top.length + 1);
        const rightSpacing = size / (this.letters.right.length + 1);
        const bottomSpacing = size / (this.letters.bottom.length + 1);
        const leftSpacing = size / (this.letters.left.length + 1);
        const fontSize = 24;
        const borderWidth = this.config.borderThickness;

        // Create letters for each side
        const createSideLetters = (side, letters, getCirclePosition) => {
            /*
            getCirclePosition needs to give the center point of the circle in x and y coordinates
            */  
            letters.forEach((letter, i) => {
                const circlePos = getCirclePosition(i);
                
                // Create circle
                const letterCircle = document.createElement('div');
                letterCircle.className = 'letter-circle';
                letterCircle.style.borderWidth = Math.min(borderWidth, radius) + 'px';
                letterCircle.style.width = (radius * 2) + 'px';
                letterCircle.style.height = (radius * 2) + 'px';
                letterCircle.style.left = (circlePos.x - radius) + 'px';
                letterCircle.style.top = (circlePos.y - radius) + 'px';
                letterCircle.addEventListener('click', () => this.handleLetterClick(letter));
                
                // Create text element
                const letterText = document.createElement('div');
                letterText.className = 'letter-text';
                letterText.textContent = letter;
                letterText.style.width = fontSize + 'px';
                letterText.style.height = fontSize + 'px';

                let letterXOffset = 0;
                let letterYOffset = 0;
                if (side === 'top') {
                    // offset X by half the size of text
                    letterXOffset = - fontSize / 2;
                    // offset Y by the radius of the circle + padding above
                    letterYOffset = -radius - textCirclePadding - fontSize;
                } else if (side === 'right') {
                    letterXOffset = radius + textCirclePadding;
                    letterYOffset = - fontSize / 2;
                } else if (side === 'bottom') {
                    letterXOffset = - fontSize / 2;
                    letterYOffset = radius + textCirclePadding;
                } else if (side === 'left') {
                    letterXOffset = - radius - textCirclePadding - fontSize;
                    letterYOffset = - fontSize / 2;
                }

                letterText.style.left = (circlePos.x + letterXOffset) + 'px';
                letterText.style.top = (circlePos.y + letterYOffset) + 'px';

                gameBoard.appendChild(letterCircle);
                gameBoard.appendChild(letterText);
                this.letterElements.set(letter, {
                    element: letterCircle,
                    x: circlePos.x,
                    y: circlePos.y
                });
            });
        };

        // Top side - circles on border, letters above
        createSideLetters('top', this.letters.top, 
            (i) => ({
                x: margin + topSpacing + (i * topSpacing),
                y: margin + Math.min(borderWidth, radius) / 2
            })
        );

        // Right side - circles on border, letters to the right
        createSideLetters('right', this.letters.right, 
            (i) => ({
                x: margin + size - Math.min(borderWidth, radius) / 2,
                y: margin + rightSpacing + (i * rightSpacing)
            })
        );

        // Bottom side - circles on border, letters below
        createSideLetters('bottom', this.letters.bottom, 
            (i) => ({
                x: margin + bottomSpacing + (i * bottomSpacing),
                y: margin + size - Math.min(borderWidth, radius) / 2
            })
        );

        // Left side - circles on border, letters to the left
        createSideLetters('left', this.letters.left, 
            (i) => ({
                x: margin + Math.min(borderWidth, radius) / 2,
                y: margin + leftSpacing + (i * leftSpacing)
            })
        );
    }

    /**
     * Handles letter click events and updates the current word
     * @private
     * @param {string} letter - The clicked letter
     */
    handleLetterClick(letter) {
        let lastLetter = null;
        if (this.currentWord.length === 0) {
            if (this.usedWords.length > 0) {
                lastLetter = this.usedWords[this.usedWords.length - 1][this.usedWords[this.usedWords.length - 1].length - 1];
            }
        } else {
            lastLetter = this.currentWord[this.currentWord.length - 1];
        }

        if (lastLetter && this.areLettersOnSameSide(letter, lastLetter)) {
            return;
        }

        this.currentWord += letter;
        this.currentWordDisplay.textContent = [...this.usedWords, this.currentWord].join(', ');

        this.drawConnections();
    }

    /**
     * Checks if two letter elements are on the same side of the square
     * @private
     * @param {HTMLElement} elem1 - First letter element
     * @param {HTMLElement} elem2 - Second letter element
     * @returns {boolean} True if letters are on the same side
     */
    areLettersOnSameSide(letter1, letter2) {
        return this.letters.top.includes(letter1) && this.letters.top.includes(letter2) ||
            this.letters.right.includes(letter1) && this.letters.right.includes(letter2) ||
            this.letters.bottom.includes(letter1) && this.letters.bottom.includes(letter2) ||
            this.letters.left.includes(letter1) && this.letters.left.includes(letter2);
    }

    /**
     * Updates the visual path connecting letters
     * @private
     */
    drawConnections() {

        this.linesContainer.innerHTML = '';

        // reset all the letter elements to not visited or submitted
        // this.letterElements.forEach(letterElement => {
        //     letterElement.element.classList.remove('visited');
        //     letterElement.element.classList.remove('submitted');
        // });

        // draw the connections for the used words
        for (let i = 0; i < this.usedWords.length; i++) {
            const word = this.usedWords[i];
            for (let j = 0; j < word.length - 1; j++) {
                this.letterElements.get(word[j]).element.classList.add('submitted');
                this.drawConnectionLine(word[j], word[j + 1], true);
            }
        }

        // mark the last letter of the last used word as submitted
        if (this.usedWords.length > 0) {
            this.letterElements.get(
                this.usedWords[this.usedWords.length - 1][this.usedWords[this.usedWords.length - 1].length - 1]
            ).element.classList.add('submitted');
        }

        if (this.currentWord.length > 0) {
            this.letterElements.get(this.currentWord[0]).element.classList.add('visited');
            for (let i = 0; i < this.currentWord.length - 1; i++) {
                this.letterElements.get(this.currentWord[i + 1]).element.classList.add('visited');
                this.drawConnectionLine(this.currentWord[i], this.currentWord[i + 1], false);
            }
        }

    }

    /**
     * Draw connection line between two letters 
    */
    drawConnectionLine(start, end, visited = false) {
        const startElement = this.letterElements.get(start);
        const endElement = this.letterElements.get(end);

        const line = document.createElement('div');
        line.className = 'connection-line';
        if (visited) {
            line.classList.add('visited');
        }
        line.dataset.start = `${start}`;
        line.dataset.end = `${end}`;

        // Calculate line properties
        const dx = endElement.x - startElement.x;
        const dy = endElement.y - startElement.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
        const thickness = Math.min(this.config.borderThickness, this.config.circleRadius);

        line.style.width = length + 'px';
        line.style.height = thickness + 'px';
        line.style.transform = `rotate(${angle}deg)`;
        line.style.borderWidth = thickness / 2 + 'px';
        line.style.left = (startElement.x) + 'px';
        line.style.top = (startElement.y - thickness / 2) + 'px';

        this.linesContainer.appendChild(line);
    }


    /**
     * Submits the current word if valid
     * @public
     */
    submitWord() {
        if (this.currentWord.length < 3) {
            // TODO: do not alert
            alert('Words must be at least 3 letters long');
            return;
        }

        /* TODO: check if the currentWord is a valid word, load the file called filtered_words_dictionary.json
        that has keys of english words and values that are not relevant and check if the currentWord is a valid  
         */
        console.log('checking if', this.currentWord, 'is a valid word');
        const wordKeys = Object.keys(this.words);
        console.log('First 10 words:', wordKeys.slice(0, 10));

        if (!this.words[this.currentWord]) {
            // TODO: do not alert
            alert(`${this.currentWord} is not a valid word`);
            return;
        }

        for (const letter of this.currentWord) {
            this.visitedLetters.add(letter);
        }

        this.usedWords.push(this.currentWord);
        this.currentWord = this.currentWord[this.currentWord.length - 1];
        this.currentWordDisplay.textContent = [...this.usedWords, this.currentWord].join(', ');
        this.drawConnections();

        if (this.visitedLetters.size === this.totalLetters) {
            alert("Congratulations! You've used all letters!");
        }
    }

    /**
     * Deletes the last letter from the current word
     * @public
     */
    deleteLastLetter() {
        if (this.currentWord.length === 0) {
            return;
        }

        if (this.currentWord.length > 1) {
            console.log('> 1 char left in current word', this.currentWord, this.usedWords)
            let lastLetter = this.currentWord[this.currentWord.length - 1];
            // unmark the last letter as visited
            this.letterElements.get(lastLetter).element.classList.remove('visited');

            // remove the last letter from the current word
            this.currentWord = this.currentWord.slice(0, -1);
        } else {
            console.log('<= 1 char left in current word', this.currentWord, this.usedWords)
            console.log('unmark the first letter of the current word', this.currentWord[0])
            this.letterElements.get(this.currentWord[0]).element.classList.remove('visited');
            this.letterElements.get(this.currentWord[0]).element.classList.remove('submitted');
            // if the length is 1 and there are previous words, replace the last last word with the current word
            if (this.usedWords.length > 0) {
                // remove the the first letter of the current word as visited and submitted
                // replace the current word with the last used word
                let lastWord = this.usedWords[this.usedWords.length - 1];
                // remove submitted from every letter of the last word
                for (let i = this.usedWords.length > 1 ? 1 : 0 ; i < lastWord.length; i++) {
                    console.log('unmark the letter', lastWord, lastWord[i], 'as submitted')
                    this.letterElements.get(lastWord[i]).element.classList.remove('submitted');
                }
                console.log('make the last word the current word', lastWord)
                this.currentWord = lastWord;
                this.usedWords.pop();
            } else {
                // if there are no previous words, clear the current word
                console.log('no previous words, clear the current word')
                this.currentWord = '';
                this.currentWordDisplay.textContent = '';
                this.drawConnections();
            }
        }
        this.currentWordDisplay.textContent = [...this.usedWords, this.currentWord].join(', ');
        this.drawConnections();
    }

    /**
     * Restarts the game to initial state
     * @public
     */
    restart() {
        this.currentWord = '';
        this.usedWords = [];
        this.visitedLetters = new Set();
        this.lastLetter = null;
        this.currentWordDisplay.textContent = '';
        this.drawConnections();
        
        // Reset letter styles
        this.letterElements.forEach(letterElement => {
            letterElement.element.classList.remove('visited');
            letterElement.element.classList.remove('submitted');
        });
    }

    /**
     * Initializes the game
     * @private
     */
    init() {
        this.setupGameDOM();
    }
}

/**
 * Initialize game when the script loads
 * Creates a new LetterBoxed game instance with default configuration
 */
document.addEventListener('DOMContentLoaded', () => {
    loadCSS('letterboxed.css');
    
    // Dictionary mapping suffixes to game configurations
    const gameConfigs = {
        'main': {
            letters: {
                top: ['A', 'I', 'E'],
                right: ['R', 'T', 'K'], 
                bottom: ['L', 'U', 'M'],
                left: ['B', 'O', 'H']
            },
            displayConfig: {
            showSettings: false,
            gameSize: 450,
            margin: 55,
            circleRadius: 12,
            wordCircleGap: 5,
            borderThickness: 5
            }
        }
    };

    // Find all elements with class starting with letter-boxed-game-
    const gameContainers = document.querySelectorAll('[class^="letter-boxed-game-"]');
    
    gameContainers.forEach(gameContainer => {
        // Get the suffix from the class name
        const className = Array.from(gameContainer.classList)
            .find(c => c.startsWith('letter-boxed-game-'));
        const suffix = className?.split('letter-boxed-game-')[1];
        
        /** @type {DisplayConfig} */
        const gameConfig = gameConfigs[suffix];
        
        new LetterBoxed(gameContainer, gameConfig.letters, gameConfig.displayConfig, DEFAULT_WORDS_FILE);
    });
});