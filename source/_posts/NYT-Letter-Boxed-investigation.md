title: Solving the The New York Times Letter Boxed
mathjax: true
comments: true
tags:
  - Letter Boxed
  - algorithms
  - tree-search
  - games
categories: []
date: 2025-07-11 21:01:00
---
# From Browsing to Algorithmic Play
Recently I binged the [_New York Times_ games](https://www.nytimes.com/crosswords) section library of games. I did a month of _Wordles_, a number of _sudokus_  till I found myself rather bored. Then out of sheer curiousity, I started playing the NYT's _Letter Boxed_.

<p align="center">
  <img src="/2025/07/11/NYT-Letter-Boxed-investigation/letter_boxed_main_page.png" alt="NYT Letter Boxed Main Page" style="max-width:250px; height:auto; width:100%;">
</p>

##  Rules of Letter Boxed

Like most games presented by the NYT, Letter Boxed and its rules are simple, simple enough  that you can play the game on a piece of paper and do not need any feedback from the computer after every turn like _Wordle_. There is no hidden information here to uncover.

* You have a list of valid words from a dictionary, and you are given a square with 3 letters on each side, total of 12 letters.

<p align="center">
  <img src="/2025/07/11/NYT-Letter-Boxed-investigation/letter_boxed_static.png" alt="NYT Letter Boxed Today" style="max-width:250px; height:auto; width:100%;">
</p>

* Starting from any letter, connect letters from to form a word in the dictionary.
* Consecutive letters cannot be from the same side. So EAT is not allowed
* Your next starting letter is last letter of the word you connected. BET > TUB > BAT

* You must visit all the letters in the square at least once.
* Each word must be at least 3 letters long.

Objective is to find a path that meets all the above.

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <div>
    <img src="/2025/07/11/NYT-Letter-Boxed-investigation/BET_TUB.gif" alt="BET and TUB is a valid start" style="max-width:250px; height:auto; width:100%;">
    <div style="font-size: 0.95em; color: #666; margin-top: 0.5em;">
      <em>BET</em> then <em>TUB</em> is allowed
    </div>
  </div>
  <div>
    <img src="/2025/07/11/NYT-Letter-Boxed-investigation/TOKAMAK.gif" alt="TOKAMAK is allowed" style="max-width:250px; height:auto; width:100%;">
    <div style="font-size: 0.95em; color: #666; margin-top: 0.5em;">
      <em>TOKAMAK</em> (a fusion device) is allowed
    </div>
  </div>
</div>

### Try playing a game for yourself!

<div class="letter-boxed-game-main"></div>
<script src="/2025/07/11/NYT-Letter-Boxed-investigation/letterboxed.js"></script>

## How can we find a solution for Letter Boxed? 

I found this game great because *Letter Boxed* is not a game traditionally analyzed through the lens of data structures and algorithms -  while *Sudoku* easilly fits into the mold of normal CSP problems, *Wordle* fits nicely with the concept of information gain and entropy. Letter Boxed is a bit unclear to solve, but can see how the traditional concepts of graphs and search can be used to pry out an **optimal** solution.

Here the **optimality** can be the following measure
- The number of letters traversed.
- The number of words used.

and there are extensions and simplifications of the game that we did not consider
- what if we used a N-sided shape instead of square?
- Given a seet of words, can a Letter Boxed be created?
- What if we used a K dimensional Letter Cube? 
- What if we have arbitrary N letters per side?





