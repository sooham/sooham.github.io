---
title: Solving the The New York Times LetterBoxed
mathjax: true
comments: true
date: 2025-07-11 21:01:01
tags: 
- letterboxed
- algorithms
- tree-search
- games
---

# From Browsing to Algorithmic Play
Recently I binged the [_New York Times_ games](https://www.nytimes.com/crosswords) section library of games. I did a month of _Wordles_, a number of _sudokus_  till I found myself rather bored. Then out of sheer curiousity, I started playing the NYT's _LetterBoxed_.

<p align="center">
  <img src="/2025/07/11/NYT-Letter-Boxed-investigation/letterboxed_main_page.png" alt="NYT Letter Boxed Main Page" style="max-width:250px; height:auto; width:100%;">
</p>

##  Rules of LetterBoxed

Like most games presented by the NYT, LetterBoxed and its rules are simple, simple enough  that you can play the game on a piece of paper and do not need any feedback from the computer after every turn like _Wordle_. There is no hidden information here to uncover.

0. You have a list of valid words from a dictionary, and you are given a square with 3 letters on each side, total of 12 letters.

<p align="center">
  <img src="/2025/07/11/NYT-Letter-Boxed-investigation/letterboxed_static.png" alt="NYT Letter Boxed Today" style="max-width:250px; height:auto; width:100%;">
</p>


1. Starting from any letter, connect letters from to form a word in the dictionary.
2. Consecutive letters cannot be from the same side. 
3. Your next starting letter is last letter of the word you connected i.e EAT > TEAM > MEAL
4. You must visit all the letters in the square at least once.
5. Each word must be at least 3 letters long.

Objective is to find a path.

