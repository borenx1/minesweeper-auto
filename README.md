# minesweeper-auto
*Created early 2017*
A script to play Windows Minesweeper using screen capture and automatic mouse actions

## Instructions
Download all files

Install the following python libraries:
- numpy
  - `pip install numpy`
  - https://numpy.org/
- pywin32
  - `pip install pywin32`
  - https://github.com/mhammond/pywin32
- pillow
  - `pip install pillow`
  - https://pillow.readthedocs.io/en/stable/index.html
- opencv
  - `pip install opencv-python`
  - https://pypi.org/project/opencv-python/
- pyautogui
  - `pip install pyautogui`
  - https://pyautogui.readthedocs.io/en/latest/index.html
  
1. Run "Minesweeper Auto.py", this will open Minesweeper and an "Analysis Window"
2. Keep the Minesweeper window in the forefront, this will allow the script to "see" the grid
3. To auto-play Minesweeper, click on the "Analysis Window" to focus it and press the keyboard button "a", this will turn auto-play mode on (see bottom right of Analysis Window)
4. The script will keep playing until there are no more 100% moves. You may need to click on random squares to open up possible moves. Not every game is 100% winnable, you may need to play a few games to win, the win rate depends on the difficulty.
5. End the script forcefully (Ctrl-C) to end the script.
