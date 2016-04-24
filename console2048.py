from __future__ import print_function

import os
import sys
from model import Game

# Python 2/3 compatibility.
if sys.version_info[0] == 2:
    range = xrange
    input = raw_input


def _getch_windows(prompt):
    """
    Windows specific version of getch.  Special keys like arrows actually post
    two key events.  If you want to use these keys you can create a dictionary
    and return the result of looking up the appropriate second key within the
    if block.
    """
    print(prompt, end="")
    key = msvcrt.getch()
    if ord(key) == 224:
        key = msvcrt.getch()
        return key
    print(key.decode())
    return key.decode()


def _getch_linux(prompt):
    """Linux specific version of getch."""
    print(prompt, end="")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    char = None
    try:
        char = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
    print(char)
    return char


# Set version of getch to use based on operating system.
if sys.platform[:3] == 'win':
    import msvcrt
    getch = _getch_windows
else:
    import termios
    getch = _getch_linux


def main():
    """
    Get user input.
    Update game state.
    Display updates to user.
    """
    keypad = "adws"
    game = Game(*map(int, sys.argv[1:]))
    game.display()
    while True:
        get_input = getch("Enter direction (w/a/s/d): ")
        if get_input in keypad:
            game.move(keypad.index(get_input))
        elif get_input == "q":
            break
        else:
            print("\nInvalid choice.")
            continue
        if game.end:
            game.display()
            print("You Lose!")
            break
        game.display()
    print("Thanks for playing.")


if __name__ == "__main__":
    main()
