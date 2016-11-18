# General AI Project
## About
General AI is a project where we try to develop a model which can be learned to play multiple games. The games
do not need to be similar in a term of gameplay - they can be vastly different. We are currently using four
games in this project:
* Alhambra (card board game)
* 2048 (simple sliding block puzzle)
* TORCS (car racing simulator)
* Mario (a well known arcade game)

For each game there must be specific interface implemented (in term of communication with AI). There are three layers (levels) of abstraction.
1. Higher layer - a controller that is available to "call the game" a get its result.
2. Mid layer - a game itself
3. Lower layer - an AI script that answers game's requests.

The communication between layers is via redirected standard I/O, using information encoded into JSON strings.

## Project structure
### Controller
Contains the core of AI in Python.

### Game-interfaces
Interfaces for every game used. Either here or in a separate repository.