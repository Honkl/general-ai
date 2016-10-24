import numpy as np
import os
import json

REQUEST = 'request.json'
ANSWER = 'answer.txt'

np.random.seed(42)

while (True):
    line = input()

    if (line == "END"):
        break

    request_data = json.loads(line)

    moves = request_data['PossibleMoves']
    index = np.random.randint(len(moves))

    print(str(index))


