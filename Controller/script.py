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

    # In case of using files:
    #__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    #f = open(os.path.join(__location__, REQUEST))
    #request_data = json.load(f)
    #target = open(os.path.join(__location__, ANSWER), 'w')

    request_data = json.loads(line)

    moves = request_data['PossibleMoves']
    index = np.random.randint(len(moves))

    #target.write(str(index))
    #target.close()

    print(str(index))


