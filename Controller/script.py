import numpy as np
import os
import json

REQUEST = 'request.json'
ANSWER = 'answer.txt'

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
f = open(os.path.join(__location__, REQUEST))
request_data = json.load(f)
target = open(os.path.join(__location__, ANSWER), 'w')

moves = request_data['PossibleMoves']
index = np.random.randint(len(moves))

target.write(str(index))
target.close()
