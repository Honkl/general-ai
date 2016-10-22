import numpy as np
import json

REQUEST = 'D:/Disk Google/Master/general-ai/Controller/request.json'
ANSWER = 'D:/Disk Google/Master/general-ai/Controller/answer.txt'


request_data = json.load(open(REQUEST))
target = open(ANSWER, 'w')

moves = request_data['PossibleMoves']
index = np.random.randint(len(moves))

target.write(str(index))
target.close()
