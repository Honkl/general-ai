# Alhambra interface
Game Alhambra has been implemented in my bachelor thesis so I decided to use this game in general AI project. 

Original Alhambra player was made using evolution (SEA with floats). We'll use same agent with replaced weights. In the original agent, we used evolution to determine value of some 'game rules' and then we selected the most valuable rule / move. This will remain same but to determine the most valuable 'rule' we call general AI process.

Code of original Alhambra can be downloaded [here](https://is.cuni.cz/webapps/zzp/detail/152723/23205131/?q=%7B%22______searchform___search%22%3A%22alhambra%22%2C%22______searchform___butsearch%22%3A%22Vyhledat%22%2C%22PNzzpSearchListbasic%22%3A1%7D&lang=en). In this project we'are using only compiled dll, stored in `AlhambraInterface/lib/`.
