# The Open Racing Car Simulator (TORCS)
[TORCS](http://torcs.sourceforge.net/index.php) is a highly portable multi platform open source car racing simulation. It is used as ordinary car racing game, as AI racing game and as research platform.

Game TORCS is used in our project 'General artificial intelligence for game playing'. There is more games, not only TORCS. Also, there are a few things to know:
- Your installation directory of TORCS must be specified in `install_directory.txt`
- TORCS is more comlicated (you need proper version and patch, for more information, please head into official manual [[pdf](https://arxiv.org/pdf/1304.1672v2.pdf)]
- We use Java TORCS client which is connected to local server
- `race_config<port>.xml` files are torcs configuration files, each using specified port and the track settings
- `torcs_starter.bat` is batch script that starts TORCS. It's main 'connection' point between TORCS and our AI.
- `torcs_starter_vis_on.bat` has same purpose but starts TORCS visually.
- More TORCS information (incl. license) could be found on [official site](http://torcs.sourceforge.net/index.php?name=Sections&op=viewarticle&artid=30#c0_1)
