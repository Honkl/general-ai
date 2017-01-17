using System;
using System.IO;
using System.Diagnostics;
using Alhambra;
using System.Threading;
using System.Globalization;

namespace AlhambraInterface
{
    class AlhambraStarter
    {
        private const int NumberOfPlayers = 3;

        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            int seed = int.Parse(args[0]);
            int gameBatchSize = int.Parse(args[1]);

            RunGame(seed, gameBatchSize);
        }

        /// <summary>
        /// Starts a batch of Alhambra games. 
        /// </summary>
        /// <param name="seed">Seed for random generator.</param>
        /// <param name="gameBatchSize">Number of games that will be played and averaged as a result.</param>
        private static void RunGame(int seed, int gameBatchSize)
        {
            StreamReader reader = new StreamReader(Console.OpenStandardInput());
            StreamWriter writer = new StreamWriter(Console.OpenStandardOutput());
            writer.AutoFlush = true;

            JsonMessageObject.InitStaticValues();

            int playedGames = 0;
            Random rnd = new Random(seed);
            double[] avgResults = new double[NumberOfPlayers]; // At index 0 is our general-ai agent
            while (playedGames < gameBatchSize)
            {
                bool ok = false;
                Random rndForGame = new Random(rnd.Next());
                Controller c = CreateGame(NumberOfPlayers, rndForGame, reader, writer);
                try
                {
                    c.ExecuteNewMove();
                    ok = true;
                }
                catch (Exception e)
                {
                    if (e.GetType() == typeof(AlhambraException) && e.Message.Contains("CYCLE"))
                    {
                        // A cyclic game detected...
                    }
                    else
                    {
                        // Some other unexpected exception...
                    }
                }
                if (ok)
                {
                    for (int ID = 0; ID < NumberOfPlayers; ID++)
                    {
                        avgResults[ID] += c.game.points[ID];
                    }
                    playedGames++;
                }
            }

            string result = "SCORES ";
            for (int ID = 0; ID < NumberOfPlayers; ID++)
            {
                avgResults[ID] /= gameBatchSize;
                result += avgResults[ID];
                if (ID < NumberOfPlayers - 1)
                {
                    result += " ";
                }
            }

            writer.WriteLine(result);
            writer.Close();
            reader.Close();
        }

        private static Controller CreateGame(int numberOfPlayers, Random rnd, StreamReader reader, StreamWriter writer)
        {
            Controller controller = new Controller(numberOfPlayers, rnd);
            Game game = controller.game;

            // Create players:
            for (int i = 0; i < numberOfPlayers; i++)
            {
                IArtificialIntelligence AI = null;
                if (i == 0)
                {
                    AI = new AIWeighedMovesV2(game, reader, writer);
                }
                else
                {
                    AI = new AIWeighedMoves(game);
                }
                AI.Initialize();
                controller.CreatePlayer(false, AI, Player.Colors[i]);
            }
            return controller;
        }
    }
}
