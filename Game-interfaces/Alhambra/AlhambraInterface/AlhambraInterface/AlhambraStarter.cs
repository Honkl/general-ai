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
        private static Random rng = new Random();

        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            string pythonScript = "\"" + args[0] + "\"";
            string pythonExe = "\"" + args[1] + "\"";
            string modelConfigFile = " \"" + args[2] + "\"";
            int gameBatchSize = int.Parse(args[3]);

            string gameConfigFile = " \"Game-interfaces\\Alhambra\\Alhambra_config.json\"";
            RunGame(pythonScript, pythonExe, gameConfigFile, modelConfigFile, gameBatchSize);
        }

        /// <summary>
        /// Starts a single Alhambra game.
        /// </summary>
        /// <param name="pythonScript">Python script to evaluate AI's move.</param>
        /// <param name="pythonExe">Python EXE file to execute .py script.</param>
        /// <param name="gameConfigFile">Game configuration file (number of inputs / outputs for AI.</param>
        /// <param name="modelConfigFile">Model configuration file for AI.</param>
        /// <param name="gameBatchSize">Number of games that will be played and averaged as a result.</param>
        private static void RunGame(string pythonScript, string pythonExe, string gameConfigFile, string modelConfigFile, int gameBatchSize)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = pythonExe;
            start.Arguments = pythonScript + gameConfigFile + modelConfigFile;
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardInput = true;

            using (Process process = Process.Start(start))
            {
                StreamWriter writer = process.StandardInput;
                StreamReader reader = process.StandardOutput;

                JsonMessageObject.InitStaticValues();

                int playedGames = 0;
                double[] avgResults = new double[NumberOfPlayers]; // At index 0 is our general-ai agent
                while (playedGames < gameBatchSize)
                {
                    bool ok = false;
                    Controller c = CreateGame(NumberOfPlayers, rng, reader, writer);
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

                for (int ID = 0; ID < NumberOfPlayers; ID++)
                {
                    avgResults[ID] /= gameBatchSize;
                    Console.WriteLine(avgResults[ID]);
                }

                writer.Write("END");
                writer.Close();
                reader.Close();
            }
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
