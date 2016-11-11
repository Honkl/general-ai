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
        private static Random rng = new Random(42);

        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            string pythonScript = "\"" + args[0] + "\"";
            string pythonExe = "\"" + args[1] + "\"";

            RunGame(pythonScript, pythonExe);
        }

        /// <summary>
        /// Starts a single Alhambra game.
        /// </summary>
        /// <param name="pythonScript">Python script to evaluate AI's move.</param>
        /// <param name="pythonExe">Python EXE file to execute .py script.</param>
        private static void RunGame(string pythonScript, string pythonExe)
        {
            //Config file for AI (relative path to master "general-ai/Game-interfaces" directory
            string configFile = " Game-interfaces\\Alhambra\\Alhambra_config.txt"; 

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = pythonExe;
            start.Arguments = pythonScript + configFile;
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardInput = true;

            using (Process process = Process.Start(start))
            {
                StreamWriter writer = process.StandardInput;
                StreamReader reader = process.StandardOutput;

                JsonMessageObject.InitStaticValues();

                Stopwatch sw = new Stopwatch();
                sw.Start();

                Controller c = CreateGame(NumberOfPlayers, rng, reader, writer);

                bool ok = false;
                try
                {
                    c.ExecuteNewMove();
                    ok = true;
                }
                catch (Exception e)
                {
                    if (e.GetType() == typeof(AlhambraException) && e.Message.Contains("CYCLE"))
                    {
                        Console.WriteLine("A cyclic game detected. Starting new game...");
                    }
                    else
                    {
                        Console.WriteLine("An exception occured in the game (" + e.Message + ")");
                        Console.WriteLine(e.StackTrace);
                    }
                    ok = false;
                }

                sw.Stop();
                Console.WriteLine("OK=" + ok);
                Console.WriteLine("Time=" + sw.Elapsed);

                for (int ID = 0; ID < NumberOfPlayers; ID++)
                {
                    Console.WriteLine("AI=" + c.players[ID].AI.ToString());
                    Console.WriteLine(c.game.points[ID]);
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
                    //AI = new GeneralAI(game, reader, writer);
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
