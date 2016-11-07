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

            int count = 1;
            int ok = 0;
            for (int i = 0; i < count; i++)
            {
                if (RunGame(pythonScript, pythonExe))
                {
                    ok++;
                }
                Console.WriteLine("Completed: " + (i + 1) + "/" + count);
                Console.WriteLine("OK: " + ok + "/" + count);
            }
        }

        private static bool RunGame(string pythonScript, string pythonExe)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = pythonExe;
            start.Arguments = pythonScript;
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardInput = true;

            using (Process process = Process.Start(start))
            {
                StreamWriter writer = process.StandardInput;
                StreamReader reader = process.StandardOutput;

                writer.WriteLine("Alhambra");
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
                Console.WriteLine("Ok: " + ok);
                Console.WriteLine("Time: " + sw.Elapsed);

                for (int ID = 0; ID < NumberOfPlayers; ID++)
                {
                    Console.WriteLine("AI: " + c.players[ID].AI.ToString() + " SCORE: " + c.game.points[ID]);
                }

                writer.Write("END");
                writer.Close();
                reader.Close();
                return ok;
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
