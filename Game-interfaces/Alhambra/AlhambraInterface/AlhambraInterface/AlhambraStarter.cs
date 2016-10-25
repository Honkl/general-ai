using System;
using System.IO;
using System.Diagnostics;
using Alhambra;

namespace AlhambraInterface
{
    class AlhambraStarter
    {
        private static string CurrentLocation = AppDomain.CurrentDomain.BaseDirectory;
        private const string PythonExePath = "C:\\Anaconda2\\envs\\py3k\\python.exe"; // TODO: use general relative path
        private const string PythonScriptPath = "..\\..\\..\\..\\..\\..\\Controller\\script.py";

        private const int NumberOfPlayers = 3;
        private static Random rng = new Random(42);

        static void Main(string[] args)
        {
            int count = 1;
            int ok = 0;
            for (int i = 0; i < count; i++)
            {
                if (RunGame())
                {
                    ok++;
                }
                Console.WriteLine("Completed: " + (i + 1) + "/" + count);
                Console.WriteLine("OK: " + ok + "/" + count);
            }
        }

        private static bool RunGame()
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = PythonExePath;
            start.Arguments = "\"" + CurrentLocation + PythonScriptPath + "\"";
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardInput = true;

            using (Process process = Process.Start(start))
            {
                StreamWriter writer = process.StandardInput;
                StreamReader reader = process.StandardOutput;
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
                    AI = new GeneralAI(game, reader, writer);
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
