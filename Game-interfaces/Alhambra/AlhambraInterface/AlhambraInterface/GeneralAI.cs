using System;
using System.Collections.Generic;
using System.IO;
using Alhambra;

namespace AlhambraInterface
{
    class GeneralAI : IArtificialIntelligence
    {
        private StreamReader reader;
        private StreamWriter writer;
        private Game game;

        public GeneralAI(Game game, StreamReader reader, StreamWriter writer)
        {
            this.reader = reader;
            this.writer = writer;
            this.game = game;
        }

        public Player RepresentedPlayer
        {
            get;
            set;
        }

        public Position GetPositionForFreeBuilding(Building toBeAccepted)
        {
            return Position.None;
        }

        public Move GetSolution()
        {
            JsonMessageObject jmo = new JsonMessageObject(RepresentedPlayer);
            writer.WriteLine(jmo.Encode());

            // Reads python script standard output as a result of AI move
            string output = reader.ReadLine();
            return jmo.Decode(output);

        }

        public void Initialize()
        {
        }
    }
}
