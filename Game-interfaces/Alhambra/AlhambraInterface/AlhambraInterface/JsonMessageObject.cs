using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Alhambra;

namespace AlhambraInterface
{
    /// <summary>
    /// Universal object that will encode game information.
    /// </summary>
    class JsonMessageObject
    {

        //
        // SERIALIZED OBJECTS TO JSON (public only)
        // 

        /// <summary>
        /// General array of possible moves for AI. Do not change the name of this array.
        /// </summary>
        public Move[] possible_moves
        {
            get;
            private set;
        }

        public List<Card> Cards
        {
            get;
            private set;

        }

        public List<Building> Constructed
        {
            get;
            private set;
        }

        public List<Building> Postponed
        {
            get;
            private set;
        }

        public int[] Points
        {
            get;
            private set;
        }

        public Building[] BuildingsOnMarket
        {
            get;
            private set;
        }

        public Card[] CardsOnMarket
        {
            get;
            private set;
        }

        private Player representedPlayer;

        public JsonMessageObject(Player representedPlayer)
        {
            Points = representedPlayer.game.points;
            Cards = representedPlayer.cards;
            Postponed = representedPlayer.postponed;
            BuildingsOnMarket = representedPlayer.game.buildingsOnMarket;
            CardsOnMarket = representedPlayer.game.cardsOnMarket;

            this.representedPlayer = representedPlayer;


            MoveGenerator mg = new MoveGenerator(representedPlayer);
            possible_moves = mg.GenerateMoves().ToArray();
        }

        /// <summary>
        /// Converts this object to JSON string.
        /// </summary>
        /// <returns>A string JSON representation of the current object.</returns>
        public string Encode()
        {
            string json = JsonConvert.SerializeObject(this);
            //Console.WriteLine(json);
            return json;
        }

        public Move Decode(string respond)
        {
            int index = int.Parse(respond);

            Move result = possible_moves[index];
            Console.WriteLine(possible_moves.Length);

            if (!representedPlayer.game.IsPermissible(result))
            {
                throw new AlhambraException("Incorrect move has been generated - " + result.ToString());
            }
            return result;
        }
    }
}
