using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Alhambra;

namespace AlhambraInterface
{
    /// <summary>
    /// Universal object that will encode game information.
    /// </summary>
    class JsonMessageObject
    {
        public double[] state
        {
            get;
            private set;
        }

        public int current_phase
        {
            get;
            private set;
        }

        private static List<Card> Cards = null;
        private static List<Building> Buildings = null;

        public static void InitStaticValues()
        {
            if (Cards != null || Buildings != null)
            {
                throw new AlhambraException("Reinitializing AlhambraInterface storage.");
            }

            Cards = new List<Card>();
            Buildings = new List<Building>();

            CardType type = CardType.Blue;
            foreach (CardType t in Enum.GetValues(typeof(CardType)))
            {
                for (int value = Card.MinValue; value <= Card.MaxValue; value++)
                {
                    Cards.Add(new Card(type, value));
                }
            }

            CodeStorage cs = new CodeStorage();
            cs.InitializeStorage();
            Buildings = cs.GetBuildingList();
        }

        /// <summary>
        /// Initializes a new instance of JsonMessageObject with the specified paramters.
        /// </summary>
        /// <param name="representedPlayer">Currently represented player.</param>
        /// <param name="gamePhase">Current game phase.</param>
        public JsonMessageObject(Player representedPlayer, int gamePhase)
        {
            state = EncodeState(representedPlayer);
            current_phase = gamePhase;
        }

        private double[] EncodeState(Player representedPlayer)
        {
            List<double> state = new List<double>();

            Game game = representedPlayer.game;
            state.Add(game.NumberOfPlayers);

            int maxNumberOfPlayers = 6;
            foreach (int p in game.points)
            {
                state.Add(p);
            }
            for (int i = 0; i < maxNumberOfPlayers - game.points.Length; i++)
            {
                state.Add(0);
            }

            foreach (Card c in Cards)
            {
                state.Add((int)c.Type);
                state.Add(c.Value);
                state.Add(GetNumberOfCards(c, representedPlayer.cards));
            }

            foreach (Card c in game.cardsOnMarket)
            {
                state.Add((int)c.Type);
                state.Add(c.Value);
            }

            // TODO: Karty ostatních hráčů (počty)???

            foreach (Building b in Buildings)
            {
                state.Add((int)b.Type);
                state.Add(b.Value);
                state.Add(b.IsInStorage ? 1 : 0);
                foreach (bool value in b.Walls)
                {
                    state.Add(value ? 1 : 0);
                }

                state.Add((representedPlayer.constructed.Contains(b) || representedPlayer.postponed.Contains(b)) ? 1 : 0);

                if (representedPlayer.constructed.Contains(b))
                {
                    state.Add(b.Position.Row);
                    state.Add(b.Position.Column);
                }
                else
                {
                    state.Add(0);
                    state.Add(0);
                }
            }

            foreach (Building b in game.buildingsOnMarket)
            {
                if (b == null)
                {
                    state.AddRange(new double[] { 0, 0, 0, 0, 0, 0 });
                    continue;
                }
                state.Add((int)b.Type);
                state.Add(b.Value);
                foreach (bool value in b.Walls)
                {
                    state.Add(value ? 1 : 0);
                }
            }
            return state.ToArray();
        }

        /// <summary>
        /// Determines how many times is the specified card contained in the list.
        /// </summary>
        /// <param name="c">Card to count.</param>
        /// <param name="cards">List where to count.</param>
        /// <returns></returns>
        private int GetNumberOfCards(Card card, List<Card> cards)
        {
            int result = 0;
            foreach (Card c in cards)
            {
                if (c.Equals(card))
                {
                    result++;
                }
            }
            return result;
        }

        /// <summary>
        /// Converts this object to JSON string.
        /// </summary>
        /// <returns>A string JSON representation of the current object.</returns>
        public string ConvertToJson()
        {
            string json = JsonConvert.SerializeObject(this);
            //Console.WriteLine(json);
            return json;
        }

        public double[] Decode(string respond)
        {
            List<double> result = new List<double>();
            foreach (string part in respond.Split(' '))
            {
                result.Add(double.Parse(part));
            }

            return result.ToArray();
        }
    }
}
