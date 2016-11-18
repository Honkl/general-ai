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

        private static List<Card> allCards = null;
        private static List<Building> allBuildings = null;
        /// <summary>
        /// Number of card colors.
        /// </summary>
        private const int TotalCardTypes = 4;
        /// <summary>
        /// Number of building colors.
        /// </summary>
        private const int TotalBuildingTypes = 6;
        /// <summary>
        /// Number of max value of one card.
        /// </summary>
        private const int MaxCardValue = 9;
        /// <summary>
        /// Number of max playres that can play the game.
        /// </summary>
        private const int MaxNumberOfPlayers = 6;
        /// <summary>
        /// Each card is three times in the package.
        /// </summary>
        private const int MaxNumberOfOneCardType = 3;
        /// <summary>
        /// Max value of building in whole game.
        /// </summary>
        private const int MaxBuildingValue = 13;


        public static void InitStaticValues()
        {
            if (allCards != null || allBuildings != null)
            {
                throw new AlhambraException("Reinitializing AlhambraInterface storage.");
            }

            allCards = new List<Card>();
            allBuildings = new List<Building>();

            CardType type = CardType.Blue;
            foreach (CardType t in Enum.GetValues(typeof(CardType)))
            {
                for (int value = Card.MinValue; value <= Card.MaxValue; value++)
                {
                    allCards.Add(new Card(type, value));
                }
            }

            CodeStorage cs = new CodeStorage();
            cs.InitializeStorage();
            allBuildings = cs.GetBuildingList();
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

        /// <summary>
        /// Encodes the state of the game into array of double. Most of the values are rescaled to interval of [0,1].
        /// </summary>
        /// <param name="representedPlayer">Player whose data will be encoded.</param>
        /// <returns>Double array with all encoded game data of the specified player.</returns>
        private double[] EncodeState(Player representedPlayer)
        {
            List<double> state = new List<double>();

            Game game = representedPlayer.game;

            state.Add((double)game.NumberOfPlayers / (double)MaxNumberOfPlayers);

            foreach (Card c in allCards)
            {
                state.Add((double)c.Type / (double)TotalCardTypes);
                state.Add((double)c.Value / (double)MaxCardValue);
                state.Add((double)GetNumberOfCards(c, representedPlayer.cards) / (double)MaxNumberOfOneCardType);
            }

            for (int i = 0; i < Game.MarketSize; i++)
            {
                if (!game.nonAvailableCardsOnMarket[i])
                {
                    Card c = game.cardsOnMarket[i];
                    state.Add((double)c.Type / (double)TotalCardTypes);
                    state.Add((double)c.Value / (double)MaxCardValue);
                }
                else
                {
                    state.Add(0);
                    state.Add(0);
                }
            }

            foreach (Building b in allBuildings)
            {
                state.Add((double)b.Type / (double)TotalBuildingTypes);
                state.Add((double)b.Value / (double)MaxBuildingValue);
                state.Add(b.IsInStorage ? 1 : 0);
                foreach (bool value in b.Walls)
                {
                    state.Add(value ? 1 : 0);
                }

                state.Add((representedPlayer.constructed.Contains(b) || representedPlayer.postponed.Contains(b)) ? 1 : 0);

                bool found = false;
                foreach (var constr in representedPlayer.constructed)
                {
                    if (constr.Equals(b))
                    {
                        state.Add((double)constr.Position.Row / (double)Game.MapSize);
                        state.Add((double)constr.Position.Column / (double)Game.MapSize);
                        found = true;
                        break;
                    }
                }
                if (!found)
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
                state.Add((double)b.Type / (double)TotalBuildingTypes);
                state.Add((double)b.Value / (double)MaxBuildingValue);
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
