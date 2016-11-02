using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using Alhambra;

namespace AlhambraInterface
{
    /// <summary>
    /// Implements the weights on which the artificial intelligence decides next move to play.
    /// </summary>
    [Serializable]
    class AIWeighedMovesV2 : IArtificialIntelligence
    {
        /// <summary>
        /// Number of methods/criteria on which AI makes decisions.
        /// </summary>
        public const int NumberOfMethods = 76;
        /// <summary>
        /// Number of weights on which AI makes decisions (for every part
        /// among scoring rounds we have different weights).
        /// </summary>
        public const int NumberOfWeights = NumberOfMethods * Game.NumberOfScoringRounds;
        /// <summary>
        /// Indicates the maximum number of cards that the current AI can take,
        /// if some other type of move is available.
        /// </summary>
        private const int MaxCardsToTakeLimit = 20;
        /// <summary>
        /// Indicates the maximum number of rebuild-moves, that can the current
        /// AI do in a row (if some other type of move is available).
        /// </summary>
        private const int MaxRebuildsInRowLimit = 10;
        /// <summary>
        /// Methods to determine properties of the move or a state of the game.
        /// </summary>
        private Func<object, bool>[] methods;

        /// <summary>
        /// Types of rebuild move - ToAlhambra, ToStorage, Swap.
        /// </summary>
        private enum TypeOfRebuildMove
        {
            ToAlhambra,
            ToStorage,
            Swap
        }

        /// <summary>
        /// Contains static arrays that specifies indices of given methods.
        /// </summary>
        [Serializable]
        public class CriteriaProperties
        {
            // Each array contains indices of methods and weights.
            // These methods are used to determine properties with
            // a given decision making.
            public static readonly int[] TakeCardMoveSelection;
            public static readonly int[] BuyBuildingMoveSelection;
            public static readonly int[] RebuildMoveSelection;

            public static readonly int[] CardsToTakeSelection;

            public static readonly int[] BuildingToBuyOrToAlhSelect;
            public static readonly int[] CardsToPurchaseSelection;

            public static readonly int[] BuildingPositionSelection;

            public static readonly int[] ToAlhambraRebuildSelection;
            public static readonly int[] ToStorageRebuildSelection;
            public static readonly int[] SwapRebuildSelection;

            public static readonly int[] BuildingToStorageSelection;
            public static readonly int[] SwapSelection;

            // Reusing some of methods (so remember their indices in 'methods' array
            public const int IndexOf_HasComplBoundedMethod = 21;
            public const int IndexOf_HasPostponedMethod = 22;

            public static List<int[]> AllCriteriaArrays;

            /// <summary>
            /// Static constructor initializes arrays for criteria (also called when game is deserialized).
            /// </summary>
            static CriteriaProperties()
            {
                int offset = 0;

                int count = 12;
                CriteriaProperties.TakeCardMoveSelection = new int[count];
                FillArray(CriteriaProperties.TakeCardMoveSelection, count, offset);
                offset += count;

                count = 8;
                CriteriaProperties.BuyBuildingMoveSelection = new int[count];
                FillArray(CriteriaProperties.BuyBuildingMoveSelection, count, offset);
                offset += count;

                count = 3;
                CriteriaProperties.RebuildMoveSelection = new int[count];
                FillArray(CriteriaProperties.RebuildMoveSelection, count, offset);
                offset += count;

                count = 12;
                CriteriaProperties.CardsToTakeSelection = new int[count];
                FillArray(CriteriaProperties.CardsToTakeSelection, count, offset);
                offset += count;

                count = 9;
                CriteriaProperties.BuildingToBuyOrToAlhSelect = new int[count];
                FillArray(CriteriaProperties.BuildingToBuyOrToAlhSelect, count, offset);
                offset += count;

                count = 11;
                CriteriaProperties.CardsToPurchaseSelection = new int[count];
                FillArray(CriteriaProperties.CardsToPurchaseSelection, count, offset);
                offset += count;

                count = 7;
                CriteriaProperties.BuildingPositionSelection = new int[count];
                FillArray(CriteriaProperties.BuildingPositionSelection, count, offset);
                offset += count;

                count = 1;
                CriteriaProperties.ToAlhambraRebuildSelection = new int[count];
                // Reusing 'HasPostponedBuilding' method
                CriteriaProperties.ToAlhambraRebuildSelection[0] = IndexOf_HasPostponedMethod;

                count = 1;
                CriteriaProperties.ToStorageRebuildSelection = new int[count];
                // Reusing 'HasCompletelyBoundedAlhambra' method
                CriteriaProperties.ToStorageRebuildSelection[0] = IndexOf_HasComplBoundedMethod;

                count = 1;
                CriteriaProperties.SwapRebuildSelection = new int[count];
                CriteriaProperties.SwapRebuildSelection[0] = offset;
                offset += count;

                count = 9;
                CriteriaProperties.BuildingToStorageSelection = new int[count];
                FillArray(CriteriaProperties.BuildingToStorageSelection, count, offset);
                offset += count;

                count = 4;
                CriteriaProperties.SwapSelection = new int[count];
                FillArray(CriteriaProperties.SwapSelection, count, offset);
                offset += count;

                /////////////////

                AllCriteriaArrays = new List<int[]>();
                AllCriteriaArrays.Add(CriteriaProperties.TakeCardMoveSelection);
                AllCriteriaArrays.Add(CriteriaProperties.BuyBuildingMoveSelection);
                AllCriteriaArrays.Add(CriteriaProperties.RebuildMoveSelection);
                AllCriteriaArrays.Add(CriteriaProperties.CardsToTakeSelection);
                AllCriteriaArrays.Add(CriteriaProperties.BuildingToBuyOrToAlhSelect);
                AllCriteriaArrays.Add(CriteriaProperties.CardsToPurchaseSelection);
                AllCriteriaArrays.Add(CriteriaProperties.BuildingPositionSelection);
                AllCriteriaArrays.Add(CriteriaProperties.ToAlhambraRebuildSelection);
                AllCriteriaArrays.Add(CriteriaProperties.ToStorageRebuildSelection);
                AllCriteriaArrays.Add(CriteriaProperties.SwapRebuildSelection);
                AllCriteriaArrays.Add(CriteriaProperties.BuildingToStorageSelection);
                AllCriteriaArrays.Add(CriteriaProperties.SwapSelection);

                /////////////////
            }

            private static void FillArray(int[] array, int count, int offset)
            {
                for (int i = 0; i < count; i++)
                {
                    array[i] = i + offset;
                }
            }
        }

        /// <summary>
        /// Based on these weights, the AI will make a decisions.
        /// </summary>
        public double[] Weights
        {
            get;
            set;
        }

        /// <summary>
        /// Player who is played by the current 'Alhambra.AIWeighedMoves' instance.
        /// </summary>
        public Player RepresentedPlayer
        {
            get;
            set;
        }

        /// <summary>
        /// Indicates whether the current instance of 'Alhambra.AIWeighedMoves'
        /// were initialized from file succesfully.
        /// </summary>
        public bool InitializedFromFileSuccessfully
        {
            get;
            private set;
        }

        /// <summary>
        /// An instance of 'Alhambra.MoveChecker' used to determine move correctness.
        /// </summary>
        internal readonly MoveChecker checker;

        private Game game;
        private StreamReader reader;
        private StreamWriter writer;

        // '_alreadyPositionedDueMove' is used in some
        // of 'CriteriaProperties.BuildingPositionSelection' methods (used for performace boost).
        private List<Building> _alreadyPositionedDueMove;

        /// <summary>
        /// Represents a number of Rebuild moves in a row.
        /// </summary>
        private int rebuildInRow = 0;

        /// <summary>
        /// Initializes a new instance of Alhambra.AIWeighedMoves. This instance has no weights yet.
        /// </summary>
        /// <param name="game">An instance of 'Alhambra.Game' where the game will be played.</param>
        public AIWeighedMovesV2(Game game, StreamReader reader, StreamWriter writer)
        {
            this.game = game;
            this.reader = reader;
            this.writer = writer;
            checker = new MoveChecker(game);
            _alreadyPositionedDueMove = new List<Building>();
            InitializeMethods();
        }

        /// <summary>
        /// Initializes the current instance of 'Alhambra.AIWeighedMoves'.
        /// </summary>
        public void Initialize()
        {
            InitializeWeights();
        }

        /// <summary>
        /// Initializes Weights of the current artificial intelligence.
        /// </summary>
        public void InitializeWeights()
        {
            try
            {
                string color = GetStringColorForFileName(RepresentedPlayer.Color);
                string fileName = color + "_" + game.NumberOfPlayers + ".txt";
                string directory = "Weights";
                switch (game.NumberOfPlayers)
                {
                    case 2:
                        directory += "\\TwoPlayers";
                        break;
                    case 3:
                        directory += "\\ThreePlayers";
                        break;
                    case 4:
                        directory += "\\FourPlayers";
                        break;
                    case 5:
                        directory += "\\FivePlayers";
                        break;
                    case 6:
                        directory += "\\SixPlayers";
                        break;
                }
                string fullFilePath = directory + "\\" + fileName;
                Weights = ParseFile(fullFilePath);
                InitializedFromFileSuccessfully = true;
            }

            // Encapsulating IOExceptions and double.Parse exceptions
            catch (Exception)
            {
                InitializedFromFileSuccessfully = false;

                // Set default weights
                switch (game.NumberOfPlayers)
                {
                    case 2:
                        Weights = WeightsStorage.WeightsForTwo;
                        break;
                    case 3:
                        Weights = WeightsStorage.WeightsForThree;
                        break;
                    case 4:
                        Weights = WeightsStorage.WeightsForFour;
                        break;
                    case 5:
                        Weights = WeightsStorage.WeightsForFive;
                        break;
                    case 6:
                        Weights = WeightsStorage.WeightsForSix;
                        break;
                    default:
                        throw new AlhambraException("Incorrect number of players within the game.");
                }
            }
        }

        /// <summary>
        /// Parses file with weights and returns them.
        /// </summary>
        /// <returns>Array of weights for specified number of players.</returns>
        private double[] ParseFile(string file)
        {
            StreamReader reader = new StreamReader(file);
            double[] weightsFromFile = new double[NumberOfWeights];

            // We read first 3*76 (NumberOfWeights) of lines from file
            for (int i = 0; i < NumberOfWeights; i++)
            {
                // CultureInfo.InvariantCulture is important (using decimal dot or comma)
                double value = double.Parse(reader.ReadLine(), System.Globalization.CultureInfo.InvariantCulture);
                weightsFromFile[i] = value;
            }
            return weightsFromFile;
        }

        /// <summary>
        /// Gets a string representation of the specified color. This string is not
        /// based on resource files, because these strings are not connected to any
        /// language. They are only in names of files.
        /// </summary>
        /// <param name="color">A color to be translated.</param>
        /// <returns>A string representation of the specified color. Used in file name.</returns>
        private string GetStringColorForFileName(Color color)
        {
            // Strings that we return are not from resources!
            // They are not connected to any language, but they are in file names
            if (color == Color.DeepSkyBlue)
            {
                return "Blue";
            }
            else if (color == Color.Red)
            {
                return "Red";
            }
            else if (color == Color.WhiteSmoke)
            {
                return "White";
            }
            else if (color == Color.Green)
            {
                return "Green";
            }
            else if (color == Color.Orange)
            {
                return "Orange";
            }
            else if (color == Color.Yellow)
            {
                return "Yellow";
            }
            throw new AlhambraException("Unexpected color of player.");
        }

        /// <summary>
        /// Initializes Weights of current artificial intelligence randomly.
        /// </summary>
        public void InitializeWeightsRandom()
        {
            Weights = AlhambraRandom.NextDoubles(NumberOfWeights);
        }

        /// <summary>
        /// Evaluates and selects move to player.
        /// </summary>
        /// <returns>Instance of child class of Move.</returns>
        public Move GetSolution()
        {
            // Determine which type of move AI will play
            Type typeOfMove = GetMoveTypeToPlay();

            // There's no available move within the game
            if (typeOfMove == null)
            {
                return null;
            }

            Move move = null;
            if (typeOfMove == typeof(TakeCardMove))
            {
                TakeCardMove tcm = new TakeCardMove(RepresentedPlayer.ID);
                tcm.takenCards = GetCardsToTake();
                tcm.modified = new List<Building>(game.unresolved);
                Building mainModified = null;
                SolveBuildingPosition(ref mainModified, tcm.modified);

                move = tcm;
            }
            else if (typeOfMove == typeof(BuyBuildingMove))
            {
                CardType typeForPurchase;
                Building toBuy = GetBuildingToBuy(out typeForPurchase);
                List<Card> cardsToPurchase = GetCardsToPurchaseBuilding(toBuy, typeForPurchase);
                BuyBuildingMove bbm = new BuyBuildingMove(RepresentedPlayer.ID, toBuy, cardsToPurchase, typeForPurchase);
                bbm.modified = new List<Building>(game.unresolved);
                SolveBuildingPosition(ref toBuy, bbm.modified); // Set buildings possitions into their properties

                move = bbm;
            }
            else if (typeOfMove == typeof(RebuildMove))
            {
                Building toAlhambra = null;
                Building toStorage = null;
                switch (GetTypeOfRebuildMove())
                {
                    case TypeOfRebuildMove.ToAlhambra:
                        toAlhambra = GetBuildingToAlhambra();
                        break;
                    case TypeOfRebuildMove.ToStorage:
                        toStorage = GetBuildingToStorage();
                        break;
                    case TypeOfRebuildMove.Swap:
                        GetBuildingForSwap(out toAlhambra, out toStorage);
                        toAlhambra.Position = toStorage.Position;
                        toAlhambra.IsInStorage = false;
                        break;
                    default:
                        throw new AlhambraException("Unexpected type of Rebuild move.");
                }

                // Positions of toAlhambra and toStorage buildings are already solved
                RebuildMove rm = new RebuildMove(RepresentedPlayer.ID, toAlhambra, toStorage);
                rm.modified = new List<Building>(game.unresolved);
                ResolvePositionsForRebuild(ref rm);
                move = rm;
            }
            else
            {
                throw new AlhambraException("Unexpected type of move - " + typeOfMove.ToString() + ".");
            }

            if (!game.IsPermissible(move))
            {
                throw new AlhambraException("Incorrect move has been generated - " + move.ToString());
            }

            if (move is RebuildMove)
            {
                rebuildInRow++;
            }
            else
            {
                rebuildInRow = 0;
            }
            if (rebuildInRow >= MaxRebuildsInRowLimit * 2)
            {
                // Cyclic game -- always a Rebuild move is done (only for non GUI games...)
                if (!game.HasAttachedGameForm)
                {
                    throw new AlhambraException("CYCLE");
                }
            }

            return move;
        }

        #region Methods for compute solution

        /// <summary>
        /// Returns a position for a free building. This building can be obtained at the end of the game.
        /// </summary>
        /// <param name="toBeAccepted"></param>
        /// <returns>A position for the building obtained at the end of the game.</returns>
        public Position GetPositionForFreeBuilding(Building toBeAccepted)
        {
            return GetPositionForBuilding(toBeAccepted, new List<Building>(), alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
        }

        /// <summary>
        /// Evaluates sum of appropriate weights computed by a specified methods.
        /// </summary>
        /// <param name="criteriaArray">Indices of methods that will be called.</param>
        /// <param name="argumentForMethod">Argument to computational methods.</param>
        /// <returns>Evaluated sum of appropriate weights.</returns>
        private double GetSumOfWeights(int[] criteriaArray, object argumentForMethod)
        {
            int gamePhase = -1;

            // Determine game phase based on passed array
            for (int i = 0; i < CriteriaProperties.AllCriteriaArrays.Count; i++)
            {
                if (criteriaArray.Equals(CriteriaProperties.AllCriteriaArrays[i]))
                {
                    gamePhase = i;
                }
            }

            JsonMessageObject jmo = new JsonMessageObject(RepresentedPlayer, gamePhase);
            writer.WriteLine(jmo.ConvertToJson());

            // Reads python script standard output as a result of AI move
            string output = reader.ReadLine();
            double[] results = jmo.Decode(output);
            int resultIndex = 0;
            double sum = 0;

            if (criteriaArray.Length != results.Length)
            {
                throw new AlhambraException("Wrong number of results from general AI");
            }

            foreach (int index in criteriaArray)
            {
                if (methods[index](argumentForMethod))
                {
                    sum += results[resultIndex];
                    /**
                    switch (game.Controller.scoreCountingRound)
                    {

                        case ScoreRound.First:
                            sum += Weights[index];
                            break;
                        case ScoreRound.Second:
                            sum += Weights[index + NumberOfMethods];
                            break;
                        case ScoreRound.Third:
                            sum += Weights[index + (2 * NumberOfMethods)];
                            break;
                    }
                    /**/
                }
                resultIndex++;
            }
            return sum;
        }

        /// <summary>
        /// Selects the type of move player should play. Determines among TakeCardMove,
        /// BuyBuildingMove and RebuildMove.
        /// </summary>
        /// <returns>One of types - TakeCardMove, BuyBuildingMove, RebuildMove.</returns>
        private Type GetMoveTypeToPlay()
        {
            double takeCardMoveWeight = -1;
            double buyBuildingWeight = -1;
            double rebuildWeight = -1;

            if (checker.IsAnyMoveAvailable(RepresentedPlayer, typeof(TakeCardMove)))
            {
                takeCardMoveWeight = GetSumOfWeights(CriteriaProperties.TakeCardMoveSelection, null);
                if (RepresentedPlayer.cards.Count > MaxCardsToTakeLimit)
                {
                    takeCardMoveWeight = 0;
                }
            }
            if (checker.IsAnyMoveAvailable(RepresentedPlayer, typeof(BuyBuildingMove)))
            {
                buyBuildingWeight = GetSumOfWeights(CriteriaProperties.BuyBuildingMoveSelection, null);
            }
            if (checker.IsAnyMoveAvailable(RepresentedPlayer, typeof(RebuildMove)))
            {
                rebuildWeight = GetSumOfWeights(CriteriaProperties.RebuildMoveSelection, null);
                if (rebuildInRow > MaxRebuildsInRowLimit)
                {
                    rebuildWeight = 0;
                }
            }

            // In every state of the game, there must be available move
            if ((takeCardMoveWeight == -1) &&
                (buyBuildingWeight == -1) &&
                (rebuildWeight == -1))
            {
                return null;
            }

            // If buy-building move has the biggest or equal value of weight, we'll choose it
            if ((buyBuildingWeight >= takeCardMoveWeight) &&
                (buyBuildingWeight >= rebuildWeight))
            {
                return typeof(BuyBuildingMove);
            }
            if ((takeCardMoveWeight >= buyBuildingWeight) &&
                (takeCardMoveWeight >= rebuildWeight))
            {
                return typeof(TakeCardMove);
            }
            if ((rebuildWeight >= buyBuildingWeight) &&
                (rebuildWeight >= takeCardMoveWeight))
            {
                return typeof(RebuildMove);
            }

            throw new AlhambraException("Logically-unreachable code reached.");
        }

        /// <summary>
        /// Computes a good selection of cards to take.
        /// </summary>
        /// <returns>List of cards that should be taken in Take-card move.</returns>
        private List<Card> GetCardsToTake()
        {
            List<Card> cardsOnMarket = new List<Card>();
            for (int i = 0; i < Game.MarketSize; i++)
            {
                if (!game.nonAvailableCardsOnMarket[i])
                {
                    cardsOnMarket.Add(game.cardsOnMarket[i]); // This card is available and could be taken
                }
            }
            var subsets = Controller.GetSubsets<Card>(cardsOnMarket);
            double max = double.MinValue;
            int indexOfMax = -1;
            for (int i = 0; i < subsets.Count; i++)
            {
                if (!checker.CanTakeCards(subsets[i]))
                {
                    continue;
                }
                double subsetWeight = GetSumOfWeights(CriteriaProperties.CardsToTakeSelection, subsets[i]);
                if (subsetWeight > max)
                {
                    indexOfMax = i;
                    max = subsetWeight;
                }
            }
            return subsets[indexOfMax];
        }

        /// <summary>
        /// Computes a good selection of building to buy.
        /// </summary>
        /// <param name="typeForPurchase">Parameter will be filed with a CardType needed to purchase
        /// selected building.</param>
        /// <returns>Building to be selected to buy.</returns>
        private Building GetBuildingToBuy(out CardType typeForPurchase)
        {
            double max = double.MinValue;
            int indexOfMax = -1;
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                CardType expectedType = game.cardTypesForBuy[i];
                if (!checker.CanBuyBuilding(RepresentedPlayer, onMarket, expectedType))
                {
                    continue;
                }
                double buildingWeight = GetSumOfWeights(CriteriaProperties.BuildingToBuyOrToAlhSelect, onMarket);
                if (buildingWeight > max)
                {
                    indexOfMax = i;
                    max = buildingWeight;
                }
            }

            // index of maximum is always non-negative number (player
            // can buy any of the buildings - it is checked before)
            typeForPurchase = game.cardTypesForBuy[indexOfMax];
            return game.buildingsOnMarket[indexOfMax];
        }

        /// <summary>
        /// Computes a good selection of cards to be purchased for a specified building.
        /// </summary>
        /// <param name="toBePurchased">Building that will be purhased.</param>
        /// <param name="expectedType">CardType that is used for cards to purchase the specified building.</param>
        /// <returns></returns>
        private List<Card> GetCardsToPurchaseBuilding(Building toBePurchased, CardType expectedType)
        {
            // Variable of 'AIWeighedMoves' class --- usage in methods for 
            // evaluating 'CriteriaProperties.CardsToPurchaseSelection'
            this.toBePurchased = toBePurchased;

            List<Card> correctColored = new List<Card>();
            foreach (Card c in RepresentedPlayer.cards)
            {
                if (c.Type == expectedType)
                {
                    correctColored.Add(c);
                }
            }

            // Get all subsets of cards of correct type
            var subsets = Controller.TryGetSubsets<Card>(correctColored);
            double max = double.MinValue;
            int indexOfMax = -1;
            for (int i = 0; i < subsets.Count; i++)
            {
                if (checker.IsPossibleToBuyUsingAny(toBePurchased, subsets[i], expectedType))
                {
                    // Get weight of subset 
                    double setWeight = GetSumOfWeights(CriteriaProperties.CardsToPurchaseSelection, subsets[i]);
                    if (setWeight > max)
                    {
                        indexOfMax = i;
                        max = setWeight;
                    }
                }
            }

            this.toBePurchased = null;

            return subsets[indexOfMax];
        }

        /// <summary>
        /// To building specified by the parameter will enter position for game map.
        /// </summary>
        /// <param name="building">Building to be positioned.</param>
        /// <param name="fromPreviousMove">List of buildings that have been modified in previous move.</param>
        private void SolveBuildingPosition(ref Building building, List<Building> fromPreviousMove)
        {
            if (building != null)
            {
                // Solving first this building (due rules, in rebuild move must be
                // 'rebuilding' building be placed first
                Position p = GetPositionForBuilding(building, fromPreviousMove, alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
                SetBuildingProperties(building, p);
            }
        }

        /// <summary>
        /// Sets a position of the specified building (used during rebuild move).
        /// </summary>
        /// <param name="rm">A building to have set position.</param>
        private void ResolvePositionsForRebuild(ref RebuildMove rm)
        {
            if (rm.modified.Count == 0)
            {
                return;
            }

            Building[,] mapOrignal = (Building[,])game.map[RepresentedPlayer.ID].Clone();

            if (rm.ToStorage != null)
            {
                int row = rm.ToStorage.Position.Row;
                int col = rm.ToStorage.Position.Column;

                // Building to storage is removed from game map (simulating move)
                game.map[RepresentedPlayer.ID][row, col] = null;
            }
            if (rm.ToAlhambra != null)
            {
                // Get positions for the rest of the buildins, while "rebuilded" building is
                // simulated in the game map
                int row = rm.ToAlhambra.Position.Row;
                int col = rm.ToAlhambra.Position.Column;

                game.map[RepresentedPlayer.ID][row, col] = rm.ToAlhambra;
            }


            List<Building> alreadyPositioned = new List<Building>();
            for (int i = 0; i < rm.modified.Count; i++)
            {
                Position p = GetPositionForBuilding(rm.modified[i], alreadyPositioned, alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
                SetBuildingProperties(rm.modified[i], p);
                alreadyPositioned.Add(rm.modified[i]);
            }

            // returning back to game map old building
            game.map[RepresentedPlayer.ID] = mapOrignal;
        }

        /// <summary>
        /// Computes a good position for the specified building by parameter.
        /// </summary>
        /// <param name="toBePositioned">Sets buildings position.</param>
        /// <param name="alreadyPositioned">List of already positioned buildings from previous move or moves.</param>
        /// <param name="alsoGenerPosInStorage">Determines whether the function will generate position for storage (Position.None).</param>
        /// <param name="alsoGenerPosForImag">Determines whether the function will generace position for imaginary player (Position.Imaginary).</param>
        /// <returns>Position for the specified building.</returns>
        public Position GetPositionForBuilding(Building toBePositioned, List<Building> alreadyPositioned, bool alsoGenerPosInStorage, bool alsoGenerPosForImag)
        {
            int top, left, bottom, right;
            GetNewBoundsOfMap(out top, out left, out bottom, out right);

            Building temporaryCopy = (Building)toBePositioned.Clone();
            List<Position> availablePositions = new List<Position>();
            double max = double.MinValue;
            int indexOfMax = -1;
            if (alsoGenerPosInStorage)
            {
                availablePositions.Add(Position.None); // Also generate position "in storage"
                _toBePlaced = temporaryCopy;
                max = GetSumOfWeights(CriteriaProperties.BuildingPositionSelection, Position.None);
                _toBePlaced = null;
                indexOfMax = 0;
            }

            // left, right, top and bottom now contains a new bounds;
            // Based on these bounds, positions for buildings will be generated
            for (int row = top; row <= bottom; row++)
            {
                for (int col = left; col <= right; col++)
                {
                    Position p = new Position(row, col);
                    temporaryCopy.Position = p;
                    temporaryCopy.IsInStorage = false;
                    alreadyPositioned.Add(temporaryCopy);
                    bool isPermissible = checker.CanAddBuildingMoreBuildings(RepresentedPlayer.ID, alreadyPositioned);
                    alreadyPositioned.Remove(temporaryCopy);
                    if (!isPermissible)
                    {
                        continue;
                    }
                    availablePositions.Add(p);

                    // '_alreadyPositionedDueMove' is used in some
                    // of 'CriteriaProperties.BuildingPositionSelection' methods
                    _alreadyPositionedDueMove = alreadyPositioned;

                    _toBePlaced = temporaryCopy;
                    double positionWeight = GetSumOfWeights(CriteriaProperties.BuildingPositionSelection, p);
                    _toBePlaced = null;

                    _alreadyPositionedDueMove = null;

                    if (positionWeight >= max)
                    {
                        indexOfMax = availablePositions.IndexOf(p);
                        max = positionWeight;
                    }
                }
            }

            if (alsoGenerPosForImag)
            {
                availablePositions.Add(Position.Imaginary);
                _toBePlaced = temporaryCopy;
                double weight = GetSumOfWeights(CriteriaProperties.BuildingPositionSelection, Position.Imaginary);
                _toBePlaced = null;
                if (weight > max)
                {
                    max = weight;
                    indexOfMax = availablePositions.IndexOf(Position.Imaginary);
                }
            }


            // This Building can't be added to Alhambra
            if (indexOfMax == -1)
            {
                // Caller should check whether method returns Position.None,
                // in case that indexOfMax == -1, Position.None means "there's no available position for this building",
                // (also depends on parameter 'alsoGenerPosInStorage'.
                return Position.None;
            }
            return availablePositions[indexOfMax];
        }

        /// <summary>
        /// Fill parameters with the new bounds of rectangle created by player's 'topLeft' and 'bottomRight' positions.
        /// </summary>
        private void GetNewBoundsOfMap(out int top, out int left, out int bottom, out int right)
        {
            // We are not generating all positions - we use the nearest
            // positions next to already built buildings in player map.
            // Top left and bottom right positions for the specific player 
            // will be increased by number of buildings to be added and it's 1.
            int increase = 1;
            top = game.topLeft[RepresentedPlayer.ID].Row - increase;
            if (top < 0)
            {
                top = 0; // We were out of the game bounds...
            }

            left = game.topLeft[RepresentedPlayer.ID].Column - increase;
            if (left < 0)
            {
                left = 0;
            }

            bottom = game.bottomRight[RepresentedPlayer.ID].Row + increase;
            if (bottom >= Game.MapSize)
            {
                bottom = Game.MapSize - 1;
            }

            right = game.bottomRight[RepresentedPlayer.ID].Column + increase;
            if (right >= Game.MapSize)
            {
                right = Game.MapSize - 1;
            }
        }

        /// <summary>
        /// Sets building position to the specific position. If position is None (building
        /// is postponed), InStorage property will also be set.
        /// </summary>
        /// <param name="building">A building to be positioned.</param>
        /// <param name="positionToSet">A position for the building.</param>
        private void SetBuildingProperties(Building building, Position positionToSet)
        {
            building.Position = positionToSet;
            if (positionToSet == Position.None)
            {
                building.IsInStorage = true;
            }
            else
            {
                building.IsInStorage = false;
            }
        }

        /// <summary>
        /// Based on a quality, select which Rebuild move type will be selected.
        /// </summary>
        /// <returns>'Alhambra.TypeOfRebuildMove' selection.</returns>
        private TypeOfRebuildMove GetTypeOfRebuildMove()
        {
            double toAlhambraWeight = -1;
            double toStorageWeight = -1;
            double swapWeight = -1;
            if (CanMakeToAlhambraRebuild())
            {
                toAlhambraWeight = GetSumOfWeights(CriteriaProperties.ToAlhambraRebuildSelection, null);
            }
            if (CanMakeSwapRebuild())
            {
                swapWeight = GetSumOfWeights(CriteriaProperties.SwapRebuildSelection, null);
            }
            if (CanMakeToStorageRebuild())
            {
                toStorageWeight = GetSumOfWeights(CriteriaProperties.ToStorageRebuildSelection, null);
            }


            // For safety, check availibility of Rebuild move
            if ((toAlhambraWeight == -1) && (toStorageWeight == -1) && (swapWeight == -1))
            {
                throw new AlhambraException("There's no available Rebuild move within the game.");
            }

            // Returning the most valuable type of Rebuild move
            if ((toAlhambraWeight >= toStorageWeight) &&
                (toAlhambraWeight >= swapWeight))
            {
                return TypeOfRebuildMove.ToAlhambra;
            }

            if ((swapWeight >= toAlhambraWeight) &&
              (swapWeight >= toStorageWeight))
            {
                return TypeOfRebuildMove.Swap;
            }

            if ((toStorageWeight >= toAlhambraWeight) &&
                (toStorageWeight >= swapWeight))
            {
                return TypeOfRebuildMove.ToStorage;
            }



            throw new AlhambraException("Type of rebuild move can't be choosen.");
        }

        /// <summary>
        /// Checks if is possible to make a Rebuild move with adding building to Alhambra (only).
        /// </summary>
        /// <returns>True if is possible to make a Rebuild move with adding building to Alhambra.</returns>
        internal bool CanMakeToAlhambraRebuild()
        {
            foreach (Building b in RepresentedPlayer.postponed)
            {
                int top, left, bottom, right;
                GetNewBoundsOfMap(out top, out left, out bottom, out right);

                // Loop over sub-rectangle of map and check if player can add building 
                for (int row = top; row <= bottom; row++)
                {
                    for (int col = left; col <= right; col++)
                    {
                        Building copy = (Building)b.Clone();
                        copy.Position = new Position(row, col);
                        copy.IsInStorage = false;

                        bool result = checker.CanAddBuilding(RepresentedPlayer.ID, copy);
                        if (result)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if is possible to make a Rebuild move with adding building from Alhambra to storage (only).
        /// </summary>
        /// <returns>True if is possible to make a Rebuild move with adding building from Alhambra to storage.</returns>
        internal bool CanMakeToStorageRebuild()
        {
            foreach (Building b in RepresentedPlayer.constructed)
            {
                bool result = checker.CanRemoveBuilding(RepresentedPlayer.ID, b);
                if (result)
                {
                    return true; // player can remove building from map => can make rebuild - to storage
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if is possible to make a Rebuild move with swapping two buildings.
        /// </summary>
        /// <returns>True, if is possible to make a Rebuild move with swapping two buildings.</returns>
        internal bool CanMakeSwapRebuild()
        {
            foreach (Building toStorage in RepresentedPlayer.constructed)
            {
                foreach (Building toAlhambra in RepresentedPlayer.postponed)
                {
                    if (checker.CanSwapBuildings(RepresentedPlayer.ID, toAlhambra, toStorage))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Computes a good selecting of building which will be placed to Alhambra from storage. This method
        /// also computes a position for the building.
        /// </summary>
        /// <returns>Instance of Building which will be placed to Alhambra.</returns>
        private Building GetBuildingToAlhambra()
        {
            double max = double.MinValue;
            int indexOfMax = -1;
            Position bestPosition = Position.None;
            for (int i = 0; i < RepresentedPlayer.postponed.Count; i++)
            {
                Building postponed = RepresentedPlayer.postponed[i];

                // Generating position regardless to buildings from previous move
                // -- they will be repositioned if necessary
                Position goodPositionForBuilding = GetPositionForBuilding(postponed, new List<Building>(), alsoGenerPosInStorage: false, alsoGenerPosForImag: false);
                //Position goodPositionForBuilding = GetPositionForBuilding(postponed, game.unresolved, alsoGenerPosInStorage: false);
                if (goodPositionForBuilding == Position.None)
                {
                    // We wanted position in Alhambra (and we got only in storage)
                    // => there's no available position in Alhambra for this building
                    continue;
                }
                double buildingWeight = GetSumOfWeights(CriteriaProperties.BuildingToBuyOrToAlhSelect, postponed);
                if (buildingWeight > max)
                {
                    max = buildingWeight;
                    indexOfMax = i;
                    bestPosition = goodPositionForBuilding;
                }
            }

            Building result = (Building)RepresentedPlayer.postponed[indexOfMax].Clone();
            SetBuildingProperties(result, bestPosition);

            // Returning the most valuable building with the most valuable position
            return result;
        }

        /// <summary>
        /// Computes a good selecting of building which will be placed from Alhambra to storage. This method
        /// also computes a position for the building.
        /// </summary>
        /// <returns>Instance of Building which will be placed to Alhambra.</returns>
        private Building GetBuildingToStorage()
        {
            double max = double.MinValue;
            int indexOfMax = -1;
            for (int i = 0; i < RepresentedPlayer.constructed.Count; i++)
            {
                Building toBePostponed = (Building)RepresentedPlayer.constructed[i].Clone();
                if (!checker.CanRemoveBuilding(RepresentedPlayer.ID, toBePostponed))
                {
                    continue;
                }
                double buildingWeight = GetSumOfWeights(CriteriaProperties.BuildingToStorageSelection, toBePostponed);
                if (buildingWeight > max)
                {
                    max = buildingWeight;
                    indexOfMax = i;
                }
            }
            Building result = RepresentedPlayer.constructed[indexOfMax];
            return result;
        }

        /// <summary>
        /// Computes which buildings will be selected for Rebuild - swap - move.
        /// </summary>
        /// <param name="toAlhambra">Parameter will be filled with a building to Alhambra in Rebuild move.</param>
        /// <param name="toStorage">Parameter will be filled with a building to Storage in Rebuild move.</param>
        private void GetBuildingForSwap(out Building toAlhambra, out Building toStorage)
        {
            toAlhambra = null;
            toStorage = null;
            double max = double.MinValue;
            foreach (Building _toStorage in RepresentedPlayer.constructed)
            {
                foreach (Building _toAlhambra in RepresentedPlayer.postponed)
                {
                    if (!checker.CanSwapBuildings(RepresentedPlayer.ID, _toAlhambra, _toStorage))
                    {
                        continue;
                    }
                    var t = new Tuple<Building, Building>(_toAlhambra, _toStorage);
                    double swapWeight = GetSumOfWeights(CriteriaProperties.SwapSelection, t);
                    if (swapWeight > max)
                    {
                        max = swapWeight;
                        toAlhambra = _toAlhambra;
                        toStorage = _toStorage;
                    }
                }
            }
        }

        #endregion

        //
        //
        //

        /// <summary>
        /// Initializes methods for evaluatin criteria.
        /// </summary>
        private void InitializeMethods()
        {
            methods = new Func<object, bool>[NumberOfMethods];

            int offset = 0;

            #region CriteriaProperties.TakeCardMoveSelection

            methods[offset++] = new Func<object, bool>(IsTakeCardMove);
            methods[offset++] = new Func<object, bool>(CanTakeEnoughCards);
            methods[offset++] = new Func<object, bool>(CanTakeEnoughCardsForExact);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_1);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_2);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_3);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_4);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_5);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_6);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_7);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_8);
            methods[offset++] = new Func<object, bool>(CanTakeCardOfValue_9);

            #endregion

            #region CriteriaProperties.BuyBuildingMoveSelection

            methods[offset++] = new Func<object, bool>(IsBuyBuildingMove);
            methods[offset++] = new Func<object, bool>(CanBuyBuildingExactly);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_Blue);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_Red);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_Brown);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_White);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_Green);
            methods[offset++] = new Func<object, bool>(CanBuyBuilding_Violet);

            #endregion

            #region CriteriaProperties.RebuildMoveSelection

            methods[offset++] = new Func<object, bool>(IsRebuildMove);

            // Reusing indices for 'HasCompletelyBoundedAlhambra' and 'HasPostponedBuilding' methods.
            // offset == 21
            methods[offset++] = new Func<object, bool>(HasCompletelyBoundedAlhambra);
            // offset == 22 
            methods[offset++] = new Func<object, bool>(HasPostponedBuilding);

            #endregion

            #region CriteriaProperties.CardsToTakeSelection

            methods[offset++] = new Func<object, bool>(TakingCardOfValue_1);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_2);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_3);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_4);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_5);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_6);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_7);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_8);
            methods[offset++] = new Func<object, bool>(TakingCardOfValue_9);
            methods[offset++] = new Func<object, bool>(TakingEnoughCards);
            methods[offset++] = new Func<object, bool>(TakingEnoughCardsForExact);
            methods[offset++] = new Func<object, bool>(TakingColorOfLargestSum);

            #endregion

            #region CriteriaProperties.BuildingToBuyOrToAlhSelect

            methods[offset++] = new Func<object, bool>(BuyingBlueBuilding);
            methods[offset++] = new Func<object, bool>(BuyingRedBuilding);
            methods[offset++] = new Func<object, bool>(BuyingBrownBuilding);
            methods[offset++] = new Func<object, bool>(BuyingWhiteBuilding);
            methods[offset++] = new Func<object, bool>(BuyingGreenBuilding);
            methods[offset++] = new Func<object, bool>(BuyingVioletBuilding);
            methods[offset++] = new Func<object, bool>(CanMakeSameCount);
            methods[offset++] = new Func<object, bool>(CanIncreaseLeadToOne);
            methods[offset++] = new Func<object, bool>(CanIncreaseLeadToTwo);

            #endregion

            #region CriteriaProperties.CardsToPurchaseSelection

            methods[offset++] = new Func<object, bool>(UsingForPurchase_OneCard);
            methods[offset++] = new Func<object, bool>(UsingForPurchase_TwoCards);
            methods[offset++] = new Func<object, bool>(UsingForPurchase_ThreeCards);
            methods[offset++] = new Func<object, bool>(UsingForPurchase_MoreCards);
            methods[offset++] = new Func<object, bool>(OverPaidBy1);
            methods[offset++] = new Func<object, bool>(OverPaidBy2);
            methods[offset++] = new Func<object, bool>(OverPaidBy3);
            methods[offset++] = new Func<object, bool>(OverPaidBy4);
            methods[offset++] = new Func<object, bool>(OverPaidBy5);
            methods[offset++] = new Func<object, bool>(OverPaidByMore);
            methods[offset++] = new Func<object, bool>(NotOverPaid);

            #endregion

            #region CriteriaProperties.BuildingPositionSelection

            methods[offset++] = new Func<object, bool>(IncreaseWallBy_1);
            methods[offset++] = new Func<object, bool>(IncreaseWallBy_2);
            methods[offset++] = new Func<object, bool>(IncreaseWallBy_3);
            methods[offset++] = new Func<object, bool>(IncreaseWallBy_4);
            methods[offset++] = new Func<object, bool>(IncreaseWallBy_More);
            methods[offset++] = new Func<object, bool>(AlhambraWillNotBeComplBounded);
            methods[offset++] = new Func<object, bool>(GiveToImaginaryAndMakeSameCount);

            #endregion

            #region CriteriaProperties.ToAlhambraRebuildSelection

            // Reusing HasPostponedBuilding method

            #endregion

            #region CriteriaProperties.ToStorageRebuildSelection

            // Reusing HasCompletelyBoundedAlhambra method

            #endregion

            #region CriteriaProperties.SwapRebuildSelection

            methods[offset++] = new Func<object, bool>(HasBoundedAndHasPostponed);

            #endregion

            #region CriteriaProperties.BuildingToStorageSelection

            methods[offset++] = new Func<object, bool>(DecreaseLongestWallBy_0);
            methods[offset++] = new Func<object, bool>(DecreaseLongestWallBy_1);
            methods[offset++] = new Func<object, bool>(DecreaseLongestWallBy_2);
            methods[offset++] = new Func<object, bool>(DecreaseLongestWallBy_3);
            methods[offset++] = new Func<object, bool>(DecreaseLongestWallBy_More);
            methods[offset++] = new Func<object, bool>(BuildingHasWalls_0);
            methods[offset++] = new Func<object, bool>(BuildingHasWalls_1);
            methods[offset++] = new Func<object, bool>(BuildingHasWalls_2);
            methods[offset++] = new Func<object, bool>(BuildingHasWalls_3);

            #endregion

            #region CriteriaProperties.SwapSelection

            methods[offset++] = new Func<object, bool>(SwapIncreaseWallBy_1);
            methods[offset++] = new Func<object, bool>(SwapIncreaseWallBy_2);
            methods[offset++] = new Func<object, bool>(SwapIncreaseWallBy_3);
            methods[offset++] = new Func<object, bool>(SwapIncreaseWallBy_More);

            #endregion
        }

        #region Methods for 'CriteriaProperties.TakeCardMoveSelection'

        private bool IsTakeCardMove(object param)
        {
            return true;
        }

        private bool CanTakeEnoughCards(object param)
        {
            bool isOverPaid;
            return CanTakeEnoughCards_Internal(out isOverPaid);
        }

        private bool CanTakeEnoughCardsForExact(object param)
        {
            bool isOverPaid;
            bool result = CanTakeEnoughCards_Internal(out isOverPaid);
            return (result && !isOverPaid);
        }

        private bool CanTakeCardOfValue_1(object param)
        {
            return CanTakeCardOfValue_Internal(value: 1);
        }

        private bool CanTakeCardOfValue_2(object param)
        {
            return CanTakeCardOfValue_Internal(value: 2);
        }

        private bool CanTakeCardOfValue_3(object param)
        {
            return CanTakeCardOfValue_Internal(value: 3);
        }

        private bool CanTakeCardOfValue_4(object param)
        {
            return CanTakeCardOfValue_Internal(value: 4);
        }

        private bool CanTakeCardOfValue_5(object param)
        {
            return CanTakeCardOfValue_Internal(value: 5);
        }

        private bool CanTakeCardOfValue_6(object param)
        {
            return CanTakeCardOfValue_Internal(value: 6);
        }

        private bool CanTakeCardOfValue_7(object param)
        {
            return CanTakeCardOfValue_Internal(value: 7);
        }

        private bool CanTakeCardOfValue_8(object param)
        {
            return CanTakeCardOfValue_Internal(value: 8);
        }

        private bool CanTakeCardOfValue_9(object param)
        {
            return CanTakeCardOfValue_Internal(value: 9);
        }

        #endregion

        #region Methods for 'CriteriaProperties.BuyBuildingMoveSelection'

        private bool IsBuyBuildingMove(object param)
        {
            return true;
        }

        private bool CanBuyBuildingExactly(object param)
        {
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                CardType expectedType = game.cardTypesForBuy[i];
                if (checker.IsPossibleToBuyExactUsingAny(onMarket, RepresentedPlayer.cards, expectedType))
                {
                    return true;
                }
            }
            return false;
        }

        private bool CanBuyBuilding_Blue(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.Blue);
        }

        private bool CanBuyBuilding_Red(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.Red);
        }

        private bool CanBuyBuilding_Brown(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.Brown);
        }

        private bool CanBuyBuilding_White(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.White);
        }

        private bool CanBuyBuilding_Green(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.Green);
        }

        private bool CanBuyBuilding_Violet(object param)
        {
            return CanBuyBuilding_Internal(BuildingType.Violet);
        }

        #endregion

        #region Methods for 'CriteriaProperties.RebuildMoveSelection'

        private bool IsRebuildMove(object param)
        {
            return true;
        }

        private bool HasCompletelyBoundedAlhambra(object toBePlaced)
        {
            return HasComplBoundedAlh_Internal((Building)toBePlaced);
        }

        private bool HasPostponedBuilding(object param)
        {
            return (RepresentedPlayer.postponed.Count > 0);
        }

        #endregion

        #region Methods for 'CriteriaProperties.CardsToTakeSelection'

        const int NumberOfCardValues = 9;
        // Only one time initialized and represents if card of specified value (=index) will be taken
        bool[] takingCard = new bool[NumberOfCardValues];

        private bool TakingCardOfValue_1(object listOfCards)
        {
            for (int i = 0; i < takingCard.Length; i++)
            {
                takingCard[i] = false;
            }
            List<Card> subset = ((List<Card>)listOfCards);
            foreach (Card c in subset)
            {
                takingCard[c.Value - 1] = true;
            }
            return takingCard[0];
        }

        private bool TakingCardOfValue_2(object listOfCards)
        {
            return takingCard[1];
        }

        private bool TakingCardOfValue_3(object listOfCards)
        {
            return takingCard[2];
        }

        private bool TakingCardOfValue_4(object listOfCards)
        {
            return takingCard[3];
        }

        private bool TakingCardOfValue_5(object listOfCards)
        {
            return takingCard[4];
        }

        private bool TakingCardOfValue_6(object listOfCards)
        {
            return takingCard[5];
        }

        private bool TakingCardOfValue_7(object listOfCards)
        {
            return takingCard[6];
        }

        private bool TakingCardOfValue_8(object listOfCards)
        {
            return takingCard[7];
        }

        private bool TakingCardOfValue_9(object listOfCards)
        {
            return takingCard[8];
        }

        private bool TakingEnoughCards(object listOfCards)
        {
            return CanBuyNewBuildingAfter((List<Card>)listOfCards);
        }

        private bool TakingEnoughCardsForExact(object listOfCards)
        {
            return CanBuyNewBldAfterExact((List<Card>)listOfCards);
        }

        /// <summary>
        /// Computes the sum of values of each card type in player owned cards. Determines if
        /// some of cards specified by parameter are of this the most often type.
        /// </summary>
        /// <param name="listOfCards">Must has type of List&lt;Card&gt;. Represents list of cards to take.</param>
        /// <returns>True if any of cards specified by parameter has same type as type with
        /// the largest sum of values.</returns>
        private bool TakingColorOfLargestSum(object listOfCards)
        {
            if (RepresentedPlayer.cards.Count == 0)
            {
                return true;
            }

            const int NumberOfTypes = 4;
            Dictionary<CardType, int> sums = new Dictionary<CardType, int>(NumberOfTypes);
            foreach (CardType ct in Enum.GetValues(typeof(CardType)))
            {
                sums.Add(ct, 0);
            }
            foreach (Card c in RepresentedPlayer.cards)
            {
                sums[c.Type] += c.Value;
            }
            List<CardType> mostFrequent = new List<CardType>();
            int max = -1;
            foreach (int sum in sums.Values)
            {
                if (sum > max)
                {
                    max = sum;
                }
            }
            foreach (CardType ct in sums.Keys)
            {
                if (sums[ct] == max)
                {
                    mostFrequent.Add(ct);
                }
            }

            foreach (Card c in ((List<Card>)listOfCards))
            {
                if (mostFrequent.Contains(c.Type))
                {
                    return true;
                }
            }
            return false;
        }

        #endregion

        #region Methods for 'CriteriaProperties.BuildingToBuyOrToAlhSelect'

        private bool BuyingBlueBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.Blue);
        }

        private bool BuyingRedBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.Red);
        }

        private bool BuyingBrownBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.Brown);
        }

        private bool BuyingWhiteBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.White);
        }

        private bool BuyingGreenBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.Green);
        }

        private bool BuyingVioletBuilding(object building)
        {
            return HasColor((Building)building, BuildingType.Violet);
        }

        /// <summary>
        /// Evaluates whether the player can make same amount of buildings of any color (against to other player) 
        /// by taking building specified by the parameter. The player with more buildings of the specified color
        /// must be the first, the second, or the third in amount of these buildings against another player.
        /// </summary>
        /// <param name="building">Must be type of Building.</param>
        /// <returns>True if the player can make same amount of buildings of any color.</returns>
        internal bool CanMakeSameCount(object building)
        {
            return HasDifference((Building)building, differenceSize: 1);
        }

        internal bool CanIncreaseLeadToOne(object building)
        {
            return HasDifference((Building)building, differenceSize: 0);
        }

        internal bool CanIncreaseLeadToTwo(object building)
        {
            return HasDifference((Building)building, differenceSize: -1);
        }

        #endregion

        #region Methods for 'CriteriaProperties.CardsToPurchaseSelection'

        // Using these members for more methods (for better perfomance; don't want to recompute
        // same results more times...)
        int numberOfCardsToTake = 0; // Member is set in first call of 'UsingForPurchase_OneCard' method.
        int overPaidBy = 0; // Member is set in first call of 'OverPaidBy1' method.
        Building toBePurchased; // Building to be purchased in next move

        private bool UsingForPurchase_OneCard(object listOfCards)
        {
            numberOfCardsToTake = ((List<Card>)listOfCards).Count;
            return numberOfCardsToTake == 1;
        }

        private bool UsingForPurchase_TwoCards(object listOfCards)
        {
            return numberOfCardsToTake == 2;
        }

        private bool UsingForPurchase_ThreeCards(object listOfCards)
        {
            return numberOfCardsToTake == 3;
        }

        private bool UsingForPurchase_MoreCards(object listOfCards)
        {
            return numberOfCardsToTake > 3;
        }

        private bool OverPaidBy1(object listOfCards)
        {
            // Only first time calling 'OverPaidBy_Internal'
            overPaidBy = OverPaidBy_Internal((List<Card>)listOfCards);

            return overPaidBy == 1;
        }

        private bool OverPaidBy2(object listOfCards)
        {
            return overPaidBy == 2;
        }

        private bool OverPaidBy3(object listOfCards)
        {
            return overPaidBy == 3;
        }

        private bool OverPaidBy4(object listOfCards)
        {
            return overPaidBy == 4;
        }

        private bool OverPaidBy5(object listOfCards)
        {
            return overPaidBy == 5;
        }

        private bool OverPaidByMore(object listOfCards)
        {
            return overPaidBy > 5;
        }

        private bool NotOverPaid(object listOfCards)
        {
            return overPaidBy == 0;
        }

        #endregion

        #region Methods for 'CriteriaProperties.BuildingPositionSelection'

        /// <summary>
        /// Used in 'IncreaseWallBy' methods.
        /// </summary>
        Building _toBePlaced;
        int _increasedBy;

        private bool IncreaseWallBy_1(object position)
        {
            Building clone = (Building)_toBePlaced.Clone();
            clone.Position = (Position)position;
            _increasedBy = FindChangeOfWallLength(clone, isRemoving: false);

            return _increasedBy == 1;
        }

        private bool IncreaseWallBy_2(object position)
        {
            return _increasedBy == 2;
        }

        private bool IncreaseWallBy_3(object position)
        {
            return _increasedBy == 3;
        }

        private bool IncreaseWallBy_4(object position)
        {
            return _increasedBy == 4;
        }

        private bool IncreaseWallBy_More(object position)
        {
            return _increasedBy > 4;
        }

        /// <summary>
        /// Evaluates whether the player will have completely bounded Alhambra
        /// after adding building to specified position. 
        /// </summary>
        /// <param name="position">A position where the building will be placed</param>
        /// <returns>True if after placing the specified building player gets
        /// completely bounded Alhambra with walls.</returns>
        private bool AlhambraWillNotBeComplBounded(object position)
        {
            Position p = (Position)position;

            // Building goes to storage...
            if ((p == Position.None) || (p == Position.Imaginary))
            {
                return !HasComplBoundedAlh_Internal(null);
            }
            else
            {
                _toBePlaced.Position = p;
                return !HasComplBoundedAlh_Internal(_toBePlaced);
            }
        }

        /// <summary>
        /// Returns true if player can give a building to imaginary player and the count of buildings
        /// (of the actual type) will be the same as different player's and final scoring round is near.
        /// </summary>
        private bool GiveToImaginaryAndMakeSameCount(object position)
        {
            Position p = (Position)position;
            if ((p != Position.Imaginary) || (game.GameMode != GameMode.WithImaginary))
            {
                return false;
            }

            // We want only end of the game
            if (game.package.Count > 1)
            {
                return false;
            }
            BuildingType type = _toBePlaced.Type;
            int imaginary = game.GetNumberOfSpecifiedBuilding(Game.IndexOfImaginary, type);
            for (int ID = 0; ID < game.NumberOfPlayers; ID++)
            {
                Player player = game.Controller.players[ID];
                if (player == RepresentedPlayer)
                {
                    continue;
                }
                int count = game.GetNumberOfSpecifiedBuilding(ID, type);
                if ((count == imaginary) || (imaginary == count - 1))
                {
                    return true;
                }
            }
            return false;
        }

        #endregion

        #region Methods for 'CriteriaProperties.ToAlhambraRebuildSelection'

        // HasPostponedBuilding is already implemented in #region Methods for 'CriteriaProperties.RebuildMoveSelection'

        #endregion

        #region Methods for 'CriteriaProperties.ToStorageRebuildSelection'

        // HasCompletelyBoundedAlhambra is already implemented in #region Methods for 'CriteriaProperties.RebuildMoveSelection'

        #endregion

        #region Methods for 'CriteriaProperties.SwapRebuildSelection'

        private bool HasBoundedAndHasPostponed(object param)
        {
            bool bounded = HasComplBoundedAlh_Internal(null);
            bool hasPostponed = (RepresentedPlayer.postponed.Count > 0);
            return bounded && hasPostponed;
        }

        #endregion

        #region Methods for 'CriteriaProperties.BuildingToStorageSelection'

        int _decreasedBy;

        private bool DecreaseLongestWallBy_0(object toBePostponed)
        {
            _decreasedBy = FindChangeOfWallLength((Building)toBePostponed, isRemoving: true);

            return _decreasedBy == 0;
        }

        private bool DecreaseLongestWallBy_1(object buildingToRemove)
        {
            return _decreasedBy == 1;
        }

        private bool DecreaseLongestWallBy_2(object buildingToRemove)
        {
            return _decreasedBy == 2;
        }

        private bool DecreaseLongestWallBy_3(object buildingToRemove)
        {
            return _decreasedBy == 3;
        }

        private bool DecreaseLongestWallBy_More(object buildingToRemove)
        {
            return _decreasedBy > 3;
        }

        private bool BuildingHasWalls_0(object buildingToRemove)
        {
            return GetNumberOfWallsOfBuilding((Building)buildingToRemove) == 0;
        }

        private bool BuildingHasWalls_1(object buildingToRemove)
        {
            return GetNumberOfWallsOfBuilding((Building)buildingToRemove) == 1;
        }


        private bool BuildingHasWalls_2(object buildingToRemove)
        {
            return GetNumberOfWallsOfBuilding((Building)buildingToRemove) == 2;
        }


        private bool BuildingHasWalls_3(object buildingToRemove)
        {
            return GetNumberOfWallsOfBuilding((Building)buildingToRemove) == 3;
        }


        #endregion

        #region Methods for 'CriteriaProperties.SwapSelection'

        int _increaseSwapWall;

        private bool SwapIncreaseWallBy_1(object buildinsTuple)
        {
            var t = (Tuple<Building, Building>)buildinsTuple;
            Building toAlh = t.Item1;
            // t.Item2 is building to Storage
            _increaseSwapWall = FindChangeOfWallLength(toAlh, isRemoving: false);

            return _increaseSwapWall == 1;
        }

        private bool SwapIncreaseWallBy_2(object buildinsTuple)
        {
            return _increaseSwapWall == 2;
        }

        private bool SwapIncreaseWallBy_3(object buildinsTuple)
        {
            return _increaseSwapWall == 3;
        }

        private bool SwapIncreaseWallBy_More(object buildinsTuple)
        {
            return _increaseSwapWall > 3;
        }

        #endregion

        //
        //
        //

        #region Internal implementation methods

        /// <summary>
        /// Returns true if the current player can take enough cards to purchase a new building;
        /// otherwise false (if player has already been albe to buy the specified building
        /// before taking any card, also false!).
        /// </summary>
        /// <param name="isOverPaid">True if player can take enough cards to purchase a new
        /// building and building is not overpaid.</param>
        /// <returns>True if the current player can take enough cards to purchase a new
        /// building.</returns>
        internal bool CanTakeEnoughCards_Internal(out bool isOverPaid)
        {
            isOverPaid = false;
            List<Card> onMarket = new List<Card>(game.cardsOnMarket);
            var subsets = Controller.GetSubsets<Card>(onMarket);

            for (int i = 0; i < subsets.Count; i++)
            {
                List<Card> set = subsets[i];
                if (!checker.CanTakeCards(set))
                {
                    subsets.Remove(set);
                    i--;
                }
            }

            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building b = game.buildingsOnMarket[i];
                if (b == null)
                {
                    continue;
                }
                CardType expectedType = game.cardTypesForBuy[i];
                if (checker.IsPossibleToBuyUsingAny(b, RepresentedPlayer.cards, expectedType))
                {
                    // Player can already buy this building - take some new
                    // cards for the reason of this building is disadvantage
                    continue;
                }
                foreach (List<Card> set in subsets)
                {
                    List<Card> newSetOfCards = new List<Card>(RepresentedPlayer.cards);
                    newSetOfCards.AddRange(set);
                    if (checker.IsPossibleToBuyUsingAny(b, newSetOfCards, expectedType))
                    {
                        isOverPaid = !checker.IsPossibleToBuyExactUsingAny(b, newSetOfCards, expectedType);
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Determines whether the current player can take card of the specified value.
        /// </summary>
        /// <param name="value">Value of the card to be taken.</param>
        /// <returns>True if the current player can take card of the specified value.</returns>
        internal bool CanTakeCardOfValue_Internal(int value)
        {
            foreach (Card c in game.cardsOnMarket)
            {
                if (c.Value == value)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Determines whether the current player can purchase a new building of the specified BuildingType.
        /// </summary>
        /// <param name="color">A BuildingType to be taken.</param>
        /// <returns>True if the current player can purchase a new building of the specified BuildingType.</returns>
        internal bool CanBuyBuilding_Internal(BuildingType color)
        {
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                CardType expectedType = game.cardTypesForBuy[i];
                if ((onMarket == null) || (onMarket.Type != color))
                {
                    continue;
                }
                if (checker.IsPossibleToBuyUsingAny(onMarket, RepresentedPlayer.cards, expectedType))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// True if player has completely bounded Alhambra with walls.
        /// </summary>
        /// <param name="toBeAdded">A building to be placed in Alhambra, or null.</param>
        /// <returns>True if player has completely bounded Alhambra with walls.</returns>
        internal bool HasComplBoundedAlh_Internal(Building toBeAdded)
        {
            if (toBeAdded != null)
            {
                int row = toBeAdded.Position.Row;
                int col = toBeAdded.Position.Column;
                if (game.map[RepresentedPlayer.ID][row, col] != null)
                {
                    throw new AlhambraException("Two buildings at same position...");
                }
                else
                {
                    game.map[RepresentedPlayer.ID][row, col] = toBeAdded;
                    game.constructed[RepresentedPlayer.ID].Add(toBeAdded);
                }
            }

            int count = game.constructed[RepresentedPlayer.ID].Count;
            Position[] neighbors = new Position[MoveChecker.NumberOfNeighbors];

            bool result = true;
            for (int i = count - 1; i >= 0; i--)
            {
                Building b = game.constructed[RepresentedPlayer.ID][i];
                Position neighbor1, neighbor2, neighbor3, neighbor4;
                MoveChecker mc = new MoveChecker(game);
                int row1 = b.Position.Row;
                int col1 = b.Position.Column;
                mc.GetSurroundingPositions(row1, col1, out neighbor1, out neighbor2, out neighbor3, out neighbor4);
                neighbors[0] = neighbor1;
                neighbors[1] = neighbor2;
                neighbors[2] = neighbor3;
                neighbors[3] = neighbor4;
                foreach (Position p in neighbors)
                {
                    int row2 = p.Row;
                    int col2 = p.Column;
                    if (game.map[RepresentedPlayer.ID][row2, col2] != null)
                    {
                        continue; // The specific neighbor is not empty
                    }
                    int index = -1;
                    if (row1 < row2)
                    {
                        index = 0;
                    }
                    if (row1 > row2)
                    {
                        index = 2;
                    }
                    if (col1 < col2)
                    {
                        index = 3;
                    }
                    if (col1 > col2)
                    {
                        index = 1;
                    }

                    // The specified building hasn't wall at the side to the empty neighbor
                    // So Alhambra could not be fully bounded
                    if (b.Walls[(index + 2) % 4] == false)
                    {
                        result = false;
                    }

                    if (!result)
                    {
                        break;
                    }
                }
                if (!result)
                {
                    break;
                }
            }

            // Need to clean up the game map
            if (toBeAdded != null)
            {
                int row = toBeAdded.Position.Row;
                int col = toBeAdded.Position.Column;
                game.map[RepresentedPlayer.ID][row, col] = null;
                game.constructed[RepresentedPlayer.ID].Remove(toBeAdded);
            }
            return result;
        }

        /// <summary>
        /// Evaluates whether player will be available to buy new building after taking
        /// cards specified by parameter. Returns false if all buildings which can
        /// be bought by player are available to buy before taking the specified set of cards.
        /// </summary>
        /// <param name="cardsToTake">Cards to take.</param>
        /// <returns>True if player can buy new building after taking the specified set of cards.</returns>
        internal bool CanBuyNewBuildingAfter(List<Card> cardsToTake)
        {
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                if (onMarket == null)
                {
                    continue;
                }

                CardType expected = game.cardTypesForBuy[i];
                if (checker.IsPossibleToBuyUsingAny(onMarket, RepresentedPlayer.cards, expected))
                {
                    continue; // Player already can buy this building (don't need to take other cards...)
                }

                List<Card> newCards = new List<Card>(cardsToTake);
                newCards.AddRange(RepresentedPlayer.cards);
                if (checker.IsPossibleToBuyUsingAny(onMarket, newCards, expected))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Evaluates whether player will be available to buy new building after taking
        /// cards specified by parameter, without over paid. Returns false,
        /// if all buildings which can be bought by player (without over paid) are available 
        /// to buy before taking the specified set of cards.
        /// </summary>
        /// <param name="cardsToTake">Cards to take.</param>
        /// <returns>True if after taking the specified cards player will be available to purchase a new building without over paid.</returns>
        internal bool CanBuyNewBldAfterExact(List<Card> cardsToTake)
        {
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                if (onMarket == null)
                {
                    continue;
                }

                CardType expected = game.cardTypesForBuy[i];
                bool result = checker.IsPossibleToBuyExactUsingAny(onMarket, RepresentedPlayer.cards, expected);
                if (result)
                {
                    continue; // Player already can buy this building without over paid (so don't need to take another cards...)
                }

                List<Card> newCards = new List<Card>(cardsToTake);
                newCards.AddRange(RepresentedPlayer.cards);
                if (checker.IsPossibleToBuyExactUsingAny(onMarket, newCards, expected))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Determines if the specified building has a correct BuildingType.
        /// </summary>
        /// <returns>True if the specified building has same type as the specified BuildingType.</returns>
        private bool HasColor(Building building, BuildingType bt)
        {
            return building.Type == bt;
        }

        /// <summary>
        /// Evaluates if difference in a number of buildings of specified color is exactly the same as a 'differenceSize' parameter.
        /// </summary>
        /// <param name="toBuy">Evaluating the number of specified buildings by type of this building.</param>
        /// <param name="differenceSize">Determines the difference in a number of specified buildings.</param>
        /// <returns>True, if some player has a number of buildings of a certain type specified by parameters.</returns>
        private bool HasDifference(Building toBuy, int differenceSize)
        {
            int myCount = game.GetNumberOfSpecifiedBuilding(RepresentedPlayer.ID, toBuy.Type);
            int[] countOfPlayersBlds = new int[game.NumberOfPlayers - 1]; // Do not include "myself"
            int index = 0;
            for (int i = 0; i < game.NumberOfPlayers; i++)
            {
                if (i == RepresentedPlayer.ID)
                {
                    continue;
                }
                countOfPlayersBlds[index++] = game.GetNumberOfSpecifiedBuilding(i, toBuy.Type);
            }

            Array.Sort(countOfPlayersBlds);

            const int Points = 3; // Maximum order of the player that he can get points (points can get the first, the second and the third)
            int alreadyChecked = 0;
            for (int i = countOfPlayersBlds.Length - 1; i >= 0; i--)
            {
                if (countOfPlayersBlds[i] == (differenceSize + myCount))
                {
                    return true;
                }
                if (++alreadyChecked == Points)
                {
                    break; // Don't want to compare between player at the end of order
                }
            }
            return false;
        }

        /// <summary>
        /// Evaluates how much is 'toBePurchased' building over paid.
        /// </summary>
        /// <param name="cards">Cards used for the purchase of the building.</param>
        /// <returns>Value indicating how much is the specific building over paid.</returns>
        private int OverPaidBy_Internal(List<Card> cards)
        {
            int sum = 0;
            foreach (Card c in cards)
            {
                sum += c.Value;
            }
            return sum - toBePurchased.Value;
        }

        /// <summary>
        /// Evaluates a new length of the longest wall around the player's Alhambra, using
        /// the building specified by the parameter. Based of isRemoving parameter method will compute
        /// the longest wall after adding the building or after removing the building.
        /// </summary>
        /// <param name="toPlaceOrRemove">Based on this building will be evaluated a new length of the 
        /// longest wall around the Alhambra.</param>
        /// <param name="isRemoving">If true, method evaluates the longest wall around player's Alhambra after removing
        /// the building instead off adding it.</param>
        /// <returns>A length change of the longest wall around the Alhambra 
        /// after placing or removing the specified building.</returns>
        internal int FindChangeOfWallLength(Building toPlaceOrRemove, bool isRemoving)
        {
            if (toPlaceOrRemove.Position == Position.None ||
                toPlaceOrRemove.Position == Position.Imaginary)
            {
                return 0;
            }

            int length = game.GetLongestWall(RepresentedPlayer.ID);
            List<Building> allModified = new List<Building>(game.unresolved);

            // Trying to add or remove the building in the map

            if (!isRemoving)
            {
                allModified.Add(toPlaceOrRemove);
            }

            for (int i = 0; i < allModified.Count; i++)
            {
                if (allModified[i].IsInStorage)
                {
                    allModified.RemoveAt(i); // Buildings in storage does not increase longest wall
                    i--;
                }
            }

            // Saving reference for original list with constructed buildings of the player
            List<Building> originalBuildings = new List<Building>();
            foreach (Building b in game.constructed[RepresentedPlayer.ID])
            {
                originalBuildings.Add((Building)b.Clone());
            }


            // Checking the longest wall after removing the building from the map
            if (isRemoving)
            {
                game.constructed[RepresentedPlayer.ID].Remove(toPlaceOrRemove);
            }
            foreach (Building b in allModified)
            {
                game.constructed[RepresentedPlayer.ID].Add(b);
            }

            // Compute length of the longest wall (it is based only on game.constructed)
            int newLength = game.GetLongestWall(RepresentedPlayer.ID);

            game.constructed[RepresentedPlayer.ID] = originalBuildings;
            RepresentedPlayer.constructed = originalBuildings;

            if (isRemoving)
            {
                return length - newLength;
            }
            else
            {
                return newLength - length;
            }
        }

        /// <summary>
        /// Returns a number of walls around the specified building.
        /// </summary>
        /// <param name="building">A building which walls will be detected.</param>
        /// <returns>A number of walls around the specified building.</returns>
        private int GetNumberOfWallsOfBuilding(Building building)
        {
            int result = 0;
            foreach (bool wall in building.Walls)
            {
                if (wall)
                {
                    result++;
                }
            }
            return result;
        }

        #endregion

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return "Artificial Intelligence with weighed moves.";
        }
    }
}

