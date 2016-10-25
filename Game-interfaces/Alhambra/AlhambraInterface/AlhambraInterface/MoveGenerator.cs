using System;
using System.Collections.Generic;
using Alhambra;

namespace AlhambraInterface
{
    class MoveGenerator
    {
        /// <summary>
        /// Types of rebuild move - ToAlhambra, ToStorage, Swap.
        /// </summary>
        private enum TypeOfRebuildMove
        {
            ToAlhambra,
            ToStorage,
            Swap
        }

        public Player RepresentedPlayer
        {
            get;
            private set;
        }

        private Game game;
        private MoveChecker checker;

        public MoveGenerator(Player player)
        {
            RepresentedPlayer = player;
            game = player.game;
            checker = new MoveChecker(game);
        }


        /// <summary>
        /// Resused code from Alhambra. Generates possible moves.
        /// </summary>
        /// <returns>List of possible moves to make (in the current round).</returns>
        public List<Move> GenerateMoves()
        {
            List<Move> result = new List<Move>();

            foreach (List<Card> takenCards in GetCardsToTake())
            {
                Move move = null;
                TakeCardMove tcm = new TakeCardMove(RepresentedPlayer.ID);
                tcm.takenCards = takenCards;
                tcm.modified = new List<Building>(game.unresolved);
                move = tcm;
                result.Add(move);
            }

            foreach (var toBuyPair in GetBuildingToBuy())
            {
                Building toBuy = toBuyPair.Item1;
                CardType typeForPurchase = toBuyPair.Item2;
                foreach (List<Card> cardsToPurchase in GetCardsToPurchaseBuilding(toBuy, typeForPurchase))
                {
                    BuyBuildingMove bbm = new BuyBuildingMove(RepresentedPlayer.ID, toBuy, cardsToPurchase, typeForPurchase);
                    bbm.modified = new List<Building>(game.unresolved);
                    foreach (Building positionedToBuy in SolveBuildingPosition(toBuy, bbm.modified))
                    {
                        bbm.purchased = positionedToBuy;
                        result.Add(bbm);
                    }
                }
            }

            foreach (TypeOfRebuildMove type in Enum.GetValues(typeof(TypeOfRebuildMove)))
            {
                Building toAlhambra = null;
                Building toStorage = null;
                switch (type)
                {
                    case TypeOfRebuildMove.ToAlhambra:
                        if (!CanMakeToAlhambraRebuild())
                        {
                            continue;
                        }
                        toAlhambra = GetBuildingToAlhambra();
                        break;
                    case TypeOfRebuildMove.ToStorage:
                        if (!CanMakeToStorageRebuild())
                        {
                            continue;
                        }
                        toStorage = GetBuildingToStorage();
                        break;
                    case TypeOfRebuildMove.Swap:
                        if (!CanMakeSwapRebuild())
                        {
                            continue;
                        }
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
                result.AddRange(ResolvePositionsForRebuild(rm));
            }
            if (result.Count == 0)
            {
                result.Add(null);
            }
            return result;
        }

        /// <summary>
        /// Returns a position for a free building. This building can be obtained at the end of the game.
        /// </summary>
        /// <param name="toBeAccepted"></param>
        /// <returns>A position for the building obtained at the end of the game.</returns>
        public Position GetPositionForFreeBuilding(Building toBeAccepted)
        {
            return Position.None; // TODO
            //return GetPositionForBuilding(toBeAccepted, new List<Building>(), alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
        }

        /// <summary>
        /// Generates possible subsets of carts to take from the market.
        /// </summary>
        private List<List<Card>> GetCardsToTake()
        {
            var result = new List<List<Card>>();
            List<Card> cardsOnMarket = new List<Card>();
            for (int i = 0; i < Game.MarketSize; i++)
            {
                if (!game.nonAvailableCardsOnMarket[i])
                {
                    cardsOnMarket.Add(game.cardsOnMarket[i]); // This card is available and could be taken
                }
            }
            var subsets = Controller.GetSubsets<Card>(cardsOnMarket);
            for (int i = 0; i < subsets.Count; i++)
            {
                if (checker.CanTakeCards(subsets[i]))
                {
                    result.Add(subsets[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Gets list of buildings that can be bought. Tuples contain Building-CardType pairs.
        /// Card types are types, that are needed to buy this building.
        /// </summary>
        private List<Tuple<Building, CardType>> GetBuildingToBuy()
        {
            var result = new List<Tuple<Building, CardType>>();
            for (int i = 0; i < Game.MarketSize; i++)
            {
                Building onMarket = game.buildingsOnMarket[i];
                CardType expectedType = game.cardTypesForBuy[i];
                if (checker.CanBuyBuilding(RepresentedPlayer, onMarket, expectedType))
                {
                    result.Add(new Tuple<Building, CardType>(onMarket, expectedType));
                }
            }
            return result;
        }

        /// <summary>
        /// Computes possible selections of cards to be spent for specified building.
        /// </summary>
        /// <param name="toBePurchased">Building that will be purhased.</param>
        /// <param name="expectedType">CardType that is used for cards to purchase the specified building.</param>
        private List<List<Card>> GetCardsToPurchaseBuilding(Building toBePurchased, CardType expectedType)
        {
            var result = new List<List<Card>>();
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
            for (int i = 0; i < subsets.Count; i++)
            {
                if (checker.IsPossibleToBuyUsingAny(toBePurchased, subsets[i], expectedType))
                {
                    result.Add(subsets[i]);
                }
            }
            return result;
        }


        private List<Building> SolveBuildingPosition(Building building, List<Building> fromPreviousMove)
        {
            List<Building> result = new List<Building>();
            if (building != null)
            {
                // Solving first this building (due rules, in rebuild move must be
                // 'rebuilding' building be placed first
                var positions = GetPositionForBuilding(building, fromPreviousMove, alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
                foreach (Position p in positions)
                {
                    Building copy = (Building)building.Clone();
                    SetBuildingProperties(copy, p);
                    result.Add(copy);
                }
            }
            return result;
        }

        private List<RebuildMove> ResolvePositionsForRebuild(RebuildMove rm)
        {
            List<RebuildMove> result = new List<RebuildMove>();

            if (rm.modified.Count == 0)
            {
                result.Add(rm);
                return result;
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
                var positions = GetPositionForBuilding(rm.modified[i], alreadyPositioned, alsoGenerPosInStorage: true, alsoGenerPosForImag: true);
                if (i == rm.modified.Count - 1)
                {
                    foreach (Position p in positions)
                    {
                        RebuildMove copy = new RebuildMove(rm.PlayerID, rm.ToAlhambra, rm.ToStorage);
                        SetBuildingProperties(rm.modified[i], p);
                        foreach (var modifBuildng in rm.modified)
                        {
                            copy.modified.Add((Building)modifBuildng.Clone());
                        }
                        result.Add(copy);
                    }
                }
                else
                {
                    RebuildMove copy = new RebuildMove(rm.PlayerID, rm.ToAlhambra, rm.ToStorage);
                    SetBuildingProperties(rm.modified[i], positions[0]);
                    foreach (var modifBuildng in rm.modified)
                    {
                        copy.modified.Add((Building)modifBuildng.Clone());
                    }
                    alreadyPositioned.Add(rm.modified[i]);
                    result.Add(copy);
                }
            }

            // returning back to game map old building
            game.map[RepresentedPlayer.ID] = mapOrignal;

            return result;
        }

        /// <summary>
        /// Computes a possible positions for the specified building by parameter.
        /// </summary>
        /// <param name="toBePositioned">Sets buildings position.</param>
        /// <param name="alreadyPositioned">List of already positioned buildings from previous move or moves.</param>
        /// <param name="alsoGenerPosInStorage">Determines whether the function will generate position for storage (Position.None).</param>
        /// <param name="alsoGenerPosForImag">Determines whether the function will generace position for imaginary player (Position.Imaginary).</param>
        /// <returns>Positions for the specified building.</returns>
        public List<Position> GetPositionForBuilding(Building toBePositioned, List<Building> alreadyPositioned, bool alsoGenerPosInStorage, bool alsoGenerPosForImag)
        {
            int top, left, bottom, right;
            GetNewBoundsOfMap(out top, out left, out bottom, out right);

            Building temporaryCopy = (Building)toBePositioned.Clone();
            List<Position> availablePositions = new List<Position>();
            if (alsoGenerPosInStorage)
            {
                availablePositions.Add(Position.None); // Also generate position "in storage"
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
                }
            }

            //if (alsoGenerPosForImag)
            //{
            //    availablePositions.Add(Position.Imaginary);
            //}
            return availablePositions;
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
            for (int i = 0; i < RepresentedPlayer.postponed.Count; i++)
            {
                Building postponed = RepresentedPlayer.postponed[i];

                // Generating position regardless to buildings from previous move
                // -- they will be repositioned if necessary
                Position goodPositionForBuilding = GetPositionForBuilding(postponed, new List<Building>(), alsoGenerPosInStorage: false, alsoGenerPosForImag: false)[0]; // TODO !!!
                //Position goodPositionForBuilding = GetPositionForBuilding(postponed, game.unresolved, alsoGenerPosInStorage: false);
                if (goodPositionForBuilding == Position.None)
                {
                    // We wanted position in Alhambra (and we got only in storage)
                    // => there's no available position in Alhambra for this building
                    continue;
                }

                Building result = (Building)RepresentedPlayer.postponed[i].Clone();
                SetBuildingProperties(result, goodPositionForBuilding);

                // Returning the most valuable building with the most valuable position
                return result;
            }
            return null;
        }

        /// <summary>
        /// Computes a good selecting of building which will be placed from Alhambra to storage. This method
        /// also computes a position for the building.
        /// </summary>
        /// <returns>Instance of Building which will be placed to Alhambra.</returns>
        private Building GetBuildingToStorage()
        {
            List<Building> result = new List<Building>();
            for (int i = 0; i < RepresentedPlayer.constructed.Count; i++)
            {
                Building toBePostponed = (Building)RepresentedPlayer.constructed[i].Clone();
                if (!checker.CanRemoveBuilding(RepresentedPlayer.ID, toBePostponed))
                {
                    continue;
                }
                result.Add(RepresentedPlayer.constructed[i]);
            }
            return result[0]; // TODO
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
            foreach (Building _toStorage in RepresentedPlayer.constructed)
            {
                foreach (Building _toAlhambra in RepresentedPlayer.postponed)
                {
                    if (!checker.CanSwapBuildings(RepresentedPlayer.ID, _toAlhambra, _toStorage))
                    {
                        continue;
                    }
                    var t = new Tuple<Building, Building>(_toAlhambra, _toStorage);
                    toAlhambra = _toAlhambra;
                    toStorage = _toStorage;
                }
            }
        }
    }
}
