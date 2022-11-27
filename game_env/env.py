from logging import raiseExceptions
from random import randint
from collections import Counter


class CardPool:
    def __init__(self, n_deck: int):
        """initialize the card pool with total of n_deck * 52 cards
        the 13-length cardLeft array can be used as the state
        """
        self.n_deck = n_deck
        self.N = n_deck * 52
        self.mapping = {}  # blacklist mapping
        self.idxs = {
            1: "A",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
            11: "J",
            12: "Q",
            13: "K",
        }
        self.cardMapping = lambda x: self.idxs[x % (13) + 1]
        ## for input state, rest of the card counts
        self.cardLeft = [n_deck * 4] * 13
        self.cardCounter = Counter({i: n_deck * 4 for i in self.idxs.values()})

    def reset(self):
        self.cardLeft = [self.n_deck * 4] * 13
        self.mapping = {}
        self.N = self.n_deck * 52

    def pick(self) -> str:
        """return a random card from the card pool in O(1) time"""
        i = randint(0, self.N - 1)
        sample = self.mapping.get(i, i)
        if sample != self.N - 1:
            if self.N - 1 not in self.mapping:
                self.mapping[i] = self.N - 1
            else:
                self.mapping[i] = self.mapping[self.N - 1]
                del self.mapping[self.N - 1]
        self.N -= 1
        card = self.cardMapping(sample)

        # update cardLeft and cardCounter
        self.cardLeft[sample % (13)] -= 1
        self.cardCounter[card] -= 1

        return card

    def test(self):
        """randomly pick 100 cards for test
        record the picked cards and check the number of each card in the pool
        """
        pickN = 100
        print("=" * 40)
        print(f"Testing for random pick {pickN} cards")
        cardpicked = []
        for _ in range(pickN):
            cardpicked.append(self.pick())
        pickedCounter = Counter(cardpicked)
        print(
            "picked counter: \n",
            [(card, pickedCounter[card]) for card in self.idxs.values()],
        )
        print(
            "card pool counter: \n",
            [(card, self.cardCounter[card]) for card in self.idxs.values()],
        )
        print("card left array: \n", self.cardLeft)
        result = self.test_count()
        print("test result :", result)
        print("=" * 40)

    def test_count(self):
        """test if the number of each kind of card match the card Counter"""
        testCounter = Counter(
            [self.cardMapping(self.mapping.get(i, i)) for i in range(0, self.N)]
        )
        # check if match the self.cardCounter dict
        if any([testCounter[i] != self.cardCounter[i] for i in self.idxs.values()]):
            return False
        # check if match the self.cardLeft state array
        if any([testCounter[self.idxs[i + 1]] != self.cardLeft[i] for i in range(13)]):
            return False
        return True


class Player:
    def __init__(
        self, initial_money=1000,
    ):
        self.money = initial_money
        self.actions = ["Hit", "Double", "Stand"]
        self.double = False

        self.hands = []  # current cards
        self.hands_sum = [0]  # current possibilities of card sums

    def reset_round(self):
        """reset the hands and bet
        """
        self.hands = []
        self.hands_sum = [0]
        self.bet = None
        self.double = False
        return

    def get_actions(self):
        if self.double:
            return self.actions[:1] + self.actions[2:]
        return self.actions

    def random_actions(self):
        return self.actions[randint(0, len(self.actions) - 1)]

    def predict_bet(self, cardpool, betmodel):
        """reset the round, and let neural network to predict the best bet
        """
        self.bet = betmodel(self.money, cardpool.cardLeft)
        self.money -= self.bet
        return self.bet

    def make_bet(self, bet):
        """given a player's choice on bet, proceed
        """
        self.bet = bet
        self.money -= bet

    def choose_bet(self):
        """prompt the player to make a bet choice
        """
        bet = round(
            float(
                input(
                    f"how much do you want to bet, in percentages based on current money {self.money}: ?% "
                )
            )
            * self.money
            / 100
        )
        self.make_bet(bet)
        print(f"you bet {bet} $")
        return bet

    def start_round(self, cardpool):
        """start the round, get two cards from the pool
        """
        newhand = cardpool.pick()
        self.update_hand(newhand)
        newhand = cardpool.pick()
        self.update_hand(newhand)

    def deal_one_card(self, cardpool):
        newhand = cardpool.pick()
        self.update_hand(newhand)
        return newhand

    def update_hand(self, hand: str):
        """update the hands sum, return all possibilities
        for example, hands = ['A', '5'] -> hands_sum = [6,16]
        """
        self.hands.append(hand)
        if hand == "A":
            self.hands_sum = [i + 1 for i in self.hands_sum]
            self.hands_sum.append(self.hands_sum[-1] + 10)
        elif hand in ["J", "Q", "K"]:
            self.hands_sum = [i + 10 for i in self.hands_sum]
        else:
            self.hands_sum = [i + int(hand) for i in self.hands_sum]

        if len(self.hands_sum) >= 3:
            self.hands_sum.pop()

    def best_hand(self):
        """return the best valid hand_sum that is less than 21 and closest to 21
        """
        if len(self.hands_sum) == 1 or self.hands_sum[1] > 21:
            best_hands_sum = self.hands_sum[0]
        else:
            best_hands_sum = self.hands_sum[1]
        return best_hands_sum

    def predict_actions(self, dealer, cardpool, net):
        """get the action model prediction: net(s,a) -> v
        then make the best action
        """
        predictions = {}  # predictions for all possibility of hands sum

        for sum in self.hands_sum:
            # [action (3x1 vector), sum,revealed_card, cardLeft] 18x1 vector
            states = [sum] + [dealer.revealed_card] + cardpool.cardLeft
            for action in self.get_actions():
                pred = net(states, action)
                predictions[(sum, action)] = pred

        return predictions

        ## in progress: return and make the best action

        return


class Dealer:
    def __init__(self, threshold=17):
        self.revealed = None
        self.hands = []
        self.hands_sum = [0]
        self.money = 0
        self.threshold = threshold

    def reset_round(self):
        self.revealed = None
        self.hands = []
        self.hands_sum = [0]

    def start_round(self, cardpool):
        """start the round, get two cards from the pool, reveal the first
        """
        newhand = cardpool.pick()
        self.update_hand(newhand)

        # make this first card revealed
        self.revealed = newhand
    
    def deal_one_card(self, cardpool):
        newhand = cardpool.pick()
        self.update_hand(newhand)
        return newhand

    def update_hand(self, hand: str):
        """update the hands sum, return all possibilities
        for example, hands = ['A', '5'] -> hands_sum = [6,16]
        """
        self.hands.append(hand)
        if hand == "A":
            self.hands_sum = [i + 1 for i in self.hands_sum]
            self.hands_sum.append(self.hands_sum[-1] + 10)
        elif hand in ["J", "Q", "K"]:
            self.hands_sum = [i + 10 for i in self.hands_sum]
        else:
            self.hands_sum = [i + int(hand) for i in self.hands_sum]

    def make_action(self, cardpool):
        """default policy, dealer stands on any sum of 17 or greater, and hits otherwise
        """
        while self.hands_sum[-1] < 17:
            newhand = cardpool.pick()
            self.update_hand(newhand)

    def best_hand(self):
        """return the dealer's hand sum
        """
        if len(self.hands_sum) == 1 or self.hands_sum[1] > 21:
            best_hands_sum = self.hands_sum[0]
        else:
            best_hands_sum = self.hands_sum[1]
        return best_hands_sum


class BlackJackState:
    def __init__(self, cardleft, revealed, hands_sum, turn, earned, bet, lost):
        self.cardleft = cardleft
        self.revealed = revealed
        self.hands_sum = hands_sum
        # the current turn
        self.turn = turn
        # money earned so far
        self.earned = earned
        # the current bet, if double it becomes 2 times big
        self.bet = bet
        # whether the game has lost: 1 -player wins, -1 -dealer wins, 0 - draw
        self.lost = lost

    def copy(self):
        return BlackJackState(
            self.cardleft.copy(),
            self.revealed,
            self.hands_sum.copy(),
            self.turn,
            self.earned,
            self.bet,
            self.lost,
        )


class BlackJackEnv:
    def __init__(self, n_deck=6, player_num=1):

        self.cardpool = CardPool(n_deck)
        if player_num > 1:
            print("multiple players in progress, player_num is set to 1")
        """if multiple players:
            in progress
        """
        self.player = Player(initial_money=1000)
        self.dealer = Dealer()

        self.state = None
        self.earned_current_turn = 0

    def reset_env(self):
        """only reset the environment, call new_round to start a new round and return the new state
        """
        self.cardpool.reset()
        self.dealer.reset_round()
        self.player.reset_round()
        self.state = None

    def new_round(self):
        """reset and then start a new round, prompt the player to make a bet if interactive,
        otherwise, use player.make_bet(bet) for a specific bet
        """
        self.dealer.reset_round()
        self.player.reset_round()
        # deal one card to the dealer, and reveal the card
        # deal two cards to the player
        self.dealer.start_round(self.cardpool)
        self.player.start_round(self.cardpool)

        revealed = self.dealer.revealed
        hands_sum = self.player.hands_sum
        turn = 0
        earned = 0
        bet = self.player.choose_bet()
        # bet = self.player.make_bet(bet)  if not interactive
        lost = None
        self.state = BlackJackState(
            self.cardpool.cardLeft, revealed, hands_sum, turn, earned, bet, lost
        )
        return self.state.copy()

    def get_actions(self):
        return self.player.get_actions()

    def render(self):
        if self.state is not None:
            print("\nThe cardpool has ")
            for card, left in self.cardpool.cardCounter.items():
                print(f"{card}: {left}", end=" ," + "\n" * (card == "7"))
            print(f"\nThe dealer has one revealed card : {self.dealer.revealed}")
            print(f"\nCurrent player has hands: {self.player.hands}")
        else:
            print("use env.new_round to start a new round")

    def check_game_end(self):
        return

    def step(self, action):
        self.state.turn += 1
        self.earned_current_turn = 0

        if action == "Stand":
            # if action is stand, the dealer picks cards based on the default policy
            self.dealer.make_action(self.cardpool)
            dealer_hand_sum = self.dealer.best_hand()
            player_hand_sum = self.player.best_hand()

            # if dealer bust, the player wins because player doesn't bust
            if dealer_hand_sum > 21:
                self.state.lost = 1
                self.earned_current_turn = self.state.bet * 2
            else:
                if dealer_hand_sum == player_hand_sum:
                    self.state.lost = 0
                elif dealer_hand_sum > player_hand_sum:
                    self.state.lost = -1
                    self.earned_current_turn = self.state.bet * -1
                else:
                    self.state.lost = 1
                    self.earned_current_turn = self.state.bet * 2
                    self.player.money += self.earned_current_turn
            self.state.earned+=self.earned_current_turn
            return self.state, self._get_reward(), True, {}

        if action == "Double":
            self.player.double = True
            self.player.money-=self.state.bet
            self.state.bet *= 2

        self.player.deal_one_card(self.cardpool)
        self.state.hands_sum = self.player.hands_sum.copy()

        # if all the hands_sum has bigger than 21 values, then the player loses and round ends
        # otherwise, continue the game
        if all([n > 21 for n in self.player.hands_sum]):
            self.state.lost = -1
            self.earned_current_turn = self.state.bet * -1
            self.dealer.deal_one_card(self.cardpool)
            self.state.earned+=self.earned_current_turn
            
            return self.state.copy(), self._get_reward(), True, {}

        return self.state.copy(), self._get_reward(), False, {}

    def _get_reward(self):
        return

