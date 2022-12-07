import torch
from random import randint
from collections import Counter
from typing import List

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
        self.cardCounter = Counter({i: self.n_deck * 4 for i in self.idxs.values()})
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


class Player:
    def __init__(
            self, initial_money=1000,
    ):
        self.money = initial_money
        self.actions = [0, 1, 2]
        self.hands = []  # current cards
        self.hands_sum = [0]  # current possibilities of card sums

    def reset_round(self):
        """reset the hands and bet
        """
        self.hands = []
        self.hands_sum = [0]
        self.bet = None
        self.money = 1000
        return

    def get_actions(self):
        return self.actions

    '''def random_actions(self):
        return self.actions[randint(0, len(self.actions) - 1)]
    '''

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

    '''def choose_bet(self):
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
    '''

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

        self.hands_sum = [i for i in self.hands_sum if i <= 21]

class Dealer:
    def __init__(self, threshold=17):
        self.revealed = None
        self.hands = []
        self.hands_sum = [0]
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

        self.hands_sum = [i for i in self.hands_sum if i <= 21]

    def make_action(self, cardpool):
        """default policy, dealer stands on any sum of 17 or greater, and hits otherwise
        """
        while self.hands_sum and self.hands_sum[-1] < 17:
            newhand = cardpool.pick()
            self.update_hand(newhand)


class BlackJackState:
    def __init__(self, cardleft, revealed, hands_sum, turn, earned, bet, lost,bet_input):
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
        # bet_input variable
        self.bet_input = bet_input

    def copy(self):
        return BlackJackState(
            self.cardleft.copy(),
            self.revealed,
            self.hands_sum.copy(),
            self.turn,
            self.earned,
            self.bet,
            self.lost,
            self.bet_input,
        )

    def input(self) ->List:
        revealed = 1 if self.revealed == 'A' else (10 if self.revealed in ['J', 'Q', 'K'] else int(self.revealed))
        hands_sum = self.hands_sum.copy() * 2 if len(self.hands_sum) == 1 else self.hands_sum.copy()
        return self.cardleft + [revealed] + hands_sum


class BlackJackEnv:
    def __init__(self, n_deck=6, player_num=1,default_bet=True):

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
        self.default_bet = default_bet

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
        if self.cardpool.N < 12:
            self.cardpool.reset()

        self.dealer.reset_round()
        self.player.reset_round()

        # change bet_input
        bet_input = self.cardpool.cardLeft.copy()

        # deal one card to the dealer, and reveal the card
        # deal two cards to the player
        self.dealer.start_round(self.cardpool)
        self.player.start_round(self.cardpool)

        revealed = self.dealer.revealed
        hands_sum = self.player.hands_sum
        turn = 0 if not self.state else self.state.turn
        earned = 0 if not self.state else self.state.earned

        lost = None
        # bet set to be 1
        bet = 1
        self.state = BlackJackState(
            self.cardpool.cardLeft, revealed, hands_sum, turn, earned, bet, lost,bet_input
        )

    def get_actions(self):
        return self.player.get_actions()

    def render(self):
        if self.state is not None:
            print("\nThe cardpool has ")
            for card, left in self.cardpool.cardCounter.items():
                print(f"{card}: {left}", end=" ," + "\n" * (card == "7"))
            print(f"\nThe dealer has one revealed card : {self.dealer.revealed}")
            print(f"\nCurrent player has hands: {self.player.hands}")
            print(f"\nCurrent player has money: {self.player.money}")
        else:
            print("use env.new_round to start a new round")

    def step(self, action):
        '''return reward, done
        '''
        self.state.turn += 1
        self.earned_current_turn = 0

        if action == 1 or action == 2:
            # if the action is double, dealt one card
            double = 1
            if action == 1:
                double=2
                self.state.bet *= 2
                self.player.deal_one_card(self.cardpool)
                self.state.hands_sum = self.player.hands_sum.copy()

                # check if player bust
                if not self.state.hands_sum:
                    self.state.lost = -0.5
                    self.earned_current_turn = self.state.bet * -1
                    self.dealer.deal_one_card(self.cardpool)
                    self.state.earned += self.earned_current_turn
                    self.player.money += self.earned_current_turn

                    #self.history.append(self.state.lost)
                    reward = self._get_reward()
                    self.new_round()
                    return reward*double, True

            # then the dealer picks cards based on the default policy
            self.dealer.make_action(self.cardpool)

            # if dealer bust, the player wins because player doesn't bust
            if not self.dealer.hands_sum:
                self.state.lost = 0.5
                self.earned_current_turn = self.state.bet
            else:
                if self.dealer.hands_sum[-1] == self.player.hands_sum[-1]:
                    self.state.lost = 0
                elif self.dealer.hands_sum[-1] > self.player.hands_sum[-1]:
                    self.state.lost = -0.5
                    self.earned_current_turn = self.state.bet * -1
                else:
                    self.state.lost = 0.5
                    self.earned_current_turn = self.state.bet

            self.player.money += self.earned_current_turn
            self.state.earned += self.earned_current_turn

            #self.history.append(self.state.lost)
            reward = self._get_reward()
            self.new_round()
            return reward*double, True

        else:
            self.player.deal_one_card(self.cardpool)
            self.state.hands_sum = self.player.hands_sum.copy()

            # if all the hands_sum has bigger than 21 values, then the player loses and round ends
            # otherwise, continue the game
            if not self.state.hands_sum:
                self.state.lost = -0.5
                self.earned_current_turn = self.state.bet * -1
                self.dealer.deal_one_card(self.cardpool)

                self.state.earned += self.earned_current_turn
                self.player.money += self.earned_current_turn
                #self.history.append(self.state.lost)
                reward = self._get_reward()
                self.new_round()
                return reward, True

            return self._get_reward(), False

    def _get_reward(self):
        return self.state.lost