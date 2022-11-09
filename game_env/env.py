from logging import raiseExceptions
from random import randint
from collections import Counter

class CardPool:
    def __init__(self, n_deck: int):
        """initialize the card pool with total of n_deck * 52 cards
        the 13-length cardLeft array can be used as the state
        """
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
        self.cardLeft = [n_deck * 4] * 13
        self.cardCounter = Counter({i: n_deck * 4 for i in self.idxs.values()})

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
    def __init__(self, inital_money=1000, bets_options):
        self.money = 1000
        self.options =['Hit','Double','Stand']
        self.bets_options = bets_options

        self.hands=[] #current cards
        self.hands_sum=[] #current possibilities of card sums

    def reset_round(self):
        '''reset the hands and bet
        '''
        self.hands=[]
        self.hands_sum=[0]
        self.bet = None
        return

    def choose_bet(self,cardpool,betmodel):
        '''reset the round, and choose the best bet
        '''
        self.bet = betmodel(self.money,cardpool.cardLeft)
        self.money-=self.bet
        return

    def start_round(self,cardpool):
        '''start the round, get two cards from the pool
        '''
        newhand = cardpool.pick()
        self.update_sum(newhand)
        newhand = cardpool.pick()
        self.update_sum(newhand)

    def update_sum(self, hand :str):
        '''update the hands sum, return all possibilities
        for example, hands = ['A', '5'] -> hands_sum = [6,16]
        '''
        if hand=='A':
            self.hands_sum = [ i+1 for i in self.hands_sum]
            self.hands_sum.append(self.hands_sum[-1]+10)
        elif hand in ['J','Q','K']:
            self.hands_sum = [i+10 for i in self.hands_sum]
        else:
            self.hands_sum =[i+ int(hand) for i in self.hands_sum]

    def make_action(self,dealer,cardpool,actionmodel):
        '''get the action model prediction -> p,v
        then make the best action
        '''
        predictions=[] # predictions for all possibility of hands sum

        for sum in self.hands_sum:
            # current state to be determined
            # [sum, revealed_card, cardLeft] 15x1 vector
            # or [action (3x1 vector), sum,revealed_card, cardLeft] 18x1 vector
            states = [sum]+ [dealer.revealed_card] + cardpool.cardLeft
            p,v = actionmodel(states)
            predictions.append((p,v))

        return predictions
        ## in progress: return and make the best action
        return 

