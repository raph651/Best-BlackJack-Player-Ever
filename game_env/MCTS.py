import env


class MCT:
    def __init__(self):
        self.states = {}

    class node:
        def __init__(self, state, parent, child=None):
            self.state = state
            self.parent = parent
            self.child = child
            self.action_count = [0]*3

    def get_child(self, root, state):
        if state in self.states:
            root.child = self.states[state]
        else:
            root.child = self.node(state, root)
            self.states[state] = root.child

    def roll_out(self, root, state):
        for action in env.Player().options:
            print(action)

def simulate():
