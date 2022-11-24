import torch


class SafeSet:
    def __init__(self, Xleft, Xright, V, DiscrX=0.12) -> None:
        self.Xleft = Xleft
        self.Xright = Xright
        self.V = V
        self.DiscrX = DiscrX
        self.StateInSet = self.__compute_states_in_limits()
        self.idx = 0

    @classmethod
    def from_init_set(cls, InitSet, V, DiscrX) -> None:
        Xleft = InitSet.min()
        Xright = InitSet.max()
        return cls(Xleft, Xright, V, DiscrX)

    def Update(self, Xleft, Xright):
        self.Xleft = Xleft
        self.Xright = Xright
        # self.StateInSet = torch.where((self.V >= self.Xleft) & (
        #     self.V <= self.Xright), True, False)
        self.StateInSet = self.__compute_states_in_limits()

    def UpdateByTensor(self, Set):
        loc_of_set = self.V[Set]
        self.Xleft = loc_of_set[0]
        self.Xright = loc_of_set[-1]
        self.StateInSet = Set
        # self.StateInSet = torch.where((self.V >= self.Xleft) & (
        #     self.V <= self.Xright), True, False)
        self.StateInSet = self.__compute_states_in_limits()

    def __compute_states_in_limits(self):
        left_condn = torch.logical_or(
            self.V > self.Xleft, torch.isclose(self.V, self.Xleft)
        )
        right_condn = torch.logical_or(
            self.V < self.Xright, torch.isclose(self.V, self.Xright)
        )
        return torch.logical_and(left_condn, right_condn)

    # return torch.where((self.V >= self.Xleft) & (
    #     self.V <= self.Xright), True, False)
    # N = int(torch.abs((self.Xright - self.Xleft)/self.DiscrX) + 1 + 1e-4)
    # return torch.linspace(self.Xleft, self.Xright, N)

    # def Xleft(self):
    #     return self.Xleft

    # def Xright(self):
    #     return self.Xright

    # def IDleft(self):
    #     return

    # def IDright(self):
    #     return


if __name__ == "__main__":
    V = torch.linspace(-2.0, 10.0, 101).reshape((-1, 1))
    X_train = torch.Tensor([6.40, 6.52, 6.64]).reshape(-1, 1)

    SafePessi = SafeSet.from_init_set(X_train, V, 0.12)
    print(SafePessi.StateInSet)
