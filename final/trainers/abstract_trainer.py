from abc import ABC, abstractmethod

class Trainer(ABC):
  @abstractmethod
  def train(self):
    """ 実際の訓練を行う部分。 """
    pass


  @abstractmethod
  def evaluate(self):
    """ 一定間隔で方策評価を行う。 """
    pass


  @abstractmethod
  def plot(self):
    """ evaluateで評価を行う際に記録した平均リターンのプロットを行う。 """
    pass