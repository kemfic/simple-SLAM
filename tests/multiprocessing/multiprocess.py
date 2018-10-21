import multiprocessing as multi
from multiprocessing import Queue, Process

if __name__ == '__main__':
  print("Cores: ", multi.cpu_count())
