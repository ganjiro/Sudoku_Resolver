import random
import itertools


def LoadRandomSolvedSudoku():
    f = open("SolvedSudoku.txt", "r")
    for i in range(1, random.randint(0, 9999)):
        f.readline()
    sudokuLine = f.readline()
    sudoku = []
    for i in range(0, 9):
        sudoku.append(list(sudokuLine[9 * i:9 * (i + 1)]))
    return sudoku


def CreateSudoku(solvedSudoku):
    nCells = random.randint(8, 20)
    for i in range(1, 81 - nCells):
        solvedSudoku[random.randint(0, 8)][random.randint(0, 8)] = 0
    return solvedSudoku


def ArcConsistency(puzzle):
    queue = [[i, j] for i, j in itertools.product(range(0, 9), range(0, 9))]
    while queue:
        Revise(queue.pop)


def Revise(index):
    revised = False


class Sudoku:
    def __init__(self, puzzle):
        Sudoku.puzzle = puzzle
        Sudoku.costraints = [[[i, j], ([i, k] for k in range(0, 9))] for i, j in
                             itertools.product(range(0, 9), range(0, 9))]
        Sudoku.domains = [[range(1, 10) for i in range(0, 9)] for j in range(0, 9)]
        # sudoku.domains = arcConsistency()


def CreateCostraints(self):
    costraints = []
    costraints += [[[i, j], [i, k]] for i, j, k in itertools.product(range(0, 9), range(0, 9), range(0, 9))]
    costraints += [[[i, j], [k, j]] for i, j, k in itertools.product(range(0, 9), range(0, 9), range(0, 9))]
    for x, y in itertools.product(range(0, 9, 3), range(0, 9, 3)):
        costraints += [[[i, j], [k, l]] for i, j, k, l in
                       itertools.product(range(x, x + 4), range(y, y + 4), range(x, x + 4), range(y, y + 4))]

    costraints.sort()
    costraints = list(costraints for costraints, _ in itertools.groupby(costraints))
    for i, j in zip(range(0, 9), range(0, 9)):
        costraints.remove([[i, j], [i, j]])
    return costraints


print(CreateCostraints(0))