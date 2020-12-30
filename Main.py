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
    queue = puzzle.costraints
    while queue:
        index = queue.pop(0)
        if Revise(index,puzzle):
            if not puzzle.domains[index[0][0]][index[0][1]]:
                return False
            newarcs = [[[index[0][0], index[0][1]], [index[0][0], k]] for k in range(0, 9)]
            newarcs += [[[index[0][0], index[0][1]], [k, index[0][1]]] for k in range(0, 9)]
            x = index[0][0] - index[0][0] % 3
            y = index[0][1] - index[0][1] % 3
            newarcs += [[[index[0][0], index[0][1]], [k, l]] for k, l in
                        itertools.product(range(x, x + 3), range(y, y + 3))]
            newarcs.sort()
            newarcs = list(newarcs for newarcs, _ in itertools.groupby(newarcs))
            for i, j in itertools.product(range(0, 9), range(0, 9)):
                if [[i, j], [i, j]] in newarcs: newarcs.remove([[i, j], [i, j]])
            newarcs.remove([[index[0][0], index[0][1]], [index[1][0], index[1][1]]])
            queue += newarcs


def Revise(index,puzzle):
    revised = False
    print(index)
    for x in puzzle.domains[index[0][0]][index[0][1]]:
        if x == puzzle.variables[index[1][0]][index[1][1]]:
            puzzle.domains[index[0][0]][index[0][1]].remove(x)
            revised = True
    return revised


class Sudoku:
    def __init__(self):
        self.variables = None
        self.costraints = self.CreateCostraints()
        print(self.costraints)
        self.domains = [[[1,2,3,4,5,6,7,8,9] for _ in range(0, 9)] for _ in range(0, 9)]

    def LoadSudoku(self, puzzle):
        self.variables = puzzle
        ArcConsistency(self)

    def isComplete(self):
        result = True


    def CreateCostraints(self):
        costraints = []
        costraints += [[[i, j], [i, k]] for i, j, k in itertools.product(range(0, 9), range(0, 9), range(0, 9))]
        costraints += [[[i, j], [k, j]] for i, j, k in itertools.product(range(0, 9), range(0, 9), range(0, 9))]
        for x, y in itertools.product(range(0, 9, 3), range(0, 9, 3)):
            costraints += [[[i, j], [k, l]] for i, j, k, l in
                           itertools.product(range(x, x + 3), range(y, y + 3), range(x, x + 3), range(y, y + 3))]
        costraints.sort()
        costraints = list(costraints for costraints, _ in itertools.groupby(costraints))
        for i, j in itertools.product(range(0, 9), range(0, 9)):
            if [[i, j], [i, j]] in costraints: costraints.remove([[i, j], [i, j]])
        return costraints


a=[[0,0,3,0,2,0,6,0,0],[9,0,0,3,0,5,0,0,1],[0,0,1,8,0,6,4,0,0],[0,0,8,1,0,2,9,0,0],[7,0,0,0,0,0,0,0,8],[0,0,6,7,0,8,2,0,0],[0,0,2,6,0,9,5,0,0],[8,0,0,2,0,3,0,0,9],[0,0,5,0,1,0,3,0,0,]]
s = Sudoku()
s.LoadSudoku(a)
