import copy
import math
import random
import itertools
from math import trunc
from timeit import Timer
from random import sample
import requests




def LoadRandomSolvedSudoku():
    f = open("SolvedSudoku.txt", "r")
    for i in range(1, random.randint(0, 9998)):
        f.readline()
    sudokuLine = f.readline()
    sudoku = []
    for i in range(0, 9):
        sudoku.append(list(map(int, sudokuLine[9 * i:9 * (i + 1)])))
    return sudoku


def CreateRandomSolvedSudoku():
    rows = [g * 3 + r for g in sample(range(3), 3) for r in sample(range(3), 3)]
    cols = [g * 3 + c for g in sample(range(3), 3) for c in sample(range(3), 3)]
    nums = sample(range(1, 10), 9)
    return [[nums[(3 * (r % 3) + r // 3 + c) % 9] for c in cols] for r in rows]


def RandomCellElimination(solvedSudoku):
    nCells = random.randint(55,65)
    return DeleteCell(solvedSudoku, nCells)


def ToThonkyNotation(sudoku):
    output = ''
    for i in range(0, 9):
        for j in range(0, 9):
            output += str(sudoku[i][j])

    output = output.replace('0', '.')

    return output


def GenerateEvilPuzzle(sudokuMatrix):
    n=0
    while True:
        n+=1
        randomSudoku = copy.deepcopy(sudokuMatrix)
        randomSudoku = RandomCellElimination(randomSudoku)
        url = requests.get('https://www.thonky.com/sudoku/evaluate-sudoku?puzzlebox='+ToThonkyNotation(randomSudoku))
        hmtlText = url.text
        score = hmtlText[hmtlText.find('Difficulty Score:') + 18:hmtlText.find('Difficulty Score:') + 19]
        if score.isdigit() and int(score) > 4:
            return randomSudoku


def DeleteCell(sudoku, nCells):
    maxIter = 90
    iter = 0
    i = 0
    while i != nCells and iter < maxIter:
        iter += 1
        while True:
            k = random.randint(0, 8)
            j = random.randint(0, 8)
            if sudoku[k][j] != 0: break
        tmp = sudoku[k][j]
        sudoku[k][j] = 0
        url = requests.get('https://www.thonky.com/sudoku/solution-count?puzzle=' + ToThonkyNotation(sudoku))
        hmtlText = url.text
        nSolution = hmtlText[hmtlText.find('Number of solutions: ') + 21:]
        nSolution = nSolution[:nSolution.find('<')]
        if not nSolution.isdigit() or int(nSolution) !=1:
            sudoku[k][j] = tmp
        else:
            i+=1

    return sudoku


def ArcConsistency(puzzle):
    queue = copy.deepcopy(puzzle.costraints)
    while queue:
        index = queue.pop(0)
        if Revise(index, puzzle):
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
    return True


def Revise(x, y, puzzle):
    revised = False
    if x != y:
        for val in puzzle.domains[x[0]][x[1]]:
            if val == puzzle.variables[y[0]][y[1]]:
                puzzle.domains[x[0]][x[1]].remove(val)
                revised = True
    return revised


def FowardChaining(puzzle, var):
    '''
    global nFalseEuristic

    for i in range(0, 9):
        Revise([var[0], var[1]],[var[0], i], puzzle)
        Revise([var[0], var[1]],[i, var[1]],  puzzle)
        if not (puzzle.domains[i][var[1]] and puzzle.domains[var[0]][i]) \
                or (puzzle.domains[var[0]][var[1]] == puzzle.domains[var[0]][i] and len(puzzle.domains[var[0]][i]) and var != [var[0],i])\
                or (puzzle.domains[var[0]][var[1]] == puzzle.domains[i][var[1]] and len(puzzle.domains[i][var[1]]) and var != [i,var[1]]):
            nFalseEuristic += 1
            return False
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
        Revise([var[0], var[1]],[i, j],  puzzle)
        if not puzzle.domains[i][j]:
            nFalseEuristic += 1
            return False
    return True

    puzzle.domains[var[0]][var[1]] = [puzzle.variables[var[0]][var[1]]]
    queue = [[[var[0], i], [var[0], var[1]]] for i in range(0, 9)]
    queue += [[[i, var[1]], [var[0], var[1]]] for i in range(0, 9)]
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    queue += [[[i, j], [var[0], var[1]]] for i, j in itertools.product(range(x, x + 3), range(y, y + 3))]
    queue.sort()
    queue = list(queue for queue, _ in itertools.groupby(queue))
    for i, j in itertools.product(range(0, 9), range(0, 9)):
        if [[i, j], [i, j]] in queue: queue.remove([[i, j], [i, j]])
    while queue:
        index = queue.pop(0)
        if Revise(index[1], index[0], puzzle):
            if not puzzle.domains[index[0][0]][index[0][1]]:
                global nFalseEuristic
                nFalseEuristic += 1
                return False
    return True
    '''
    global nFalseEuristic
    puzzle.domains[var[0]][var[1]] = [puzzle.variables[var[0]][var[1]]]
    for i in range(0, 9):
        if puzzle.domains[i][var[1]] == puzzle.domains[var[0]][var[1]] and [i, var[1]] != [var[0], var[1]]:
            nFalseEuristic += 1
            return False
        if puzzle.domains[var[0]][i] == puzzle.domains[var[0]][var[1]] and [var[0], i] != [var[0], var[1]]:
            nFalseEuristic += 1
            return False
        Revise([var[0], i], [var[0], var[1]], puzzle)
        Revise([i, var[1]], [var[0], var[1]], puzzle)
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
        if puzzle.domains[i][j] == puzzle.domains[var[0]][var[1]] and [i, j] != [var[0], var[1]]:
            nFalseEuristic += 1
            return False
        Revise([i, j], [var[0], var[1]], puzzle)
    return True


def MAC(puzzle, var):
    queue = [[[var[0], i], [var[0], var[1]]] for i in range(0, 9)]
    queue += [[[i, var[1]], [var[0], var[1]]] for i in range(0, 9)]
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    queue += [[[i, j], [var[0], var[1]]] for i, j in itertools.product(range(x, x + 3), range(y, y + 3))]
    for i, j in itertools.product(range(0, 9), range(0, 9)):
        if [[i, j], [i, j]] in queue: queue.remove([[i, j], [i, j]])
    while queue:
        index = queue.pop(0)
        if Revise(index[0],index[1], puzzle):
            if not puzzle.domains[index[0][0]][index[0][1]]: #or (len(puzzle.domains[index[0][0]][index[0][1]])==1 and puzzle.domains[index[0][0]][index[0][1]]== puzzle.domains[index[1][0]][index[1][1]]):
                global nFalseEuristic
                nFalseEuristic += 1
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
    return True

class Sudoku:
    def __init__(self, puzzle):
        self.variables = puzzle
        self.costraints = self.CreateCostraints()
        self.domains = self.CreateDomains()

    def IsComplete(self):
        for k, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[k][j] == 0:
                return False
        return True

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

    def CreateDomains(self):
        domains = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(0, 9)] for _ in range(0, 9)]
        for i, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[i][j] != 0:
                domains[i][j] = [self.variables[i][j]]
        return domains

    def GetVariable(self):
        domainDimension = [len(self.domains[i][j]) if self.variables[i][j] == 0 else math.inf for i, j in
                           itertools.product(range(0, 9), range(0, 9))]
        vars = [i for i, x in enumerate(domainDimension) if x == min(domainDimension)]
        return [trunc(vars[0] / 9), vars[0] % 9]

    def IsConsistent(self, var, val):
        for i in range(0, 9):
            if self.variables[var[0]][i] == val:
                return False
            if self.variables[i][var[1]] == val:
                return False
        x = var[0] - var[0] % 3
        y = var[1] - var[1] % 3
        for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
            if self.variables[i][j] == val:
                return False
        return True

    def GetOrder(self, var):
        rating = [0 for _ in range(0, 10)]
        for i in range(0, 9):
            for k in self.domains[var[0]][i]:
                rating[k] += 1
            for k in self.domains[i][var[1]]:
                rating[k] += 1
        x = var[0] - var[0] % 3
        y = var[1] - var[1] % 3
        for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
            for k in self.domains[i][j]:
                rating[k] += 1
        finalOrder = []
        for i in self.domains[var[0]][var[1]]:
            finalOrder.append(rating[i])
        zipped_lists = zip(finalOrder, self.domains[var[0]][var[1]])
        sorted_zipped_lists = sorted(zipped_lists, reverse=True)
        zipped_lists = [element for _, element in sorted_zipped_lists]
        return zipped_lists


def BackTrackingSudokuMAC(sudoku):
    if sudoku.IsComplete(): return sudoku
    var = sudoku.GetVariable()
    for val in sudoku.GetOrder(var):
        if sudoku.IsConsistent(var, val):
            sudoku.variables[var[0]][var[1]] = val
            domainBacktrack = copy.deepcopy(sudoku.domains)
            if MAC(sudoku, var):
                result = BackTrackingSudokuMAC(sudoku)
                if result:
                    return result
            sudoku.domains = domainBacktrack
            global nbackTrack
            nbackTrack += 1
            sudoku.variables[var[0]][var[1]] = 0
    return False


def BackTrackingSudokuFwdChaining(sudoku):
    if sudoku.IsComplete(): return sudoku
    var = sudoku.GetVariable()
    for val in sudoku.GetOrder(var):
        if sudoku.IsConsistent(var, val):
            sudoku.variables[var[0]][var[1]] = val
            domainBacktrack = copy.deepcopy(sudoku.domains)
            if FowardChaining(sudoku, var):
                result = BackTrackingSudokuFwdChaining(sudoku)
                if result:
                    return result
            sudoku.domains = domainBacktrack
            global nbackTrack
            nbackTrack += 1
            sudoku.variables[var[0]][var[1]] = 0
    return False



a = [[0, 4, 0, 0, 0, 0, 0, 0, 0], [6, 0, 7, 0, 0, 0, 8, 1, 0], [9, 0, 0, 6, 0, 0, 2, 0, 0], [0, 9, 0, 0, 4, 0, 0, 0, 0], [0, 6, 0, 8, 0, 3, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 3, 5, 0, 8, 0, 0, 4], [0, 7, 0, 0, 0, 2, 9, 0, 5], [5, 0, 0, 0, 0, 0, 3, 0, 0]]
c = [[0, 0, 5, 2, 3, 0, 0, 6, 0], [8, 2, 9, 6, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 8, 0, 0, 0, 0, 0, 2],
     [0, 0, 0, 7, 0, 9, 0, 0, 0], [3, 0, 0, 0, 0, 0, 4, 9, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 4, 1, 3, 7],
     [0, 6, 0, 0, 7, 5, 9, 0, 0]]

s = Sudoku(a) #GenerateEvilPuzzle(CreateRandomSolvedSudoku())
print('Found!')
s1 = copy.deepcopy(s)

nbackTrack = 0
nFalseEuristic = 0
t = Timer(lambda: BackTrackingSudokuFwdChaining(s1))
print(t.timeit(number=1))
print(nbackTrack)
print(nFalseEuristic)
print(BackTrackingSudokuFwdChaining(s1))

nbackTrack = 0
nFalseEuristic = 0
t = Timer(lambda: BackTrackingSudokuMAC(s))            
print(t.timeit(number=1))
print(nbackTrack)
print(nFalseEuristic)
print(BackTrackingSudokuMAC(s).variables)


