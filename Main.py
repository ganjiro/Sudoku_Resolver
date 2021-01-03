import copy
import math
import random
import itertools
from math import trunc
from timeit import Timer
from random import sample
import requests


def LoadRandomSolvedSudoku(sudokuLine):
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


def GeneratePuzzle(difficulty):
    sudokuMatrix=CreateRandomSolvedSudoku()
    n = 0
    while True:
        n+=1
        randomSudoku = copy.deepcopy(sudokuMatrix)
        randomSudoku = RandomCellElimination(randomSudoku)
        url = requests.get('https://www.thonky.com/sudoku/evaluate-sudoku?puzzlebox='+ToThonkyNotation(randomSudoku))
        hmtlText = url.text
        score = hmtlText[hmtlText.find('Difficulty Score:') + 18:hmtlText.find('Difficulty Score:') + 19]
        if score.isdigit() and int(score) == difficulty:
            return randomSudoku


def SaveSudokuToFile(sudoku):
    f = open("puzzle.txt", "a")
    print(''.join([str(sudoku[i][j]) for i, j in itertools.product(range(0, 9), range(0, 9))]), file=f)
    f.close()


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


def ArcConsistency(puzzle,queue): #todo Levarla
    global nFalseEuristic
    while queue:
        index = queue.pop(0)
        if Revise(index[0], index[1], puzzle):
            if not puzzle.domains[index[0][0]][index[0][1]]:
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


def Revise(x, y, puzzle):
    revised = False
    if x != y:
        for val in puzzle.domains[x[0]][x[1]]:
            if val == puzzle.variables[y[0]][y[1]]:
                puzzle.domains[x[0]][x[1]].remove(val)
                revised = True
    return revised


def FowardChaining(puzzle, var):
    global nFalseEuristic
    queue = [[[var[0], i], [var[0], var[1]]] for i in range(0, 9)]
    queue += [[[i, var[1]], [var[0], var[1]]] for i in range(0, 9)]
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    queue += [[[i, j], [var[0], var[1]]] for i, j in itertools.product(range(x, x + 3), range(y, y + 3))]
    for i, j in itertools.product(range(0, 9), range(0, 9)):
        if [[i, j], [i, j]] in queue: queue.remove([[i, j], [i, j]])
    while queue:
        index = queue.pop(0)
        if Revise(index[0], index[1], puzzle):
            if not puzzle.domains[index[0][0]][index[0][1]]:
                nFalseEuristic += 1
                return False
    return True


def MAC2(puzzle, var):#todo levarla
    queue = [[[var[0], i], [var[0], var[1]]] for i in range(0, 9)]
    queue += [[[i, var[1]], [var[0], var[1]]] for i in range(0, 9)]
    x = var[0] - var[0] % 3
    y = var[1] - var[1] % 3
    queue += [[[i, j], [var[0], var[1]]] for i, j in itertools.product(range(x, x + 3), range(y, y + 3))]
    for i, j in itertools.product(range(0, 9), range(0, 9)):
        if [[i, j], [i, j]] in queue: queue.remove([[i, j], [i, j]])
    return ArcConsistency(puzzle, queue)


def MAC(puzzle, var):
    global nFalseEuristic
    newarcs = []
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
            if not puzzle.domains[index[0][0]][index[0][1]]:
                nFalseEuristic += 1
                return False
            newarcs = [[[index[0][0], k], [index[0][0], index[0][1]]] for k in range(0, 9)]
            newarcs += [[[k, index[0][1]], [index[0][0], index[0][1]]] for k in range(0, 9)]
            x = index[0][0] - index[0][0] % 3
            y = index[0][1] - index[0][1] % 3
            newarcs += [[[k, l],[index[0][0], index[0][1]]] for k, l in
                        itertools.product(range(x, x + 3), range(y, y + 3))]
            newarcs.sort()
            newarcs = list(newarcs for newarcs, _ in itertools.groupby(newarcs))
            for i, j in itertools.product(range(0, 9), range(0, 9)):
                if [[i, j], [i, j]] in newarcs: newarcs.remove([[i, j], [i, j]])
            newarcs.remove([[index[1][0], index[1][1]],[index[0][0], index[0][1]]])
    while newarcs:
        index = newarcs.pop(0)
        if puzzle.domains[index[0][0]][index[0][1]] == puzzle.domains[index[1][0]][index[1][1]] and len(
                puzzle.domains[index[1][0]][index[1][1]]) == 1:
            nFalseEuristic += 1
            return False
    return True


class Sudoku:
    def __init__(self, puzzle):
        self.variables = puzzle
        self.costraints = self.CreateCostraints() #todo levarla
        self.domains = []
        self.GetDomain()

    def IsComplete(self):
        for k, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[k][j] == 0:
                return False
        return True

    def CreateCostraints(self):  #todo levarla
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

    def GetDomain(self):
        domains = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(0, 9)] for _ in range(0, 9)]
        for i, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[i][j] != 0:
                domains[i][j] = [self.variables[i][j]]
                for k in range(0,9):
                    if k == j: continue
                    if self.variables[i][j] in domains[i][k]: domains[i][k].remove(self.variables[i][j])
                for k in range(0, 9):
                    if k == i: continue
                    if self.variables[i][j] in domains[k][j]: domains[k][j].remove(self.variables[i][j])
                x = i - i % 3
                y = j - j % 3
                for n, m in itertools.product(range(x, x + 3), range(y, y + 3)):
                    if [n, m] == [i, j]: continue
                    if self.variables[i][j] in domains[n][m]: domains[n][m].remove(self.variables[i][j])
        self.domains = domains

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

    def GetOrder1(self, var):
        return copy.deepcopy(self.domains[var[0]][var[1]])

    def GetOrder2(self, var):
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
        sorted_zipped_lists = sorted(zipped_lists)
        zipped_lists = [element for _, element in sorted_zipped_lists]
        return zipped_lists

    def GetOrder3(self, var):
        rating = [1 for _ in range(0, 10)]
        for i in range(0, 9):
            for k in self.domains[var[0]][i]:
                rating[k] *= len(self.domains[var[0]][i])-1
            for k in self.domains[i][var[1]]:
                rating[k] *= len(self.domains[i][var[1]])-1
        x = var[0] - var[0] % 3
        y = var[1] - var[1] % 3
        for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
            for k in self.domains[i][j]:
                rating[k] *= len(self.domains[i][j])-1
        for k in range(0, 10):
            if rating[k] == 1: rating[k] = math.inf
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
    for val in sudoku.GetOrder3(var):
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
    for val in sudoku.GetOrder3(var):
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



def BackTrackingSudoku(sudoku):
    if sudoku.IsComplete(): return sudoku
    var = sudoku.GetVariable()
    for val in sudoku.GetOrder3(var):
        if sudoku.IsConsistent(var, val):
            sudoku.variables[var[0]][var[1]] = val
            domainBacktrack = copy.deepcopy(sudoku.domains)
            result = BackTrackingSudokuFwdChaining(sudoku)
            if result:
                return result
            sudoku.domains = domainBacktrack
            global nbackTrack
            nbackTrack += 1
            sudoku.variables[var[0]][var[1]] = 0
    return False


f = open("puzzle.txt", "r")
f.readline()

for i in range(0, 20):


    s = Sudoku(LoadRandomSolvedSudoku(f.readline())) #
    print('Found!')
    s1 = copy.deepcopy(s)
    s3 = copy.deepcopy(s)
    s2 = copy.deepcopy(s)

    nbackTrack = 0
    nFalseEuristic = 0
    t = Timer(lambda: BackTrackingSudokuFwdChaining(s1))
    print('Tempo itr {} Fwd: {}'.format(i,t.timeit(number=1)))
    print('Numero di Backtrack: {}'.format(nbackTrack))
    print('Numero di False: {}'.format(nFalseEuristic))
    print('Fwd: {}'.format(BackTrackingSudokuFwdChaining(s2)))

    nbackTrack = 0
    nFalseEuristic = 0
    t = Timer(lambda: BackTrackingSudokuMAC(s))
    print('Tempo itr {} MAC: {}'.format(i,t.timeit(number=1)))
    print('Numero di Backtrack: {}'.format(nbackTrack))
    print('Numero di False: {}'.format(nFalseEuristic))
    print('MAC: {}'.format(BackTrackingSudokuMAC(s3)))

f.close()

