import copy
import math
import random
import itertools
from math import trunc
from timeit import Timer
from random import sample
import requests
import matplotlib.pyplot as plt


def StringToMatrix(sudokuLine):
    """
    Trasforma una stringa in una matrice 9x9
    :param sudokuLine:
    :return: Matrice 9x9
    """
    sudoku = []
    for i in range(0, 9):
        sudoku.append(list(map(int, sudokuLine[9 * i:9 * (i + 1)])))
    return sudoku


def CreateRandomSolvedSudoku():
    """
    Crea una Matrice 9x9 che rispetta le regole del sudoku
    :return: matrice 9x9
    """
    rows = [g * 3 + r for g in sample(range(3), 3) for r in sample(range(3), 3)]
    cols = [g * 3 + c for g in sample(range(3), 3) for c in sample(range(3), 3)]
    nums = sample(range(1, 10), 9)
    return [[nums[(3 * (r % 3) + r // 3 + c) % 9] for c in cols] for r in rows]


def GenerateRandomSudoku():
    """
    Crea un sudoku con tra 55 a 65 celle vuote con una sola soluzione
    :return: matrice 9x9
    """
    solvedSudoku = CreateRandomSolvedSudoku()
    nCells = random.randint(55, 65)
    return DeleteCell(solvedSudoku, nCells)


def ToThonkyNotation(sudoku):
    """
    Trasforma una matrice sudoku nella notazione di thonky, una stringa con le righe concatenate e "." nelle celle vuote
    :param sudoku:
    :return: stringa
    """
    output = ''
    for i in range(0, 9):
        for j in range(0, 9):
            output += str(sudoku[i][j])

    output = output.replace('0', '.')
    return output


def GeneratePuzzle(difficulty):
    """
    Genera un puzzle con un grado di difficolta da 1 a 6, la valutazione della difficolta è rimandata al sito thonky il
    quale prende come parametro il sudoku nella sua notazione.
    :param difficulty:
    :return: matrice 9x9
    """
    n = 0
    while True:
        if difficulty < 1:
            difficulty = 1
        elif difficulty > 6:
            difficulty = 6
        n += 1
        randomSudoku = GenerateRandomSudoku()
        url = requests.get('https://www.thonky.com/sudoku/evaluate-sudoku?puzzlebox=' + ToThonkyNotation(randomSudoku))
        hmtlText = url.text
        score = hmtlText[hmtlText.find('Difficulty Score:') + 18:hmtlText.find('Difficulty Score:') + 19]
        if score.isdigit() and int(score) == difficulty:
            return randomSudoku


def SaveSudokuToFile(sudoku, fileName):
    """
    Converte una matrice sudoku in una stringa concatenando le righe e la salva su un file
    :param sudoku:
    :param fileName:
    """
    f = open(fileName, "a")
    print(''.join([str(sudoku[i][j]) for i, j in itertools.product(range(0, 9), range(0, 9))]), file=f)
    f.close()


def DeleteCell(sudoku, nCells):
    """
    Elimina dal sudoku delle celle mantenendo l'unicità della soluzione, il controllo è rimandato al sito thonky il
    quale prende come parametro il sudoku nella sua notazione.
    :param sudoku:
    :param nCells:
    :return: matrice 9x9
    """
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
        if not nSolution.isdigit() or int(nSolution) != 1:
            sudoku[k][j] = tmp
        else:
            i += 1
    return sudoku


def BackTrackingSudokuMAC(sudoku):
    """
    Risolve il sudoku con backtraking usando MAC
    :param sudoku: class Sudoku
    :return: False se non risolvibile else la matrice del sudoku risolto
    """
    if sudoku.IsComplete(): return sudoku
    var = sudoku.GetVariableMRV()
    for val in sudoku.SortDomain2(var):
        if sudoku.IsConsistent(var, val):
            sudoku.variables[var[0]][var[1]] = val
            domainBacktrack = copy.deepcopy(sudoku.domains)
            if sudoku.MAC(var):
                result = BackTrackingSudokuMAC(sudoku)
                if result:
                    return result
            sudoku.domains = domainBacktrack
            global nbackTrack
            nbackTrack += 1
            sudoku.variables[var[0]][var[1]] = 0
    return False


def BackTrackingSudokuFwdChaining(sudoku):
    """
    Risolve il sudoku con backtraking usando Forward Checking
    :param sudoku: class Sudoku
    :return: False se non risolvibile else la matrice del sudoku risolto
    """
    if sudoku.IsComplete(): return sudoku
    var = sudoku.GetVariableMRV()
    for val in sudoku.SortDomain2(var):
        if sudoku.IsConsistent(var, val):
            sudoku.variables[var[0]][var[1]] = val
            domainBacktrack = copy.deepcopy(sudoku.domains)
            if sudoku.FowardChaining(var):
                result = BackTrackingSudokuFwdChaining(sudoku)
                if result:
                    return result
            sudoku.domains = domainBacktrack
            global nbackTrack
            nbackTrack += 1
            sudoku.variables[var[0]][var[1]] = 0
    return False


def Test(difficulty, nTest):
    """
    Esegue backtracking con mac e Forward Checking e salva i risultati in file.
    :param difficulty: da 0 a 6 se 0 si usano sudoku di difficolta casuale
    :param nTest: Numero di sudoku da testare
    """
    global nbackTrack, nFalseInference
    if difficulty == 0:
        f = open("puzzle.txt", "r")
        output = "RandomSudokuComparison.png"
        if nTest > 300:
            nTest = 300
    else:
        f = open("puzzle" + str(difficulty) + ".txt", "r")
        output = "Diff" + str(difficulty) + "SudokuComparison.png"
        if nTest > 30:
            nTest = 30

    totalBacktrack = [0 for _ in range(0, 2)]
    totalTime = [0 for _ in range(0, 2)]

    for i in range(0, nTest):
        s = Sudoku(StringToMatrix(f.readline()))
        s1 = copy.deepcopy(s)

        nFalseInference = 0
        nbackTrack = 0
        t = Timer(lambda: BackTrackingSudokuMAC(s))
        totalTime[0] += t.timeit(number=1)
        totalBacktrack[0] += nbackTrack

        nbackTrack = 0
        t = Timer(lambda: BackTrackingSudokuFwdChaining(s1))
        totalTime[1] += t.timeit(number=1)
        totalBacktrack[1] += nbackTrack

    f.close()

    avgTime = [i / nTest for i in totalTime]
    avgBacktrack = [i / nTest for i in totalBacktrack]

    label = ['MAC', 'Fwd Check']
    plt.figure(figsize=(9, 3))

    plt.subplot(121)
    plt.bar(label, avgTime)
    plt.title('Execution Time')
    plt.margins(x=0, y=+0.25)
    plt.subplot(122)
    plt.bar(label, avgBacktrack)
    plt.title('N° Backtrack')
    plt.margins(x=0, y=+0.25)
    plt.rcParams.update({'font.size': 20})

    plt.savefig(output)
    plt.close()


class Sudoku:
    def __init__(self, puzzle):
        self.variables = puzzle
        self.domains = []
        self.GetDomain()

    def IsComplete(self):
        """
        :return: True se è completo
        """
        for k, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[k][j] == 0:
                return False
        return True

    def GetDomain(self):
        """
        Calcola il domino del sudoku eliminando i valori che sono incompatibili con quelli già assegnati
        :return:
        """
        domains = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(0, 9)] for _ in range(0, 9)]
        for i, j in itertools.product(range(0, 9), range(0, 9)):
            if self.variables[i][j] != 0:
                domains[i][j] = [self.variables[i][j]]
                for k in range(0, 9):
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

    def GetVariableMRV(self):
        """
        Ritorna gli indici della variabile nel sudoku che che ha meno valori nel domino
        :return: lista di indici
        """
        domainDimension = [len(self.domains[i][j]) if self.variables[i][j] == 0 else math.inf for i, j in
                           itertools.product(range(0, 9), range(0, 9))]
        vars = [i for i, x in enumerate(domainDimension) if x == min(domainDimension)]
        return [trunc(vars[0] / 9), vars[0] % 9]

    def GetRandomVariable(self):
        """
        Ritorna gli indici di una variabile casuale non assegnata
        :return:
        """
        notAssigned = [[i, j] for i, j in itertools.product(range(0, 9), range(0, 9)) if self.variables[i][j] == 0]
        return notAssigned[random.randint(0, len(notAssigned)) - 1]

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

    def SortDomain1(self, var):
        """
        dominio in ordine crescente
        :param var: indici del dominio da assegnare
        :return: lista ordinata del domino
        """
        return copy.deepcopy(self.domains[var[0]][var[1]])

    def SortDomain2(self, var):
        """
        Conta il numero di occorrenza di ogni valore nel dominio dei vicini(occurrences) e ordina il domino della var
        considerata in ordine crescente rispetto ad occurrences
        :param var: indici del dominio da assegnare
        :return: lista ordinata del domino
        """
        occurrences = [0 for _ in range(0, 10)]
        for i in range(0, 9):
            if i == var[1]: continue
            for k in self.domains[var[0]][i]:
                occurrences[k] += 1
        for i in range(0, 9):
            if i == var[0]: continue
            for k in self.domains[i][var[1]]:
                occurrences[k] += 1
        x = var[0] - var[0] % 3
        y = var[1] - var[1] % 3
        for i, j in itertools.product(range(x, x + 3), range(y, y + 3)):
            if [i, j] == var: continue
            for k in self.domains[i][j]:
                occurrences[k] += 1
        finalOrder = []
        for i in self.domains[var[0]][var[1]]:
            finalOrder.append(occurrences[i])
        zipped_lists = zip(finalOrder, self.domains[var[0]][var[1]])
        sorted_zipped_lists = sorted(zipped_lists)
        zipped_lists = [element for _, element in sorted_zipped_lists]
        return zipped_lists

    def SortDomain3(self, var):
        """
          ordina la lista in maniera casuale
          :param var: indici del dominio da assegnare
          :return: lista ordinata del domino
        """
        a = copy.deepcopy(self.domains[var[0]][var[1]])
        random.shuffle(a)
        return a

    def FowardChaining(self, var):
        """
        Esegue l'algoritmo di inferenza Foward checking
        :param var: la variabile assegnata
        :return: false se il dominio di una delle variabili è vuoto else true
        """
        global nFalseInference
        queue = [[[var[0], i], [var[0], var[1]]] for i in range(0, 9)]
        queue += [[[i, var[1]], [var[0], var[1]]] for i in range(0, 9)]
        x = var[0] - var[0] % 3
        y = var[1] - var[1] % 3
        queue += [[[i, j], [var[0], var[1]]] for i, j in itertools.product(range(x, x + 3), range(y, y + 3))]
        for i, j in itertools.product(range(0, 9), range(0, 9)):
            if [[i, j], [i, j]] in queue: queue.remove([[i, j], [i, j]])
        while queue:
            index = queue.pop(0)
            if self.Revise(index[0], index[1]):
                if not self.domains[index[0][0]][index[0][1]]:
                    nFalseInference += 1
                    return False
        return True

    def MAC(self, var):
        """
        Esegue l'algoritmo di inferenza MAC
        :param var: la variabile assegnata
        :return: false se il dominio di una delle variabili è vuoto o due dominii delle variabili connesse contengono
        lo stesso unico elemento else true
        """
        global nFalseInference
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
            if self.Revise(index[0], index[1]):
                if not self.domains[index[0][0]][index[0][1]]:
                    nFalseInference += 1
                    return False
                newarcs = [[[index[0][0], k], [index[0][0], index[0][1]]] for k in range(0, 9)]
                newarcs += [[[k, index[0][1]], [index[0][0], index[0][1]]] for k in range(0, 9)]
                x = index[0][0] - index[0][0] % 3
                y = index[0][1] - index[0][1] % 3
                newarcs += [[[k, l], [index[0][0], index[0][1]]] for k, l in
                            itertools.product(range(x, x + 3), range(y, y + 3))]
                newarcs.sort()
                newarcs = list(newarcs for newarcs, _ in itertools.groupby(newarcs))
                for i, j in itertools.product(range(0, 9), range(0, 9)):
                    if [[i, j], [i, j]] in newarcs: newarcs.remove([[i, j], [i, j]])
                newarcs.remove([[index[1][0], index[1][1]], [index[0][0], index[0][1]]])
        while newarcs:
            index = newarcs.pop(0)
            if self.domains[index[0][0]][index[0][1]] == self.domains[index[1][0]][index[1][1]] and len(
                    self.domains[index[1][0]][index[1][1]]) == 1:
                nFalseInference += 1
                return False
        return True

    def Revise(self, x, y):
        """
        Elimina il valore della variabile all'indice y dal dominio della variabile x
        :return: true se il dominio è stato modificato else false
        """
        revised = False
        if x != y:
            for val in self.domains[x[0]][x[1]]:
                if val == self.variables[y[0]][y[1]]:
                    self.domains[x[0]][x[1]].remove(val)
                    revised = True
        return revised


def main():
    Test(2, 20)


if __name__ == "__main__":
    main()
