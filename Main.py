import random

def LoadRandomSolvedSudoku():
    f = open("SolvedSudoku.txt", "r")
    for i in range(1, random.randint(0,9999)):
        f.readline()
    sudokuLine=f.readline()
    sudoku=[]
    for i in range(0, 9):
        sudoku.append(list(sudokuLine[9*i:9*(i+1)]))
    return sudoku


def CreateSudoku(solvedSudoku):
    nCells = random.randint(8, 20)
    for i in range(1, 81-nCells):
        solvedSudoku[random.randint(0, 8)][random.randint(0, 8)] = 0
    return solvedSudoku

class sudoku:
    def __init__(self, puzzle):
        sudoku.puzzle = puzzle
        sudoku.domains =


print(CreateSudoku(LoadRandomSolvedSudoku()))