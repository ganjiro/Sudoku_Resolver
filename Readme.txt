Per eseguire i test su MAC e forward chaining cambiare i parametri in input all'interno della chiamata alla funzione test() nel main.

Per eseguire il test sul dataset casuale passare come parametro difficulty = 0.

Per eseguire il test senza MRV va sostituita in entrambe le versioni di backtraking
    la riga: 'var = sudoku.GetVariableMRV()' con 'var = sudoku.GetRandomVariable()' //Attenzione questo test è particolarmente lento

Per eseguire il test sull'ordinamento delle variabili bisogna aggiungere il codice nel file TestSorting.txt al main.Py
    e cambiare la funzione test(difficulty, nTest) nel main con test1()
        //Ho scelto di tenere questo test separato in modo da rendere il codice più ordinato, considerando che
         non è il focus del progetto


Materiale per la creazione del sudoku risolto:
- https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python

Sito utillizzato per la valutazione dell'unicità del sudoku e della difficoltà:
- https://www.thonky.com/sudoku/solution-count
- https://www.thonky.com/sudoku/evaluate-sudoku