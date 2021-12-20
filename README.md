# #NFA FPRAS

Implementación de código para el paper #NFA admits an FPRAS: Efficient Enumeration, Counting,
and Uniform Generation for Logspace Classes, de Marcelo Arenas, Luis Alberto Croquevielle,
Rajesh Jayaram and Cristian Riveros.

# Estructura del proyecto

## Python

La carpeta `nfa_lib` contiene una implementación en Python de este algoritmo.
El archivo `nfa.py` modela una clase `NFA` con todos los métodos
necesarios para ejecutar el algoritmo, junto con algunas utilidades para 
la generación de instancias de NFAs aleatorias. Además, implementa
una función _baseline_ para el conteo de palabras mediante fuerza bruta.

El desglose de las funcionalidades es el siguiente:

- Clase NFA: 
    Hereda de la clase FA (Finite Automaton) que, a su vez, hereda de Automaton.
    Esta clase está inspirada en la librería [automata](https://github.com/caleb531/automata).
    - Método `__init__`: inicializa la instancia de un NFA y remueve estados inútiles (inalcanzables 
    y sumideros)
    - Método `read_input`: retorna la configuración de estados tras leer el string provisto, 
    comenzando por el conjunto de estados iniciales
    - Método `reachable`: retorna True si la configuración de estados tras leer el string provisto 
    contiene el estado provisto como parámetro
    - Métodos `remove_unreachable_states` y `remove_sink_states` remueven estados inalcanzables y 
    sumideros
    - Método `unroll`: "estira" un NFA en n niveles (n = largo del string de input) para estimar
    la cantidad de strings aceptados de largo n.
    - Métodos `compute_n_for_single_state` y `compute_n_for_states_set`: calculan una estimación 
    para el número de strings que llevan a un estado o a un conjunto de ellos, respectivamente
    - Método `sample`: muestrea un símbolo de forma aleatoria. Se llama recursivamente para 
    construir un string aleatorio y utilizarlo como muestra para el conteo de aceptación
    - Método `count_accepted`: para cada estado y nivel, muestrea una cantidad fija de strings y 
    computa estimaciones para la cantidad de aceptados. Retorna la estimación para N sobre el 
    conjunto de estados finales, es decir, la estimación de la cantidad de strings aceptados por el 
    NFA
    - Método `bruteforce_count_only`: usa fuerza bruta para probar cada string de largo n y 
    contabilizarlo de forma exacta. Tarda tiempo exponencial en n
    - Método `plot`: dibuja el autómata usando la librería `graphviz`. Funciona bien en un ambiente 
    Jupyter Notebook
    - Método `random`: método estático, se debe llamar como `NFA.random(...)`. Genera una instancia 
    aleatoria de un NFA con la cantidad de estados y densidad de transiciones provista.
    - Método `to_networkx`: retorna un multigrafo dirigido mediante la librería `networkx`
    - Método `cycle_height`: calcula la "altura de ciclo" del autómata, según se describe en 
    [este paper](https://doi.org/10.1016/j.ic.2021.104690). 
    El resultado del paper indica que si la altura 
    de ciclo es finita, entonces la cantidad de palabras aceptadas por el NFA (densidad de 
    aceptación) es polinomial en dicha altura. En particular, está acotada por un polinomio de grado 
    uno menos que la altura de ciclo. Si, por el contrario, la altura de ciclo es infinita, entonces
    la densidad _puede_ ser exponencial (y también polinomial).
    - Método `fron_random_matrix`: lee una matriz aleatoria y binaria NumPy e interpreta de ella la 
    cantidad de estados, las transiciones, los estados iniciales y finales. }
    - Método `to_text`: utilizado para serializar el NFA en el formato leido por la implementación en C++
- 
