import yaml 
import copy
import re
import sys
from utils import pretty_print_timetable_aux_zile
from math import sqrt, log
from check_constraints import check_mandatory_constraints, check_optional_constraints
import numpy as np
from scipy.special import softmax
from datetime import datetime

# Constante
N = 'N'
Q = 'Q'
STATE = 'state'
PARENT = 'parent'
ACTIONS = 'actions'

# Clasa in care retin informatiile despre datele din orar.
class InformatiiOrar:
    def __init__(self, cursuri, sali, profi, zile, intervale):
        # Dictionar tip {curs: nr_studenti_curs}
        self.cursuri = cursuri
        # Dictionar tip {sala: {'capacitate': nr, 'materii': [lista_materii]}}
        self.sali = sali
        # Dictionar tip {nume_prof : {'Cursuri' : [lista_cursuri], 'Constrangeri': [lista_constrangeri]}
        self.profi = profi
        # [lista_zile]
        self.zile = zile
        # [lista_intervale]
        self.intervale = intervale

        # Dictionare folosite suplimentar pentru sortarea vecinilor atunci cand ii generam.
        self.sali_per_curs = {}
        self.profi_per_curs = {}

class Stare:
    def __init__(self):
        self.stare_orar = {}

# Initializam intervalele orarului 
def init_orar(orar):
    stare = Stare()
    stare.stare_orar =  {zi: {} for zi in orar.zile}
    for zi in orar.zile:
        stare.stare_orar[zi] = {}
        for slot in orar.intervale:
            stare.stare_orar[zi][slot] = {sala : None for sala in orar.sali}
    return stare

def calculare_cost(orar, stare_orar):
    studenti_repartizati = {curs: 0 for curs in orar.cursuri}
    for zi, sloturi in stare_orar.items():
        for slot, sali in sloturi.items():
            for sala, info_sala in sali.items():
                if info_sala:
                    prof, curs = info_sala
                    studenti_repartizati[curs] += orar.sali[sala]['Capacitate']

    studenti_nerepartizati_cost = sum(
        orar.cursuri[curs] - repartizati
        for curs, repartizati in studenti_repartizati.items()
        if orar.cursuri[curs] > repartizati
    ) * 10

    materii_neacoperite_cost = sum(
        orar.cursuri[curs] - repartizati > 0
        for curs, repartizati in studenti_repartizati.items()
    ) * 150
    
    cost_soft = calculare_cost_soft(orar, stare_orar)
    cost = cost_soft + studenti_nerepartizati_cost + materii_neacoperite_cost
    return cost
    
def parse_interval(interval : str):
    intervals = interval.split('-')
    return int(intervals[0].strip()), int(intervals[1].strip())

def calculare_cost_soft(orar, stare_orar):
    constrangeri_incalcate = 0
    for prof in orar.profi:
        for const in orar.profi[prof]['Constrangeri']:
            if const[0] != '!':
                continue
            else:
                const = const[1:]
                if const in orar.zile:
                    zi = const
                    if zi in stare_orar:
                        for slot in stare_orar[zi]:
                            for sala in stare_orar[zi][slot]:
                                if stare_orar[zi][slot][sala]:
                                    crt_prof, _= stare_orar[zi][slot][sala]
                                    if prof == crt_prof:
                                        constrangeri_incalcate += 1 
                elif '-' in const:
                    interval = parse_interval(const)
                    start, end = interval
                    if start != end - 2:
                        intervals = [(i, i + 2) for i in range(start, end, 2)]
                    else:
                        intervals = [(start, end)]
                    for zi in stare_orar:
                        for slot in intervals:
                            if slot in stare_orar[zi]:
                                for sala in stare_orar[zi][slot]:
                                    if stare_orar[zi][slot][sala]:
                                        crt_prof, _ = stare_orar[zi][slot][sala]
                                        if prof == crt_prof:
                                            constrangeri_incalcate += 1
    return constrangeri_incalcate


def generare_vecini(orar, stare):
    vecini = []
    studenti_repartizati = {curs: 0 for curs in orar.cursuri.keys()}
    nr_ore_profi = {prof: 0 for prof in orar.profi.keys()}
    profi_ocupati_in_interval = {}

    for zi, sloturi in stare.stare_orar.items():
        profi_ocupati_in_interval[zi] = {slot: set() for slot in sloturi.keys()}
        for slot, sali in sloturi.items():
            for sala, info_sala in sali.items():
                if info_sala:
                    prof, curs = info_sala
                    profi_ocupati_in_interval[zi][slot].add(prof)
                    studenti_repartizati[curs] += orar.sali[sala]['Capacitate']
                    nr_ore_profi[prof] = nr_ore_profi[prof] + 1


    # Calculam numarul de materii neacoperite
    cursuri_neacoperite = []
    for curs in orar.cursuri:
        if orar.cursuri[curs] - studenti_repartizati[curs] > 0:
            cursuri_neacoperite.append(curs)
    cursuri_neacoperite.reverse()
    
    # Generam mutari.

    # 1. Selectam cursurile care au ramas neacoperite si prioritizam dupa numarul de studenti total la curs, dupa 
    # ca si criteriu secundar numarul minim de studenti repartizati la acel curs.
    cursuri_neacoperite_sort = sorted(cursuri_neacoperite, key=lambda x: studenti_repartizati[x])
    cursuri_neacoperite_sort_2 = sorted(cursuri_neacoperite_sort, key = lambda x: orar.cursuri[x])
    for curs in cursuri_neacoperite_sort_2:
        # 2. Selectam salile disponibile pentru cursul ales si alegem sala cu capacitatea cea mai mare.
        sali_sortate_cu_capacitate_mica = sorted(orar.sali_per_curs[curs], key = lambda x: orar.sali[sala]['Capacitate'])
        for sala in sali_sortate_cu_capacitate_mica:
            # 3. Selectam profesorii, selectandu-i pe cei cu nr de ore < 7
            # si prioritizandu-i pe cei care au mai putine ore in orar.
            profi_filtrati = filter(lambda prof: nr_ore_profi[prof] < 7, orar.profi_per_curs[curs])
            profi_sort = sorted(profi_filtrati, key = lambda x: nr_ore_profi[x])
            for prof in profi_sort:
                     # 4. Pentru o zi si un interval, daca  proful nu mai  preda simultan alt curs
                     # si daca intrarea este nula, adaugam materia in orar. 
                for zi in orar.zile:
                    for slot in orar.intervale:
                        if prof not in profi_ocupati_in_interval[zi][slot]:
                            if not stare.stare_orar[zi][slot][sala]:
                                stare_noua = copy.deepcopy(stare)
                                stare_noua.stare_orar[zi][slot][sala] = prof, curs
                                vecini.append(stare_noua)
    return vecini

# Algoritmul de Hill Climbing 
def hill_climbing(orar, initial_state):
    state = initial_state
    cost = calculare_cost(orar, state.stare_orar)
    while cost > 0:
        neighbours = generare_vecini(orar, state)
        if not neighbours:
            break
        neighbours.sort(key=lambda neighbor: calculare_cost(orar, neighbor.stare_orar))
        best = neighbours[0]
        best_cost = calculare_cost(orar, best.stare_orar)
        if best_cost < cost:
            state = best
        cost = calculare_cost(orar, state.stare_orar)
    return state.stare_orar

# Selectam vecinii care au cost soft 0 si care au un numar mare de materii acoperit.
def selectare_vecini_mcts(orar, initial_state):
    neighbours = generare_vecini(orar, initial_state)
    neighbours.sort(key=lambda neighbor: calculare_cost(orar, neighbor.stare_orar))
    chosen_neighbours = []
    for neigh in neighbours:
        if calculare_cost(orar, neigh.stare_orar) < calculare_cost(orar, initial_state.stare_orar):
            chosen_neighbours.append(neigh)
    return chosen_neighbours[:10]

# Funcție ce întoarce un nod nou, eventual copilul unui nod dat ca argument
# N = numarul de vizitari
# Q = indicatie a calitatii nodului
def init_node(stare, parent = None):
    return {N: 0, Q: 0, STATE: stare, PARENT: parent, ACTIONS: {}}

# Funcție ce verifică dacă o stare este finală = daca toate materiile au fost acoperite.
# Deoarece functia de generare vecini verifica respectarea celorlaltor constrangeri hard,
# aceasta e singura pe care o verificam.
def is_final(orar, state):
    acoperire_reala = {curs: 0 for curs in orar.cursuri}
    for zi, sloturi in state.stare_orar.items():
        for slot, sali in sloturi.items():
            for sala, info_sala in sali.items():
               if info_sala is None:
                  continue
               prof, curs = info_sala
               acoperire_reala[curs] += orar.sali[sala]['Capacitate']

    return all(orar.cursuri[curs] <= acoperire_reala[curs] for curs in orar.cursuri)

# Constanta care reglează raportul între explorare și exploatare (CP = 0 -> doar exploatare)
CP = 1.0 / sqrt(2.0)

# Funcție ce alege o acțiune dintr-un nod
def select_action(node, c = CP):
    N_node = node[N]
    max_score = -1
    best_action = None
    for a, n in node[ACTIONS].items():
        curr_score = n[Q] / n[N] + c * sqrt(2 * log(N_node) / n[N])
        if max_score < curr_score:
            max_score = curr_score
            best_action = a
    return best_action


def mcts(orar, state0, budget):

    # Arborele de start este un nod gol.
    tree = init_node(state0)

    for _ in range(budget):

        state = state0
        node = tree

        # Coborâm în arbore până când ajungem la o stare finală sau la un nod cu acțiuni neexplorate.
        while node and (not is_final(orar, state) and all(action in node[ACTIONS] for action in selectare_vecini_mcts(orar, state))): 
                # Alegem cea mai buna strategie pentru nodul nostru
                action = select_action(node)
                # Actualizam starea nodului
                state = action
                # Nodul devine nodul copil asociat actiunii respective.
                node = node[ACTIONS][action]
                print(f"Coboram in arbore.")

        # Dacă am ajuns într-un nod care nu este final și din care nu s-au
        # `încercat` toate acțiunile, construim un nod nou.

        # Explorarea
        if node and not is_final(orar, state): 
            actions = []
            for a in selectare_vecini_mcts(orar, state):
                if a not in node[ACTIONS]:
                    actions.append(a)
            # Selectam nodul folosind o distributie de probabilitate softmax
            idx = np.random.choice(len(actions), p = softmax(np.array([-calculare_cost(orar, state.stare_orar) for state in actions if state is not None])))
            state = actions[idx]
            node = init_node(state, node)
            node[PARENT][ACTIONS][actions[idx]] = node
            print(f"Exploram nodul cu noul copil cu o noua actiune.")

        # Se simulează o desfășurare a jocului până la ajungerea într-o
        # starea finală. Se evaluează recompensa în acea stare.

        # Simularea
        # Starea jocului = nodul curent in arbore
        state = node[STATE]
        invalid = False

        while not is_final(orar, state) and invalid == False:
            actions = selectare_vecini_mcts(orar, state)
            # Daca nu mai gasim vecini (nu mai putem continua din acest orar), continuam.
            if len(actions) == 0:
                invalid = True
                continue  
            idx = np.random.choice(len(actions), p = softmax(np.array([-calculare_cost(orar, state.stare_orar) for state in actions if state is not None])))
            state = actions[idx]

        if is_final(orar, state) and calculare_cost(orar, state.stare_orar) == 0:
            return state
        
        reward = calculare_cost(orar, state.stare_orar)

        # Se actualizează toate nodurile de la node către rădăcină:
        #  - se incrementează valoarea N din fiecare nod
        #  - pentru nodurile corespunzătoare acestui jucător, se adună recompensa la valoarea Q
        #  - pentru nodurile celelalte, valoarea Q se adună 1 cu minus recompensa

        # Reward-ul este negativ deoarece incercam sa cautam o stare de cost minim.
        while node:
            node[N] += 1 # Incrementam numarul de vizitari
            node[Q] -= reward # Adaugam reward
            node = node[PARENT] # Ne mutam in nodul parinte
    
    if tree:
        final_action = select_action(tree, 0.0)
        if final_action:
            return final_action
    return state


# Formateaza intervalele de la string la lista de tupluri. 
def format_intervale(intervale):
    new_intervale = []
    for interval in intervale:
        times = re.findall(r'\d+', interval)[:2]
        (start, end) = (int(times[0]), int(times[1]))
        new_intervale.append((start, end))
    return new_intervale

# # Formatam timpul pentru a afisa in fisier.
# def format_timedelta(td):
#     total_seconds = int(td.total_seconds())
#     minutes = total_seconds // 60
#     seconds = total_seconds % 60
#     milliseconds = td.microseconds // 1000
#     return f"{minutes} minute {seconds} secunde {milliseconds} milisecunde"

def main():
    if len(sys.argv) != 3:
        print("Nu avem suficiente argumente.")
        sys.exit(1)
    algoritm = sys.argv[1]
    input_file = sys.argv[2]
    # Citim fisierul yaml
    file_path = f'inputs/{input_file}.yaml'
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        print("Eroare citire data din fisier")
    profesori = {nume_prof: {'Materii': prof_info['Materii'], 'Constrangeri': prof_info['Constrangeri']}
            for nume_prof, prof_info in data['Profesori'].items()}
    orar = InformatiiOrar(cursuri = data['Materii'], sali = data['Sali'], profi = profesori, zile = data['Zile'], intervale = format_intervale(data['Intervale']))

    # Facem dictionare pentru a stoca informatii pentru generarea vecinilor
    for curs in orar.cursuri:
        orar.sali_per_curs[curs] = []
        for sala in orar.sali:
            if curs in orar.sali[sala]['Materii']:
                orar.sali_per_curs[curs].append(sala)
        orar.profi_per_curs[curs] = []
        for prof in orar.profi:
            if curs in orar.profi[prof]['Materii']:
                orar.profi_per_curs[curs].append(prof)

    # Alegem algoritmul 
    if algoritm == "hc":
        start_time = datetime.now()
        stare_orar_final = hill_climbing(orar, init_orar(orar))
        end_time = datetime.now()
        time = end_time - start_time
        str = pretty_print_timetable_aux_zile(stare_orar_final, file_path)
        file_path_out = f'outputs_hc/{input_file}.txt'
        with open(file_path_out, 'w') as file:
            file.write(str)
            file.write(f"\nCOST HARD: {check_mandatory_constraints(stare_orar_final, data)}\n")
            file.write(f"COST SOFT: {check_optional_constraints(stare_orar_final, data)}\n")
            file.write(f"TIMP RULARE: {time}")
    elif algoritm == "mcts":
        state = init_orar(orar)
        start_time = datetime.now()
        state = mcts(orar, state, budget = 30)
        end_time = datetime.now()
        time = end_time - start_time
        str = pretty_print_timetable_aux_zile(state.stare_orar, file_path)
        file_path_out = f'outputs_mcts/{input_file}.txt'
        with open(file_path_out,'w') as file:
            file.write(str)
            file.write(f"\nCOST HARD: {check_mandatory_constraints(state.stare_orar, data)}\n")
            file.write(f"COST SOFT: {check_optional_constraints(state.stare_orar, data)}\n")
            file.write(f"TIMP RULARE: {time}")
            print("COST HARD:", check_mandatory_constraints(state.stare_orar, data))
            print("COST SOFT:", check_optional_constraints(state.stare_orar, data))
            print("COSTUL MEU:", calculare_cost(orar, state.stare_orar))


if __name__ == "__main__":
    main()