# QR: Modelling a sink
# (c) Jaap Jumelet

import itertools as it
import copy
from graphviz import Digraph


class State:
    def __init__(self, quantities):
        self.quantities = quantities
        self.influences = []
        self.props = []
        self.constants = []

    def __getitem__(self, key):
        if key == 0:
            return self.quantities
        elif key == 1:
            return self.influences
        elif key == 2:
            return self.props
        elif key == 3:
            return self.constants
        else:
            raise IndexError

    def __setitem__(self, key, value):
        if key == 0:
            self.quantities = value
        elif key == 1:
            self.influences = value
        elif key == 2:
            self.props = value
        elif key == 3:
            self.constants = value
        else:
            raise IndexError

    def __repr__(self):
        quantities = "Quantities: " + str(list(self.quantities.values()))
        return quantities

    # Returns a compact representation of a state based on its quantities
    def compact(self):
        compact_string = ""
        for q in self.quantities.values():
            if q[1] > 0:
                derivative = '+'
            elif q[1] < 0:
                derivative = '-'
            else:
                derivative = '0'
            compact_string += str(q[0]) + " " + derivative + "\n"
        return compact_string

    def next(self):
        for s1 in update_actions(self):
            for s2 in update_magnitudes(s1):
                output = [s for s in update_dependencies(s2, self, [list(range(len(self[0]))), [s2]]) if
                          self.correct_action(s)]
                yield output

    # A point state is a state that contains a quantity with a point-valued magnitude and a non-zero derivative
    @property
    def is_point_state(self):
        for q in self.quantities.values():
            quantity_space = q.space()
            is_point = [m[1] for m in quantity_space if m[0] == q[0]][0]
            if q[1] != 0 and is_point:
                return True
        return False

    @property
    def correct_constant(self):
        for (v, i1, i2) in self.constants:
            q1 = self.quantities[i1]
            q2 = self.quantities[i2]
            if (q1[0] == v and q2[0] != v) or (q2[0] == v and q1[0] != v):
                return False
        return True

    # Prevents an immediate transition like 0 + -> + 0
    def correct_derivative(self, other):
        if self.is_point_state:
            for i, q in self.quantities.items():
                q2 = other[0][i]
                if all([q[0] == 0, q[1] != 0, q2[0] != 0, q2[1] == 0]):
                    return False
        return True

    def correct_action(self, other):
        if self.is_point_state:
            for i, q in self.quantities.items():
                if len(q[2].keys()) > 1:
                    q2 = other[0][i]
                    is_point = get_index(q.space(), q[0])[0]
                    if q[1] != q2[1] and not is_point:
                        return False
        return True


class Quantity:
    def __init__(self, name, quantity_space):
        self.name = name
        self.quantity_space = quantity_space
        self.magnitude = 0
        self.derivative = 0
        self.actions = dict([(0, [])])
        self.time = 0

    def __getitem__(self, key):
        if key == 0:
            return self.magnitude
        elif key == 1:
            return self.derivative
        elif key == 2:
            return self.actions
        else:
            raise IndexError

    def __setitem__(self, key, value):
        if key == 0:
            self.magnitude = value
        elif key == 1:
            self.derivative = value
        elif key == 2:
            self.actions = value
        else:
            raise IndexError

    def __repr__(self):
        return self.name + "(" + str(self.magnitude) + "," + str(self.derivative) + ")" + str(self.time)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__repr__() == other.__repr__()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def time(self):
        return self.time

    def get_cur_actions(self):
        return self.actions[self.time]

    def next_actions(self):
        return self.actions[self.time + 1]

    def next_time(self):
        self.time += 1

    def name(self):
        return self.name

    def space(self):
        return self.quantity_space

    @property
    def zero_index(self):
        return self.quantity_space.index((0, True))


def update_dependencies(state, old_s, updated_quantities):
    while updated_quantities[0]:
        update_derivatives(state, old_s[0], updated_quantities[0][0], updated_quantities)

    return [s for s in updated_quantities[1] if s.correct_constant and old_s.correct_derivative(s)]


def update_derivatives(state, old_qs, q_index, updated_quantities):
    old_derivative = old_qs[q_index][1]
    influences = state[1]
    props = state[2]

    q_influences = [(r, q1) for (r, q1, q2) in influences if q2 == q_index]
    q_props = [(r, q1) for (r, q1, q2) in props if q2 == q_index]

    new_states = []

    # First update all quantities that are proportional to q_index
    for r, q1_i in q_props:
        if q1_i in updated_quantities[0]:
            update_derivatives(state, old_qs, q1_i, updated_quantities)

    for s in updated_quantities[1]:
        all_derivatives = []
        for r1, q1_i in q_props:
            prop = s[0][q1_i][1] * r1
            if old_derivative * prop == -1:  # Ensures continuity
                all_derivatives.append(0)
            else:
                all_derivatives.append(prop)

        for r2, q2_i in q_influences:
            influence = get_influence((r2, s[0][q2_i]))  # get magnitude of influence
            all_derivatives.append(influence)

        new_derivatives = get_new_derivatives(all_derivatives, old_derivative)

        for d in new_derivatives:
            new_s = copy.deepcopy(s)
            if d != old_derivative:
                print("- Derivative update of "+new_s[0][q_index].name+": "+str(old_derivative)+" -> "+str(d))
            new_s[0][q_index][1] = d
            new_states.append(new_s)

    updated_quantities[0].remove(q_index)
    if new_states:
        updated_quantities[1] = new_states


def update_actions(state):
    quantities = state[0]
    all_new_quantities = []

    for q in quantities.values():
        actions = q.get_cur_actions()
        new_quantities = [q]
        for a in actions:
            new_q = copy.copy(q)
            new_q[1] = a
            new_quantities.append(new_q)
            if new_q.next_actions():
                new_q2 = copy.copy(new_q)
                new_q2.next_time()
                if new_q2 not in new_quantities:
                    print("- Time update of "+new_q2.name+" -> "+str(new_q2.time))
                    #print()
                    new_quantities.append(new_q2)
        all_new_quantities.append(new_quantities)

    return all_q_combinations(state, all_new_quantities)


def update_magnitudes(state):
    all_new_quantities = []
    quantities = state[0]

    for q in quantities.values():
        new_quantities = []
        new_magnitudes = get_magnitudes(q[1], q[0], q.space())

        for d, m in new_magnitudes:
            new_q = copy.copy(q)
            if m != q[0]:
                print("- Magnitude update of "+new_q.name+": "+str(q[0])+" -> "+str(m))
            new_q[0] = m
            new_q[1] = d
            new_quantities.append(new_q)

        all_new_quantities.append(new_quantities)

    # All possible combinations are tried
    return all_q_combinations(state, all_new_quantities)


def all_q_combinations(state, all_new_quantities):
    quantities = state[0]
    num_of_quantities = len(quantities)

    product_quantities = [q for q in list(it.product(*all_new_quantities)) if q != quantities]

    new_states = []
    for qs in product_quantities:
        new_s = copy.deepcopy(state)
        new_s[0] = dict(zip(range(num_of_quantities), qs))
        new_states.append(new_s)

    return new_states


def get_magnitudes(derivative, magnitude, quantity_space):
    if derivative == 0:
        return [(derivative, magnitude)]

    num_of_values = len(quantity_space)

    is_point, index = get_index(quantity_space, magnitude)

    # If magnitude is set to a non-existing value:
    if index < 0:
        raise AssertionError

    next_index = index + derivative

    if next_index < 0 or next_index > num_of_values - 1:
        if is_point:
            return [(0, magnitude)]
        else:
            return [(derivative, magnitude)]
    else:
        next_magnitude = quantity_space[next_index][0]
        magnitudes = [(derivative, next_magnitude)]

    # A quantity magnitude can remain the same if it is not a point value
    if not is_point:
        magnitudes.append((derivative, magnitude))

    return magnitudes


def get_influence(influence):
    rate, quantity = influence
    magnitude = quantity[0]
    magnitude_index = get_index(quantity.space(), magnitude)[1]
    if magnitude != 0:
        if magnitude_index > quantity.zero_index:  # Any value above the zero value is positive
            magnitude = 1
        else:
            magnitude = -1
        return rate * magnitude
    else:
        return 0


# Returns the possible new derivatives given a list of incoming influences
def get_new_derivatives(all_derivatives, old_derivative):
    zero = False
    if 0 in all_derivatives:
        all_derivatives = [x for x in all_derivatives if x != 0]
        zero = True

    num_of_derivatives = len(set(all_derivatives))

    if num_of_derivatives == 0:
        if zero:
            return [0]
        else:
            return []
    if num_of_derivatives == 1:
        if all_derivatives[0] * old_derivative != -1:  # Ensures continuity: + -> 0 -> -
            return [all_derivatives[0]]
        else:
            return [0]
    else:
        if old_derivative == 0:
            return [-1, 0, 1]  # ?
        else:
            return [0, old_derivative]


# Finds the index of a magnitude in a quantity space
def get_index(quantity_space, magnitude):
    for i, x in enumerate(quantity_space):
        if x[0] == magnitude:
            is_point = x[1]
            index = i
            return is_point, index
    raise LookupError


def flatten(seq, container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s, '__iter__'):
            flatten(s, container)
        else:
            container.append(s)
    return container


def create_graph(state, search_state=State(dict([]))):
    dot = Digraph(comment='QR State Diagram')
    dot.node("start", style="invisible")  # Incoming start arrow
    dot.edge("start", state.compact())
    edges = []
    path = [False, []]
    graph, n = simulate(dot, 0, state, search_state, path, [], edges)
    print()
    print("Number of unique nodes: " + str(len(flatten(graph))))
    print("Number of unique edges: " + str(len(edges)))
    dot.render('state-diagram', view=True)
    if len(search_state[0].keys()) > 0:
        show_path(path[1], [])


def simulate(dot, n, state, search_state, path, loopcheck, edgecheck):
    if n > 200:  # To prevent endless recursion
        return [], n
    if state.compact() == search_state.compact():
        path[0] = True
        path[1].append(str(state))
    elif not path[0]:
        path[1].append(str(state))

    graph = []
    compact_rep = state.compact()
    node_string = compact_rep + str(n)
    if state.is_point_state:
        dot.node(compact_rep, node_string)
    else:
        dot.node(compact_rep, node_string, shape='square')
    print(state)
    for s1 in state.next():
        for s2 in s1:
            compact_s2 = s2.compact()
            if s2[0] not in loopcheck:
                if (compact_rep, compact_s2) not in edgecheck and compact_rep != compact_s2:
                    edgecheck.append((compact_rep, compact_s2))
                    dot.edge(compact_rep, compact_s2)
                    graph.append(s2)
                loopcheck.append(s2[0])
                nodes, n = simulate(dot, n + 1, s2, search_state, path, loopcheck, edgecheck)
                graph.append(nodes)
            # Reflexive relations are left implicit, one edge per transition
            elif compact_s2 != compact_rep and (compact_rep, compact_s2) not in edgecheck:
                edgecheck.append((compact_rep, compact_s2))
                dot.edge(compact_rep, compact_s2)
    return graph, n


def show_path(path, seen):
    for p in path:
        if not p in seen:
            print(p)
            seen.append(p)


# Exogenous quantity behaviour. Behaviours are defined as dictionaries with a timestamp as key
# The end of a sequence is denoted by the empty list. Each timestamp is presumed to be an interval.
parabola = dict([(0, [0]), (1, [1]), (2, [0]), (3, [-1]), (4, [0]), (5, [])])
linear_in = dict([(0, [0]), (1, [1]), (2, [])])
linear_dec = dict([(0, [-1]), (1, [0]), (2, [])])
random = dict([(0, [-1, 0, 1]), (1, [])])

inflow = Quantity("Inflow", [(0, True), ('+', False)])
volume = Quantity("Volume", [(0, True), ('+', False), ('m', False)])
outflow = Quantity("Outflow", [(0, True), ('+', False), ('m', False)])
# height = Quantity("Height", [(0, True), ('+', False), ('m', True)])
# pressure = Quantity("Pressure", [(0, True), ('+', False), ('m', True)])

inflow[2] = parabola

sink = State(dict([(0, inflow), (1, volume), (2, outflow)]))
sink[1] = [(1, 0, 1), (-1, 2, 1)]
sink[2] = [(1, 1, 2)]
sink[3] = [(0, 1, 2), ('+', 1, 2), ('m', 1, 2)]
# sink[2] = [(1, 1, 3), (1, 3, 4), (1, 4, 2)]
# sink[3] = [(0, 1, 3), ('+', 1, 3), ('m', 1, 3), (0, 3, 4), ('+', 3, 4), ('m', 3, 4), (0, 4, 2), ('+', 4, 2), ('m', 4, 2)]


q1 = Quantity("Inflow", [(0, True), ('+', False)])
q2 = Quantity("Volume", [(0, True), ('+', False), ('m', False)])
q3 = Quantity("Outflow", [(0, True), ('+', False), ('m', False)])

q1[0] = '+'
q1[1] = 1
q2[0] = '+'
q2[1] = -1
q3[0] = '+'
q3[1] = -1

search = State(dict([(0, q1), (1, q2), (2, q3)]))

create_graph(sink, search_state = search)
