
# Introduction to the TTT Algorithm and Its Implementation for Inferring Input Grammars of Blackbox Programs in Python
This tutorial will cover the introduction to the TTT algorithm and its implementation for inferring input grammars of blackbox programs in Python. In many previous posts, I have discussed how to parse with, fuzz with, and manipulate regular and context-free grammars. However, in many cases, such grammars may be unavailable. If you are given a blackbox program, where the program indicates in some way that the input was accepted or not, what can we do to learn the actual input specification of the blackbox? In such cases, the best option is to try and learn the input specification. This particular research field which investigates how to learn the input specification of blackbox programs is called blackbox grammar inference. The TTT algorithm is a novel approach introduced by Malte Isberner, Falk Howar, and Bernhard Steffen in 2014. TTT addresses limitations of Angluin’s L* algorithm by eliminating redundant information, achieving optimal space complexity, and excelling in scenarios with complex systems. In this tutorial we’ll discuss how the TTT algorithm works with an example.
## What is Active Automata Learning?
Active automata learning involves constructing a DFA that represents a system’s behavior by querying it. The learner operates in the **Minimally Adequate Teacher (MAT)** framework. The teacher is "minimally adequate" because it provides just enough information for the learner to infer the correct DFA. The goal is to minimize the number of queries while constructing an accurate model. Algorithms like L* and TTT operate within this framework, differing in how they process queries and counterexamples to build the DFA.
The goal is to build a DFA that accepts the same language as the system with minimal queries. L* The seminal algorithm in this field uses an observation table to store query results, but its quadratic space complexity and redundant processing of counterexamples can be problematic. TTT overcomes these issues with a tree-based approach and clever counterexample analysis.
## Symbols
-  **Q** → Set of all finite states
-  **Σ (Sigma)** → Input alphabet (set of allowed symbols).
-  **δ (Delta)** → Transition function, which determines movement between states.
-  **q₀** → Start state (initial state).
-  **F** → Set of final (accepting) states. Multiple final states are possible.
-  **F ⊆ Q** → Final states are a subset of total states.
- **λ** → The acceptance function
## Key Definitions
To understand the TTT algorithm and its advantages, here are definitions of key terms used in this article:
-  **Membership Query (MQ)**: A query in the MAT framework where the learner asks the teacher whether a specific word \( w ∈ Σ* \) is accepted by the target DFA, receiving a boolean response (true if \( w\) is accepted, false otherwise).
-  **Equivalence Query (EQ)**: A query in the MAT framework where the learner proposes a hypothesis DFA and asks the teacher if it is equivalent to the target DFA. If not, the teacher provides a counterexample—a word \(w\) where the hypothesis and target DFA produce different outputs.
-  **Counterexample**: A word \( w ∈ Σ* \) provided by the teacher in response to an equivalence query, where the hypothesis DFA (H) and the target DFA (A) disagree, i.e., \( λ~H(w)~ ≠ λ~A(w)~ \).
-  **Discrimination Tree (DT)**: A binary tree used by TTT to organize state-distinguishing information. Inner nodes are labeled with discriminators (words that distinguish states by producing different outputs), and leaves represent states in the hypothesis DFA.
-  **Discriminator Finalization**: A process in TTT where temporary discriminators (added during counterexample processing) are replaced with shorter, final discriminators to eliminate redundancy and reduce the length of future queries.
-  **Canonical DFA**: A DFA ( A \) is canonical (i.e., minimal) if:
	-  **Reachability**: For all states \( q~A~ \), there exists a word \( u ∈ Σ* \) such that ( A[u] = q \) i.e., all states are reachable from the initial state.
	-  **Separability**: For all distinct states ( q ≠ q' ∈ q~A~ \), there exists a word \( v ∈ Σ* \) such that ( λ_A(q, v) ≠ λ_A(q', v) \) (i.e., all states are pairwisely separable, and (v) is called a separator). It is well-known that canonical DFAs are unique up to isomorphism.
## Setup
### Imports
```Python
from  __future__  import  annotations
from  pathlib  import  Path
import  sys
import  re
import  random
import  os
import  math
from  functools  import  lru_cache
from  inspect  import  currentframe,  getframeinfo,  signature
from  typing  import  Optional,  Protocol,  Pattern,  TYPE_CHECKING
```
### TTT Print
Special print is a decorator that wraps a function to prepend the calling file name and line number to its output.
```python
def  special_print(func):
	def  wrapped_func(*args, **kwargs):
		if  curr_frame  :=  currentframe():
			if  prev_frame  :=  curr_frame.f_back:
				frameinfo  =  getframeinfo(prev_frame)
				return  func(f"{frameinfo.filename}  {frameinfo.lineno}:",  *args,  **kwargs)
	return  func(args,  kwargs)
return  wrapped_func
```
Formats the arguments of a function call into a string representation
```Python
def  format_args(frame):
	"""Extract and format function arguments."""
	func_name  =  frame.f_code.co_name
	locals_  =  frame.f_locals
	code  =  frame.f_code
	# Get parameter names from co_varnames (first n entries are parameters)
	arg_count  =  code.co_argcount # Number of positional parameters
	param_names  =  code.co_varnames[:arg_count]  # Parameter names from code object
	# Filter locals to include only the defined parameters
	args_repr  =  ", ".join(f"{name}={locals_.get(name,  '?')}"  for  name  in  param_names  if  name  in  locals_)
	return  f"({args_repr})"  if  args_repr  else  "()"
```
A global trace function to log calls to non-dunder functions, including arguments and call location.
```Python
def  trace_calls(frame, event, arg):
	"""Global function call tracer for user-defined, non-dunder functions with arguments."""
	if  event  ==  "call":
		function_name  =  frame.f_code.co_name
		if  not  is_dunder_method(function_name):
		frameinfo  =  getframeinfo(frame)
		args_repr  =  format_args(frame)
		print(f"Function {function_name}{args_repr} called from {frameinfo.filename}:{frameinfo.lineno}")
	return  trace_calls
```
### Hypothesis
#### Hypothesis.add_state
Adds a new state to the hypothesis with a given access sequence, initializing its transitions based on the alphabet. The method creates a new `State` object with the provided `aseq`, adds it to `states`, and initializes `transitions` as a dictionary mapping each alphabet symbol to a `Transition` with the concatenated access sequence and the root node. All transitions are initially added to `open_transitions` as non-tree transitions, returning the new state.
```python
def add_state(self, aseq: str) -> State:
    state = State(self, aseq)
    self.states.add(state)
    state.transitions = {
        a: Transition(self, aseq + a, self.root_node) for a in self.alphabet
    }
    for t in state.transitions.values():
        # all trasitions are initially be nontree
        self.open_transitions.append(t)
    return state
```
#### Hypothesis.make_final
Marks a given state as a final state in the hypothesis DFA. The method checks if the `state` is in `states` using an assertion; if true, it adds the state to `final_states`. If the state is unknown, it raises a `ValueError`, ensuring only valid states are marked as final.
```python
def make_final(self, state: State) -> None:
    if state in self.states:
        self.final_states.add(state)
    else:
        raise ValueError("Unknown state passed")
```
#### Hypothesis.run
Executes a deterministic run of the hypothesis DFA on a string, following tree transitions to a target state. Starting from an optional `start` state (defaulting to `self.start`), it checks if the string `s` is empty, returning the current state if true. Otherwise, it retrieves the transition for the first symbol of `s`, asserts the target state exists, and recursively calls `run` with the remaining string and target state, raising a `ValueError` if a transition is unclosed.
```python
def run(self, s: str, start: Optional[State] = None) -> State:
    if start is None:
        start = self.start
    if s == "":
        return start
    t = start.transitions[s[0]]
    if t.target_state is not None:
        return self.run(s[1:], t.target_state)
    else:
        raise ValueError("Only call run when all transitions are closed")
```
#### Hypothesis.run_non_deterministic
Performs a non-deterministic run of the hypothesis, resolving open transitions by sifting through the discrimination tree. Starting from an optional `start` state, it prints debug information and checks if `s` is empty, returning the current state if true. For non-empty strings, it gets the transition for the first symbol, sifts the target node using `target_node.sift` if it’s not a leaf, updates the target node, and recursively calls itself with the remaining string and teacher, handling dynamic tree updates in TTT.
```python
def run_non_deterministic(self, s: str, teacher: Teacher, start: Optional[State] = None) -> State:
    print(f"Running hypothesis: start={start}, remaining str: {s}")
    if start is None:
        start = self.start
    if s == "":
        return start
    t = start.transitions[s[0]]
    if not t.target_node.is_leaf:
        new_target = t.target_node.sift(t.aseq, teacher)
        t.target_node = new_target
    return self.run_non_deterministic(s[1:], teacher, t.target_state)
```
#### Hypothesis.evaluate_non_deterministic
Evaluates a string’s acceptance in a non-deterministic hypothesis using the teacher for sifting. Using `run_non_deterministic` with an optional `start` state, it checks if the resulting state is in `final_states`, returning a boolean to indicate acceptance, supporting TTT’s adaptive learning process.
```python
def evaluate_non_deterministic(self, s: str, teacher: Teacher, start: Optional[State] = None):
    if start is None:
        start = self.start

    return self.run_non_deterministic(s, teacher, start=start) in self.final_states
```
### Node Class
#### replace_with_final: 
This Replaces the current node with another node in place, finalizing a temporary node during TTT tree refinement. The method **prints a debug message and asserts** that both nodes have the same `is_leaf` status. It updates `is_leaf`, sets `_is_temporary` to `False`, clears `block`, and copies `_children`, `_state`, `_discriminator`, and `_incoming_non_tree` from the `node` parameter. This finalises temporary nodes in TTT, aligning with discriminator finalisation to optimise the tree.
 ```Python
 def replace_with_final(self, node: Node) -> None:
     # replace the node with another in place
     print(f"Replacing {self} with {node}")
     assert self.is_leaf == node.is_leaf

     self.is_leaf = node.is_leaf
     self._is_temporary = False
     self.block = None

     self._children = node._children
     self._state = node._state
     self._discriminator = node._discriminator
     self._incoming_non_tree = node._incoming_non_tree
 ```
#### make_leaf (classmethod)
This creates a new leaf node to represent a state in the TTT hypothesis DFA. As a class method, it calls the `Node` constructor with `is_leaf=True` and `children=(None, None)`, resulting in a node with no children or discriminator. This node serves as a placeholder for a state (e.g., \( q_\varepsilon \)) in TTT, to be linked later during tree expansion.
 ```Python
 @classmethod
 def make_leaf(cls) -> Node:
     return Node(True, (None, None))
 ```
#### make_inner (classmethod)
make_inner method creates a new inner node with a specified discriminator for TTT tree splits. The class method calls the `Node` constructor with `is_leaf=False`, the provided `children` tuple (defaulting to `(None, None)`), and the `discriminator` string. It then sets the `parent` of any non-`None` children to the new node, establishing the binary tree structure used for TTT splits (e.g., with \( aa \)).
```python
@classmethod
def make_inner(
    cls,
    discriminator: str,
    children: tuple[Optional[Node], Optional[Node]] = (None, None),
) -> Node:
    new_node = Node(False, children, discriminator=discriminator)
    for child in children:
        if child:
            child.parent = new_node
    return new_node
```
#### print_tree
Outputs the TTT discrimination tree rooted at the current node in a level-by-level format for visualisation. The method uses `child` (0, 1, or -1) to set an arrow (`->` or `=>`) and `level` for indentation. For leaf nodes, it prints with spacing; for inner nodes, it recursively calls itself on the left child, prints the current node, and then the right child, asserting both children exist. This mirrors TTT’s binary structure.
```python
def print_tree(self, child: int = -1, level: int = 0):
    if child == 1:
        arrow = "=>"
    elif child == 0:
        arrow = "->"
    else:
        arrow = "->"
    if self.is_leaf:
        print(f"{' ' * 4 * level}{arrow} {self}")
    else:
        assert self.children[0]
        assert self.children[1]
        self.children[0].print_tree(0, level+1)
        print(f"{' ' * 4 * level}{arrow} {self}")
        self.children[1].print_tree(1, level+1)
```
#### states
This Generates all `State` objects associated with leaf nodes in the TTT hypothesis DFA. Using `__iter__` to traverse the subtree, it yields the `state` attribute for each node that is a leaf and has a non-`None` state, providing the set of states e.g., \( q(a), q(aa) \) for the TTT-learned DFA.
```python
def states(self) -> Generator[State, None, None]:
    for node in self:
        if node.is_leaf and node.state is not None:
            yield node.state
```
#### is_temporary
Checks if the node is a temporary inner node during TTT tree construction. The property asserts the node is not a leaf and returns `_is_temporary`, which is set during initialization based on the presence of a discriminator. This tracks temporary nodes added during counterexample processing.
```python
@property
def is_temporary(self) -> bool:
    assert not self.is_leaf
    return self._is_temporary
```
#### parent
Returns the parent node in the TTT discrimination tree hierarchy. The property returns the `_parent` attribute, set during node creation or via the setter, maintaining the tree’s parent-child relationships for operations like LCA computation.
```python
@property
def parent(self) -> Optional[Node]:
    return self._parent
```
#### children
Returns the children of an inner node, enforcing the TTT binary tree structure. The property asserts the node is not a leaf and returns the `_children` tuple, containing two optional nodes for the left and right children, upholding the binary nature of the TTT tree.
```python
@property
def children(self) -> tuple[Optional[Node], Optional[Node]]:
    assert not self.is_leaf
    return self._children
```
#### state
Returns the `State` object associated with a leaf node in the TTT hypothesis DFA. The property asserts the node is a leaf and returns the `_state` attribute, set via the setter, linking the node to a state (e.g., \( q_\varepsilon \)) in the TTT-learned DFA.
```python
@property
def state(self) -> Optional[State]:
    assert self.is_leaf
    return self._state
```
#### discriminator
Returns the discriminator of an inner node used for TTT tree splitting. The property asserts the node is not a leaf and `_discriminator` is not `None`, then returns the string value (e.g., \( aa \)), which defines the split criterion in the TTT tree.
```python
@property
def discriminator(self):
    assert not self.is_leaf
    assert self._discriminator is not None
    return self._discriminator
```
#### parent_value
Returns the membership query outcome (0 or 1) relative to the parent for TTT state distinction. The property raises an error if `parent` is `None`, returns `False` if the node is the left child (MQ=0), `True` if the right child (MQ=1), and handles invalid cases with an error, supporting signature computation.
```python
@property
def parent_value(self) -> bool:
    if self.parent is None:
        raise ValueError(f"{self} has no parent")
    if self == self.parent.children[0]:
        return False
    if self == self.parent.children[1]:
        return True
    raise ValueError(f"{self} is not the child of its parent")
```

#### signature
Returns the signature (discriminator-value pairs) for state distinction in the TTT algorithm. The property returns an empty list if there is no parent; otherwise, it creates a list with `(self.parent.discriminator, self.parent_value)` and prepends the parent’s signature recursively, identifying unique states in TTT.
```python
@property
def signature(self) -> list[tuple[str, bool]]:
    if self.parent is None:
        return []
    else:
        return [(self.parent.discriminator, self.parent_value), *self.parent.signature]
```
#### incoming_non_tree
Returns the set of non-tree transitions to the node in the TTT hypothesis DFA. The property returns the `_incoming_non_tree` set, which stores `Transition` objects for edges outside the spanning tree, supporting complex DFA structures in TTT.
```python
@property
def incoming_non_tree(self) -> set[Transition]:
    return self._incoming_non_tree
```
#### link
Links a `State` object to the node, establishing a bidirectional connection in the TTT framework. The method sets the node’s `state` with the `state` parameter and sets `state.node` to `self`, creating a bidirectional link. Commented code for merging transitions is disabled, offering flexibility.
```python
def link(self, state: State) -> None:
    self.state = state
    if state.node:
	    self.incoming_non_tree |= state.node.incoming_non_tree
	    self.incoming_tree |= state.node.incoming_tree
    state.node = self
```
#### split_leaf
Splits a leaf node into an inner node with two leaf children for TTT counterexample processing. The method asserts the node is a leaf, sets `is_leaf` to `False`, `_is_temporary` to `True`, clears `_state`, and sets `_discriminator`. It creates two leaf nodes, sets their `parent` to `self`, updates `_children`, and returns the children tuple, implementing TTT’s split for counterexamples.
```python
def split_leaf(self, discriminator: str) -> tuple[Node, Node]:
    assert self.is_leaf
    self.is_leaf = False
    self._is_temporary = True
    self._state = None
    self._discriminator = discriminator

    children = (Node.make_leaf(), Node.make_leaf())
    for child in children:
        child.parent = self
    self._children = children

    return children
```

### Transition Class
#### Transition.is_tree
Checks whether the transition is a tree transition, indicating it points to a finalized state. The method returns `True` if `_target_state` is not `None` (indicating a closed tree transition), and `False` otherwise (indicating an open non-tree transition). This distinction is central to TTT’s spanning-tree hypothesis.
```python
def is_tree(self) -> bool:
    return self._target_state is not None
```
#### Transition.target_node
Returns the target node of the transition, adjusting based on whether it is a tree or non-tree transition. The property checks if `_target_state` exists; if true (tree transition), it returns the state’s `node`, otherwise it returns `_target_node`. This ensures the correct node is accessed depending on the transition type in the TTT tree.
```python
@property
def target_node(self) -> Node:
    if self._target_state:
        # is tree transition
        return self._target_state.node
    return self._target_node
```
#### Transition.target_state
Returns the target state of the transition, falling back to the node’s state if not a tree transition. The property returns `_target_state` if set, otherwise it returns the `state` of `_target_node`, allowing access to the associated state when available in the TTT hypothesis.
```python
@property
def target_state(self) -> Optional[State]:
    if self._target_state is None:
        return self._target_node.state
    else:
        return self._target_state
```
#### Transition.make_tree
Converts a non-tree transition into a tree transition by adding a new state and linking it to the target node. The method checks if the transition is already a tree transition, raising a `ValueError` if true. Otherwise, it adds a new state with the transition’s `aseq` using `hypothesis.add_state`, removes the transition from `incoming_non_tree`, sets the target state, and makes the state final if the node’s signature includes `("", True)`. It then links the node to the state and returns the new state, finalizing the transition in TTT.
```python
def make_tree(self, node: Node) -> State:
    if self.is_tree():
        assert self._target_state
        state = self._target_state
        raise ValueError("transition is already a tree transition")
    else:
        state = self.hypothesis.add_state(self.aseq)
        self.target_node.incoming_non_tree.remove(self)
        self.target_state = state
        if ("", True) in node.signature:
            self.hypothesis.make_final(state)
    node.link(state)
    return state
```
## THE TTT
The full implementation for TTT algorithm can be found at [Replit](https://replit.com/@ineerajrajeev/TTT-Algorithm)
### Step 1: Initialization

The algorithm begins with a single-state hypothesis and a minimal discrimination tree.

- **Discrimination Tree 1**: Structure: A single node representing the empty string (epsilon) leads to state `q0`.
```python
# Initial setup
alphabet = ["a", "b"]
teacher = SimpleDFATeacher(alphabet, pattern=r"b*a(b*ab*ab*ab*)+b*")  # Regex for (4i + 3) 'a's
hypothesis = Hypothesis(alphabet)
root = Node.make_inner("", children=(Node.make_leaf(), Node.make_leaf()))  # Root with "" discriminator
hypothesis.root_node = root
q0 = hypothesis.add_state("")  # Initial state
root.children[0].link(q0)  # Link q0 to t0
```
  ```mermaid
  graph TD
      A["Node<>"] --> B["Node (t0, q0, non-final)"]
  ```
- **DFA 1**:
  - **States**: `{q0}`
  - **Start State**: `q0`
  - **Final States**: `{}`
  - **Transitions**:
    - `q0 --a--> q0`
    - `q0 --b--> q0`
  ```mermaid
  graph TD
      A[q0] -->|a,b| A
  ```

This initial DFA has `q0` self-looping on both `a` and `b`, rejecting all strings since there are no final states.

### Step 2: First Refinement

The hypothesis is refined by introducing a new state and discriminator based on a counterexample.

- **Counterexample**: `"aaa"` (3 'a's, accepted by the target, rejected by DFA 1).
```python
def process_counterexample(hypothesis, counterexample, teacher):
    for i in range(len(counterexample) + 1):
        u = counterexample[:i]
        a = counterexample[i] if i < len(counterexample) else ""
        v = counterexample[i+1:] if i < len(counterexample) else ""
        current_state = hypothesis.run(u)
        if a and v:
            next_state = current_state.transitions[a].target_state
            if next_state and teacher.is_member(u + a + v) != hypothesis.evaluate(u + a + v):
                leaf_node = next_state.node
                leaf_node.split_leaf(discriminator=v)
                new_target = hypothesis.root_node.sift(u + a, teacher)
                current_state.transitions[a].target_node = new_target
                if teacher.is_member(u + a + v):
                    hypothesis.make_final(new_target.state)
                break
        elif not a and v == "":
            if teacher.is_member(u) != hypothesis.evaluate(u):
                leaf_node = current_state.node
                leaf_node.split_leaf(discriminator="")
                new_target = hypothesis.root_node.sift(u, teacher)
                if teacher.is_member(u):
                    hypothesis.make_final(new_target.state)
                break
    close_open_transitions(hypothesis, teacher)

process_counterexample(hypothesis, "aaa", teacher)
```
#### Hypothesis.evaluate
Evaluates whether a string is accepted by the hypothesis DFA, based on reaching a final state. Using `run` with an optional `start` state (defaulting to `self.start`), it checks if the resulting state is in `final_states`, returning a boolean to indicate acceptance.
```python
def evaluate(self, s: str, start: Optional[State] = None):
    if start is None:
        start = self.start
    return self.run(s, start=start) in self.final_states
```
- **Equivalence Query**:
  - `teacher.is_equivalent(hypothesis)` checks if the hypothesis matches the target language.
  - Current hypothesis (no final states) rejects all strings, but the target accepts `"aaa"` (3 'a's).
  - Counterexample: `"aaa"` (accepted by target, rejected by hypothesis).

- **Process**:
  - Sift `"a"`: `teacher.is_member("a")` → `False` (1 'a' ≠ 3 mod 4).

-  **Sifting**:
Sifting refers to the process of traversing the discrimination tree to classify a given word or prefix into a corresponding leaf node, leveraging membership queries (MQs) to guide the traversal. This process is critical for mapping access sequences to states in the hypothesis DFA and refining the tree based on counterexamples.
- Sift `""`: `teacher.is_member("")` returns `False` (0 'a's, not `(4i + 3)`), so `q0` is non-accepting.
- Transitions from `q0` for `a` and `b` are open, pointing to the root.
```python
# Sifting functions extracted for reference
def  sift(node: Node, s: str, teacher: Teacher) -> Node:
	"""
	Traverses the TTT discrimination tree to classify a word into a corresponding leaf node.
	Args:
	node (Node): The current node in the discrimination tree.
	s (str): The input string or prefix to classify.
	teacher (Teacher): The teacher object providing membership query responses.
	Returns:
	Node: The leaf node corresponding to the word's access sequence.
	Note:
	This function follows the TTT state classification mechanism, using membership queries
	to navigate the binary tree until a leaf is reached.
	"""
	if node.is_leaf:
		return node # Return the leaf node if already at a leaf
	# Compute the membership query outcome for the current discriminator
	subtree =  int(teacher.is_member(s + node.discriminator))
	child = node.children[subtree]
	# Assert the child exists to ensure tree integrity
	assert child is  not None,  "Child node should not be None during sift"
	# Recursively sift down the appropriate child
	return  sift(child, s, teacher)
```
  - Add state `q1` with access sequence `"a"`.
  - Use discriminator `"aabbb"` to distinguish `q0` from `q1`:
    - `q0 · "aabbb" = "aabbb"` (2 'a's) → `False`.
    - `q1 · "aabbb" = "aaabbb"` (3 'a's) → `True`.

- **Discrimination Tree 2**:
  - Root splits with discriminator `"aabbb"`.
  ```mermaid
  graph TD
      A["Node<aabbb>"] --> B["Node (t0, q0, non-final)"]
      A --> C["Node (q1, non-final)"]
  ```
#### Hypothesis.to_dfa
Converts the hypothesis into a formal DFA object for further analysis or export. It creates a `DFA` object, sets the start state ID, populates `states` and `final` sets with state IDs, and builds a transition dictionary mapping state IDs and symbols to target state IDs. It then closes the DFA with a sink state using the alphabet and returns the result.
```python
def to_dfa(self) -> DFA:
    dfa = DFA()
    dfa.start = self.start.id
    dfa.states = set(map(lambda state: state.id, self.states))
    dfa.final = set(map(lambda state: state.id, self.final_states))
    dfa.next_state = len(dfa.states)
    for h_state in self.states:
        d_state = h_state.id
        dfa.transitions[d_state] = {}
        for a, transition in h_state.transitions.items():
            assert transition.target_state  # otherwise not a DFA
            dfa.transitions[d_state][a] = transition.target_state.id
    dfa.close_with_sink(list(self.alphabet))
    return dfa
```

- **DFA 2**:
  - **States**: `{q0, q1}`
  - **Start State**: `q0`
  - **Final States**: `{}`
  - **Transitions**:
    - `q0 --a--> q1`
    - `q0 --b--> q0`
    - `q1 --a--> q0`
    - `q1 --b--> q1`
  ```mermaid
  graph TD
      A[q0] -->|a| B[q1]
      A -->|b| A
      B -->|a| A
      B -->|b| B
  ```
#### lca (classmethod)
Finds the lowest common ancestor (LCA) of a list of nodes for merge points in TTT state equivalence checks. It calculates the minimum depth among `nodes` using a `map` function, collects nodes at this depth in a set by tracing up via `parent` (with error handling for null parents), and iteratively moves up until one node remains. It prints debug information and returns the LCA, supporting TTT’s tree refinement.
```python
@classmethod
def lca(cls, nodes: list[Node]) -> Node:
    if not nodes:
        raise ValueError("Cannot compute LCA of an empty node list")
    min_depth = min(map(lambda node: node.depth, nodes))
    nodes_in_layer: set[Node] = set()
    for node in nodes:
        current = node
        while current.depth > min_depth:
            if current.parent is None:
                raise ValueError(f"Node {current} has no parent at depth {current.depth}")
            current = current.parent
        nodes_in_layer.add(current)
    while len(nodes_in_layer) > 1:
        nodes_in_layer = {node.parent for node in nodes_in_layer if node.parent is not None}
        if not nodes_in_layer:
            raise ValueError(f"No common ancestor found for nodes {nodes}")
    if not nodes_in_layer:
        raise ValueError(f"LCA of {nodes} couldn't be computed")
    return nodes_in_layer.pop()
```
This DFA switches states on `a` and self-loops on `b`, but still rejects all strings, requiring further refinement.
### Step 3: Second Refinement
A new counterexample drives the addition of states `q2` and `q3`.
- **Counterexample**: `"aaa"` (reused, as DFA 2 rejects it: `q0 --a--> q1 --a--> q0 --a--> q1`, non-final).
- **RS Decomposition**:
The rs_eager_search function performs a **binary search** to find the index i in a counterexample string where the hypothesis DFA and the target DFA produce different outputs for a specific condition, defined by the function alpha. This index represents the point of divergence, which is critical for the RS decomposition process in the TTT algorithm. RS decomposition breaks a counterexample ww into parts uu, aa, and vv (where w=uavw=uav) to pinpoint where the hypothesis fails, allowing targeted refinement of the DFA or discrimination tree.
```Python
def  rs_eager_search(self, alpha: Callable[[int],  bool], high: int, low: int  =  0) -> int:
	"""
	Performs an eager Rivest-Schapire search to find the exact index i where
	alpha(i) != alpha(i+1), indicating the divergence point in the counterexample.
	Uses memoization to cache alpha results for efficiency.
	Args:
	alpha: A function that checks if the hypothesis and target agree at index i.
	high: The upper bound of the search.
	low: The lower bound of the search (default 0).
	Returns:
	The index i where alpha(i) != alpha(i+1).
	"""
	def  beta(i: int) -> int:
		# Check cache for alpha(i) and alpha(i+1)
		if  i  not  in  self.alpha_cache:
			self.alpha_cache[i]  =  alpha(i)
		if  i  +  1  not  in  self.alpha_cache:
			self.alpha_cache[i  +  1]  =  alpha(i  +  1)
		return  self.alpha_cache[i]  +  self.alpha_cache[i  +  1]
	while  high  >  low:
		mid  =  (low  +  high)  //  2
		if  beta(mid)  ==  1: # alpha(mid) != alpha(mid+1)
			return  mid
		elif  beta(mid)  ==  0: # beta(mid+1) <= 1
			low  =  mid  +  1
		else: # beta(mid - 1) >= 1
			high  =  mid  -  1
	return  low
```
  - \( w = "aaa" \).
  - \( u = "aa" \), \( a = "a" \), \( v = "" \):
    - `q0 --a--> q1 --a--> q0`.
    - `q0 · ""` → `False`, but `"aaa"` → `True`.
  - Add `q3` (access sequence `"aaa"`, final) and `q2` (access sequence `"aa"`).
  - Discriminators `"aabbb"` and `"abbb"` refine the tree:
    - `q3 · "aabbb" = "aaaaabbb"` (5 'a's) → `False`.
    - `q1 · "aabbb" = "aaabbb"` (3 'a's) → `True`.
    - `q0 · "abbb" = "abbb"` (1 'a') → `False`.
    - `q2 · "abbb" = "aaabbb"` (3 'a's) → `True`.
- **Discrimination Tree 3**:
  - Root splits to `q3`, then `"aabbb"` and `"abbb"` distinguish others.
  ```mermaid
  graph TD
      A["Node<>"] --> B["Node<aabbb>"]
      B --> C["Node (q0, non-final)"]
      B --> D["Node<abbb>"]
      D --> E["Node (q1, non-final)"]
      D --> F["Node (q2, non-final)"]
      A --> G["Node (q3, final)"]
  ```

- **DFA 3**:
  - **States**: `{q0, q1, q2, q3}`
  - **-transition**: `q0`
  - **Final States**: `{q3}`
  - **Transitions**:
    - `q0 --a--> q1`
    - `q0 --b--> q0`
    - `q1 --a--> q2`
    - `q1 --b--> q1`
    - `q2 --a--> q3`
    - `q2 --b--> q0`
    - `q3 --a--> q0`
    - `q3 --b--> q0`
  ```mermaid
  graph TD
      A[q0] -->|a| B[q1]
      A -->|b| A
      B -->|a| C[q2]
      B -->|b| B
      C -->|a| D[q3]
      C -->|b| A
      D -->|a| A
      D -->|b| A
      class D final
  ```
This DFA now correctly handles strings up to 3 'a's, with `q3` as the accepting state.
### Step 4: Final Refinement
The hypothesis is tested and stabilized to match the target DFA.
- **Counterexample**: `"aaaaaaa"` (7 'a's, accepted, but needs verification).
- **Validation**:
  - `"aaaaaaa"`: `q0 --a--> q1 --a--> q2 --a--> q3 --a--> q0 --a--> q1 --a--> q2 --a--> q3` → `True` (correct).
  - `"aaaaaa"`: `q0 --a--> q1 --a--> q2 --a--> q3 --a--> q0 --a--> q1 --a--> q2` → `False` (correct).
- **Adjustment**:
  - Discriminator `"a"` ensures `q2` and `q3` are distinct from `q0` and `q1`.
- **Discrimination Tree 4**:
  - Final tree refines states with `"a"` and `"aabbb"`.
  ```mermaid
  graph TD
      A["Node<>"] --> B["Node<a>"]
      B --> C["Node (t0, q0, non-final)"]
      B --> D["Node<aabbb>"]
      D --> E["Node (q1, non-final)"]
      D --> F["Node (q2, non-final)"]
      A --> G["Node (q3, final)"]
  ```
- **DFA 4**:
  - **States**: `{q0, q1, q2, q3}`
  - **Start State**: `q0`
  - **Final States**: `{q3}`
  - **Transitions**:
    - `q0 --a--> q1`
    - `q0 --b--> q0`
    - `q1 --a--> q2`
    - `q1 --b--> q1`
    - `q2 --a--> q3`
    - `q2 --b--> q2`
    - `q3 --a--> q0`
    - `q3 --b--> q3`
  ```mermaid
  graph TD
      A[q0] -->|a| B[q1]
      A -->|b| A
      B -->|a| C[q2]
      B -->|b| B
      C -->|a| D[q3]
      C -->|b| C
      D -->|a| A
      D -->|b| D
      class D final
  ```
This DFA accurately represents the target language, with states corresponding to remainders modulo 4: `q0` (0), `q1` (1), `q2` (2), `q3` (3, accepting).
### Step 5: Validation
- The final hypothesis correctly accepts strings with `(4i + 3)` 'a's (e.g., `"aaa"`, `"aaaaaaa"`, `"bbbaaabbb"`) and rejects others (e.g., `"aaaa"`, `"aaaaaa"`).
## Summary
The TTT algorithm represents a significant improvement over L* for learning regular languages:
### Key Advantages of TTT:
1.  **Linear Space Complexity**: TTT uses O(n) space compared to L*'s O(n²)
2.  **No Redundancy**: Each discriminator appears at most once in the tree
3.  **Direct Counterexample Processing**: RS decomposition directly identifies where to refine
4.  **Efficient State Management**: States are managed through tree leaves
### When to Use TTT:
- Learning regular languages from black-box systems
- When space efficiency is important
- When you need to minimize redundant queries
- For protocol inference and verification tasks
#### References
- Isberner, M., Howar, F., & Steffen, B. (2014). _The TTT Algorithm: A Redundancy-Free Approach to Active Automata Learning_. Springer, Cham. https://doi.org/10.1007/978-3-319-11164-3_26
- Angluin, D. (1987). _Learning Regular Sets from Queries and Counterexamples_. Information and Computation, 75(2), 87–106.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM3MzU0NTIzNywtMTc4MTAxOTY3N119
-->