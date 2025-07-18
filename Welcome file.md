
---


---

<h1 id="introduction-to-the-ttt-algorithm-and-its-implementation-for-inferring-input-grammars-of-blackbox-programs-in-python">Introduction to the TTT Algorithm and Its Implementation for Inferring Input Grammars of Blackbox Programs in Python</h1>
<p>This tutorial will cover the introduction to the TTT algorithm and its implementation for inferring input grammars of blackbox programs in Python. In many previous posts, I have discussed how to parse with, fuzz with, and manipulate regular and context-free grammars. However, in many cases, such grammars may be unavailable. If you are given a blackbox program, where the program indicates in some way that the input was accepted or not, what can we do to learn the actual input specification of the blackbox? In such cases, the best option is to try and learn the input specification. This particular research field which investigates how to learn the input specification of blackbox programs is called blackbox grammar inference. The TTT algorithm is a novel approach introduced by Malte Isberner, Falk Howar, and Bernhard Steffen in 2014. TTT addresses limitations of Angluin’s L* algorithm by eliminating redundant information, achieving optimal space complexity, and excelling in scenarios with complex systems. In this tutorial we’ll discuss how the TTT algorithm works with an example.</p>
<h2 id="what-is-active-automata-learning">What is Active Automata Learning?</h2>
<p>Active automata learning involves constructing a DFA that represents a system’s behavior by querying it. The learner operates in the <strong>Minimally Adequate Teacher (MAT)</strong> framework. The teacher is “minimally adequate” because it provides just enough information for the learner to infer the correct DFA. The goal is to minimize the number of queries while constructing an accurate model. Algorithms like L* and TTT operate within this framework, differing in how they process queries and counterexamples to build the DFA.<br>
The goal is to build a DFA that accepts the same language as the system with minimal queries. L* The seminal algorithm in this field uses an observation table to store query results, but its quadratic space complexity and redundant processing of counterexamples can be problematic. TTT overcomes these issues with a tree-based approach and clever counterexample analysis.</p>
<h2 id="symbols">Symbols</h2>
<ul>
<li><strong>Q</strong> → Set of all finite states</li>
<li><strong>Σ (Sigma)</strong> → Input alphabet (set of allowed symbols).</li>
<li><strong>δ (Delta)</strong> → Transition function, which determines movement between states.</li>
<li><strong>q₀</strong> → Start state (initial state).</li>
<li><strong>F</strong> → Set of final (accepting) states. Multiple final states are possible.</li>
<li><strong>F ⊆ Q</strong> → Final states are a subset of total states.</li>
<li><strong>λ</strong> → The acceptance function</li>
</ul>
<h2 id="key-definitions">Key Definitions</h2>
<p>To understand the TTT algorithm and its advantages, here are definitions of key terms used in this article:</p>
<ul>
<li><strong>Membership Query (MQ)</strong>: A query in the MAT framework where the learner asks the teacher whether a specific word ( w ∈ Σ* ) is accepted by the target DFA, receiving a boolean response (true if ( w) is accepted, false otherwise).</li>
<li><strong>Equivalence Query (EQ)</strong>: A query in the MAT framework where the learner proposes a hypothesis DFA and asks the teacher if it is equivalent to the target DFA. If not, the teacher provides a counterexample—a word (w) where the hypothesis and target DFA produce different outputs.</li>
<li><strong>Counterexample</strong>: A word ( w ∈ Σ* ) provided by the teacher in response to an equivalence query, where the hypothesis DFA (H) and the target DFA (A) disagree, i.e., ( λ<sub>H(w)</sub> ≠ λ<sub>A(w)</sub> ).</li>
<li><strong>Discrimination Tree (DT)</strong>: A binary tree used by TTT to organize state-distinguishing information. Inner nodes are labeled with discriminators (words that distinguish states by producing different outputs), and leaves represent states in the hypothesis DFA.</li>
<li><strong>Discriminator Finalization</strong>: A process in TTT where temporary discriminators (added during counterexample processing) are replaced with shorter, final discriminators to eliminate redundancy and reduce the length of future queries.</li>
<li><strong>Canonical DFA</strong>: A DFA ( A ) is canonical (i.e., minimal) if:
<ul>
<li><strong>Reachability</strong>: For all states ( q<sub>A</sub> ), there exists a word ( u ∈ Σ* ) such that ( A[u] = q ) i.e., all states are reachable from the initial state.</li>
<li><strong>Separability</strong>: For all distinct states ( q ≠ q’ ∈ q<sub>A</sub> ), there exists a word ( v ∈ Σ* ) such that ( λ_A(q, v) ≠ λ_A(q’, v) ) (i.e., all states are pairwisely separable, and (v) is called a separator). It is well-known that canonical DFAs are unique up to isomorphism.</li>
</ul>
</li>
</ul>
<h2 id="setup">Setup</h2>
<h3 id="imports">Imports</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span>  __future__  <span class="token keyword">import</span>  annotations
<span class="token keyword">from</span>  pathlib  <span class="token keyword">import</span>  Path
<span class="token keyword">import</span>  sys
<span class="token keyword">import</span>  re
<span class="token keyword">import</span>  random
<span class="token keyword">import</span>  os
<span class="token keyword">import</span>  math
<span class="token keyword">from</span>  functools  <span class="token keyword">import</span>  lru_cache
<span class="token keyword">from</span>  inspect  <span class="token keyword">import</span>  currentframe<span class="token punctuation">,</span>  getframeinfo<span class="token punctuation">,</span>  signature
<span class="token keyword">from</span>  typing  <span class="token keyword">import</span>  Optional<span class="token punctuation">,</span>  Protocol<span class="token punctuation">,</span>  Pattern<span class="token punctuation">,</span>  TYPE_CHECKING
</code></pre>
<h3 id="ttt-print">TTT Print</h3>
<p>Special print is a decorator that wraps a function to prepend the calling file name and line number to its output.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span>  <span class="token function">special_print</span><span class="token punctuation">(</span>func<span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token keyword">def</span>  <span class="token function">wrapped_func</span><span class="token punctuation">(</span><span class="token operator">*</span>args<span class="token punctuation">,</span> <span class="token operator">**</span>kwargs<span class="token punctuation">)</span><span class="token punctuation">:</span>
		<span class="token keyword">if</span>  curr_frame  <span class="token punctuation">:</span><span class="token operator">=</span>  currentframe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
			<span class="token keyword">if</span>  prev_frame  <span class="token punctuation">:</span><span class="token operator">=</span>  curr_frame<span class="token punctuation">.</span>f_back<span class="token punctuation">:</span>
				frameinfo  <span class="token operator">=</span>  getframeinfo<span class="token punctuation">(</span>prev_frame<span class="token punctuation">)</span>
				<span class="token keyword">return</span>  func<span class="token punctuation">(</span>f<span class="token string">"{frameinfo.filename}  {frameinfo.lineno}:"</span><span class="token punctuation">,</span>  <span class="token operator">*</span>args<span class="token punctuation">,</span>  <span class="token operator">**</span>kwargs<span class="token punctuation">)</span>
	<span class="token keyword">return</span>  func<span class="token punctuation">(</span>args<span class="token punctuation">,</span>  kwargs<span class="token punctuation">)</span>
<span class="token keyword">return</span>  wrapped_func
</code></pre>
<p>Formats the arguments of a function call into a string representation</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span>  <span class="token function">format_args</span><span class="token punctuation">(</span>frame<span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""Extract and format function arguments."""</span>
	func_name  <span class="token operator">=</span>  frame<span class="token punctuation">.</span>f_code<span class="token punctuation">.</span>co_name
	locals_  <span class="token operator">=</span>  frame<span class="token punctuation">.</span>f_locals
	code  <span class="token operator">=</span>  frame<span class="token punctuation">.</span>f_code
	<span class="token comment"># Get parameter names from co_varnames (first n entries are parameters)</span>
	arg_count  <span class="token operator">=</span>  code<span class="token punctuation">.</span>co_argcount <span class="token comment"># Number of positional parameters</span>
	param_names  <span class="token operator">=</span>  code<span class="token punctuation">.</span>co_varnames<span class="token punctuation">[</span><span class="token punctuation">:</span>arg_count<span class="token punctuation">]</span>  <span class="token comment"># Parameter names from code object</span>
	<span class="token comment"># Filter locals to include only the defined parameters</span>
	args_repr  <span class="token operator">=</span>  <span class="token string">", "</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>f<span class="token string">"{name}={locals_.get(name,  '?')}"</span>  <span class="token keyword">for</span>  name  <span class="token keyword">in</span>  param_names  <span class="token keyword">if</span>  name  <span class="token keyword">in</span>  locals_<span class="token punctuation">)</span>
	<span class="token keyword">return</span>  f<span class="token string">"({args_repr})"</span>  <span class="token keyword">if</span>  args_repr  <span class="token keyword">else</span>  <span class="token string">"()"</span>
</code></pre>
<p>A global trace function to log calls to non-dunder functions, including arguments and call location.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span>  <span class="token function">trace_calls</span><span class="token punctuation">(</span>frame<span class="token punctuation">,</span> event<span class="token punctuation">,</span> arg<span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""Global function call tracer for user-defined, non-dunder functions with arguments."""</span>
	<span class="token keyword">if</span>  event  <span class="token operator">==</span>  <span class="token string">"call"</span><span class="token punctuation">:</span>
		function_name  <span class="token operator">=</span>  frame<span class="token punctuation">.</span>f_code<span class="token punctuation">.</span>co_name
		<span class="token keyword">if</span>  <span class="token operator">not</span>  is_dunder_method<span class="token punctuation">(</span>function_name<span class="token punctuation">)</span><span class="token punctuation">:</span>
		frameinfo  <span class="token operator">=</span>  getframeinfo<span class="token punctuation">(</span>frame<span class="token punctuation">)</span>
		args_repr  <span class="token operator">=</span>  format_args<span class="token punctuation">(</span>frame<span class="token punctuation">)</span>
		<span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Function {function_name}{args_repr} called from {frameinfo.filename}:{frameinfo.lineno}"</span><span class="token punctuation">)</span>
	<span class="token keyword">return</span>  trace_calls
</code></pre>
<h3 id="hypothesis">Hypothesis</h3>
<h4 id="hypothesis.add_state">Hypothesis.add_state</h4>
<p>Adds a new state to the hypothesis with a given access sequence, initializing its transitions based on the alphabet. The method creates a new <code>State</code> object with the provided <code>aseq</code>, adds it to <code>states</code>, and initializes <code>transitions</code> as a dictionary mapping each alphabet symbol to a <code>Transition</code> with the concatenated access sequence and the root node. All transitions are initially added to <code>open_transitions</code> as non-tree transitions, returning the new state.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">add_state</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> aseq<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> State<span class="token punctuation">:</span>
    state <span class="token operator">=</span> State<span class="token punctuation">(</span>self<span class="token punctuation">,</span> aseq<span class="token punctuation">)</span>
    self<span class="token punctuation">.</span>states<span class="token punctuation">.</span>add<span class="token punctuation">(</span>state<span class="token punctuation">)</span>
    state<span class="token punctuation">.</span>transitions <span class="token operator">=</span> <span class="token punctuation">{</span>
        a<span class="token punctuation">:</span> Transition<span class="token punctuation">(</span>self<span class="token punctuation">,</span> aseq <span class="token operator">+</span> a<span class="token punctuation">,</span> self<span class="token punctuation">.</span>root_node<span class="token punctuation">)</span> <span class="token keyword">for</span> a <span class="token keyword">in</span> self<span class="token punctuation">.</span>alphabet
    <span class="token punctuation">}</span>
    <span class="token keyword">for</span> t <span class="token keyword">in</span> state<span class="token punctuation">.</span>transitions<span class="token punctuation">.</span>values<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token comment"># all trasitions are initially be nontree</span>
        self<span class="token punctuation">.</span>open_transitions<span class="token punctuation">.</span>append<span class="token punctuation">(</span>t<span class="token punctuation">)</span>
    <span class="token keyword">return</span> state
</code></pre>
<h4 id="hypothesis.make_final">Hypothesis.make_final</h4>
<p>Marks a given state as a final state in the hypothesis DFA. The method checks if the <code>state</code> is in <code>states</code> using an assertion; if true, it adds the state to <code>final_states</code>. If the state is unknown, it raises a <code>ValueError</code>, ensuring only valid states are marked as final.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">make_final</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> state<span class="token punctuation">:</span> State<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> state <span class="token keyword">in</span> self<span class="token punctuation">.</span>states<span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>final_states<span class="token punctuation">.</span>add<span class="token punctuation">(</span>state<span class="token punctuation">)</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span><span class="token string">"Unknown state passed"</span><span class="token punctuation">)</span>
</code></pre>
<h4 id="hypothesis.run">Hypothesis.run</h4>
<p>Executes a deterministic run of the hypothesis DFA on a string, following tree transitions to a target state. Starting from an optional <code>start</code> state (defaulting to <code>self.start</code>), it checks if the string <code>s</code> is empty, returning the current state if true. Otherwise, it retrieves the transition for the first symbol of <code>s</code>, asserts the target state exists, and recursively calls <code>run</code> with the remaining string and target state, raising a <code>ValueError</code> if a transition is unclosed.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">run</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> s<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span> start<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> State<span class="token punctuation">:</span>
    <span class="token keyword">if</span> start <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        start <span class="token operator">=</span> self<span class="token punctuation">.</span>start
    <span class="token keyword">if</span> s <span class="token operator">==</span> <span class="token string">""</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> start
    t <span class="token operator">=</span> start<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>s<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">]</span>
    <span class="token keyword">if</span> t<span class="token punctuation">.</span>target_state <span class="token keyword">is</span> <span class="token operator">not</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>run<span class="token punctuation">(</span>s<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">,</span> t<span class="token punctuation">.</span>target_state<span class="token punctuation">)</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span><span class="token string">"Only call run when all transitions are closed"</span><span class="token punctuation">)</span>
</code></pre>
<h4 id="hypothesis.run_non_deterministic">Hypothesis.run_non_deterministic</h4>
<p>Performs a non-deterministic run of the hypothesis, resolving open transitions by sifting through the discrimination tree. Starting from an optional <code>start</code> state, it prints debug information and checks if <code>s</code> is empty, returning the current state if true. For non-empty strings, it gets the transition for the first symbol, sifts the target node using <code>target_node.sift</code> if it’s not a leaf, updates the target node, and recursively calls itself with the remaining string and teacher, handling dynamic tree updates in TTT.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">run_non_deterministic</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> s<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span> teacher<span class="token punctuation">:</span> Teacher<span class="token punctuation">,</span> start<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> State<span class="token punctuation">:</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Running hypothesis: start={start}, remaining str: {s}"</span><span class="token punctuation">)</span>
    <span class="token keyword">if</span> start <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        start <span class="token operator">=</span> self<span class="token punctuation">.</span>start
    <span class="token keyword">if</span> s <span class="token operator">==</span> <span class="token string">""</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> start
    t <span class="token operator">=</span> start<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>s<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">]</span>
    <span class="token keyword">if</span> <span class="token operator">not</span> t<span class="token punctuation">.</span>target_node<span class="token punctuation">.</span>is_leaf<span class="token punctuation">:</span>
        new_target <span class="token operator">=</span> t<span class="token punctuation">.</span>target_node<span class="token punctuation">.</span>sift<span class="token punctuation">(</span>t<span class="token punctuation">.</span>aseq<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>
        t<span class="token punctuation">.</span>target_node <span class="token operator">=</span> new_target
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>run_non_deterministic<span class="token punctuation">(</span>s<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">,</span> teacher<span class="token punctuation">,</span> t<span class="token punctuation">.</span>target_state<span class="token punctuation">)</span>
</code></pre>
<h4 id="hypothesis.evaluate_non_deterministic">Hypothesis.evaluate_non_deterministic</h4>
<p>Evaluates a string’s acceptance in a non-deterministic hypothesis using the teacher for sifting. Using <code>run_non_deterministic</code> with an optional <code>start</code> state, it checks if the resulting state is in <code>final_states</code>, returning a boolean to indicate acceptance, supporting TTT’s adaptive learning process.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">evaluate_non_deterministic</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> s<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span> teacher<span class="token punctuation">:</span> Teacher<span class="token punctuation">,</span> start<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> start <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        start <span class="token operator">=</span> self<span class="token punctuation">.</span>start

    <span class="token keyword">return</span> self<span class="token punctuation">.</span>run_non_deterministic<span class="token punctuation">(</span>s<span class="token punctuation">,</span> teacher<span class="token punctuation">,</span> start<span class="token operator">=</span>start<span class="token punctuation">)</span> <span class="token keyword">in</span> self<span class="token punctuation">.</span>final_states
</code></pre>
<h3 id="node-class">Node Class</h3>
<h4 id="replace_with_final">replace_with_final:</h4>
<p>This Replaces the current node with another node in place, finalizing a temporary node during TTT tree refinement. The method <strong>prints a debug message and asserts</strong> that both nodes have the same <code>is_leaf</code> status. It updates <code>is_leaf</code>, sets <code>_is_temporary</code> to <code>False</code>, clears <code>block</code>, and copies <code>_children</code>, <code>_state</code>, <code>_discriminator</code>, and <code>_incoming_non_tree</code> from the <code>node</code> parameter. This finalises temporary nodes in TTT, aligning with discriminator finalisation to optimise the tree.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">replace_with_final</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> node<span class="token punctuation">:</span> Node<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
    <span class="token comment"># replace the node with another in place</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Replacing {self} with {node}"</span><span class="token punctuation">)</span>
    <span class="token keyword">assert</span> self<span class="token punctuation">.</span>is_leaf <span class="token operator">==</span> node<span class="token punctuation">.</span>is_leaf

    self<span class="token punctuation">.</span>is_leaf <span class="token operator">=</span> node<span class="token punctuation">.</span>is_leaf
    self<span class="token punctuation">.</span>_is_temporary <span class="token operator">=</span> <span class="token boolean">False</span>
    self<span class="token punctuation">.</span>block <span class="token operator">=</span> <span class="token boolean">None</span>

    self<span class="token punctuation">.</span>_children <span class="token operator">=</span> node<span class="token punctuation">.</span>_children
    self<span class="token punctuation">.</span>_state <span class="token operator">=</span> node<span class="token punctuation">.</span>_state
    self<span class="token punctuation">.</span>_discriminator <span class="token operator">=</span> node<span class="token punctuation">.</span>_discriminator
    self<span class="token punctuation">.</span>_incoming_non_tree <span class="token operator">=</span> node<span class="token punctuation">.</span>_incoming_non_tree
</code></pre>
<h4 id="make_leaf-classmethod">make_leaf (classmethod)</h4>
<p>This creates a new leaf node to represent a state in the TTT hypothesis DFA. As a class method, it calls the <code>Node</code> constructor with <code>is_leaf=True</code> and <code>children=(None, None)</code>, resulting in a node with no children or discriminator. This node serves as a placeholder for a state (e.g., ( q_\varepsilon )) in TTT, to be linked later during tree expansion.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">classmethod</span>
<span class="token keyword">def</span> <span class="token function">make_leaf</span><span class="token punctuation">(</span>cls<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Node<span class="token punctuation">:</span>
    <span class="token keyword">return</span> Node<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<h4 id="make_inner-classmethod">make_inner (classmethod)</h4>
<p>make_inner method creates a new inner node with a specified discriminator for TTT tree splits. The class method calls the <code>Node</code> constructor with <code>is_leaf=False</code>, the provided <code>children</code> tuple (defaulting to <code>(None, None)</code>), and the <code>discriminator</code> string. It then sets the <code>parent</code> of any non-<code>None</code> children to the new node, establishing the binary tree structure used for TTT splits (e.g., with ( aa )).</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">classmethod</span>
<span class="token keyword">def</span> <span class="token function">make_inner</span><span class="token punctuation">(</span>
    cls<span class="token punctuation">,</span>
    discriminator<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span>
    children<span class="token punctuation">:</span> <span class="token builtin">tuple</span><span class="token punctuation">[</span>Optional<span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">,</span> Optional<span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Node<span class="token punctuation">:</span>
    new_node <span class="token operator">=</span> Node<span class="token punctuation">(</span><span class="token boolean">False</span><span class="token punctuation">,</span> children<span class="token punctuation">,</span> discriminator<span class="token operator">=</span>discriminator<span class="token punctuation">)</span>
    <span class="token keyword">for</span> child <span class="token keyword">in</span> children<span class="token punctuation">:</span>
        <span class="token keyword">if</span> child<span class="token punctuation">:</span>
            child<span class="token punctuation">.</span>parent <span class="token operator">=</span> new_node
    <span class="token keyword">return</span> new_node
</code></pre>
<h4 id="print_tree">print_tree</h4>
<p>Outputs the TTT discrimination tree rooted at the current node in a level-by-level format for visualisation. The method uses <code>child</code> (0, 1, or -1) to set an arrow (<code>-&gt;</code> or <code>=&gt;</code>) and <code>level</code> for indentation. For leaf nodes, it prints with spacing; for inner nodes, it recursively calls itself on the left child, prints the current node, and then the right child, asserting both children exist. This mirrors TTT’s binary structure.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">print_tree</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> child<span class="token punctuation">:</span> <span class="token builtin">int</span> <span class="token operator">=</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> level<span class="token punctuation">:</span> <span class="token builtin">int</span> <span class="token operator">=</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> child <span class="token operator">==</span> <span class="token number">1</span><span class="token punctuation">:</span>
        arrow <span class="token operator">=</span> <span class="token string">"=&gt;"</span>
    <span class="token keyword">elif</span> child <span class="token operator">==</span> <span class="token number">0</span><span class="token punctuation">:</span>
        arrow <span class="token operator">=</span> <span class="token string">"-&gt;"</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        arrow <span class="token operator">=</span> <span class="token string">"-&gt;"</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>is_leaf<span class="token punctuation">:</span>
        <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"{' ' * 4 * level}{arrow} {self}"</span><span class="token punctuation">)</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        <span class="token keyword">assert</span> self<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span>
        <span class="token keyword">assert</span> self<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>
        self<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>print_tree<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> level<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span>
        <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"{' ' * 4 * level}{arrow} {self}"</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">.</span>print_tree<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> level<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span>
</code></pre>
<h4 id="states">states</h4>
<p>This Generates all <code>State</code> objects associated with leaf nodes in the TTT hypothesis DFA. Using <code>__iter__</code> to traverse the subtree, it yields the <code>state</code> attribute for each node that is a leaf and has a non-<code>None</code> state, providing the set of states e.g., ( q(a), q(aa) ) for the TTT-learned DFA.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">states</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Generator<span class="token punctuation">[</span>State<span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> node <span class="token keyword">in</span> self<span class="token punctuation">:</span>
        <span class="token keyword">if</span> node<span class="token punctuation">.</span>is_leaf <span class="token operator">and</span> node<span class="token punctuation">.</span>state <span class="token keyword">is</span> <span class="token operator">not</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
            <span class="token keyword">yield</span> node<span class="token punctuation">.</span>state
</code></pre>
<h4 id="is_temporary">is_temporary</h4>
<p>Checks if the node is a temporary inner node during TTT tree construction. The property asserts the node is not a leaf and returns <code>_is_temporary</code>, which is set during initialization based on the presence of a discriminator. This tracks temporary nodes added during counterexample processing.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">is_temporary</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
    <span class="token keyword">assert</span> <span class="token operator">not</span> self<span class="token punctuation">.</span>is_leaf
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_is_temporary
</code></pre>
<h4 id="parent">parent</h4>
<p>Returns the parent node in the TTT discrimination tree hierarchy. The property returns the <code>_parent</code> attribute, set during node creation or via the setter, maintaining the tree’s parent-child relationships for operations like LCA computation.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">parent</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Optional<span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_parent
</code></pre>
<h4 id="children">children</h4>
<p>Returns the children of an inner node, enforcing the TTT binary tree structure. The property asserts the node is not a leaf and returns the <code>_children</code> tuple, containing two optional nodes for the left and right children, upholding the binary nature of the TTT tree.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">children</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">tuple</span><span class="token punctuation">[</span>Optional<span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">,</span> Optional<span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">assert</span> <span class="token operator">not</span> self<span class="token punctuation">.</span>is_leaf
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_children
</code></pre>
<h4 id="state">state</h4>
<p>Returns the <code>State</code> object associated with a leaf node in the TTT hypothesis DFA. The property asserts the node is a leaf and returns the <code>_state</code> attribute, set via the setter, linking the node to a state (e.g., ( q_\varepsilon )) in the TTT-learned DFA.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">state</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">assert</span> self<span class="token punctuation">.</span>is_leaf
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_state
</code></pre>
<h4 id="discriminator">discriminator</h4>
<p>Returns the discriminator of an inner node used for TTT tree splitting. The property asserts the node is not a leaf and <code>_discriminator</code> is not <code>None</code>, then returns the string value (e.g., ( aa )), which defines the split criterion in the TTT tree.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">discriminator</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">assert</span> <span class="token operator">not</span> self<span class="token punctuation">.</span>is_leaf
    <span class="token keyword">assert</span> self<span class="token punctuation">.</span>_discriminator <span class="token keyword">is</span> <span class="token operator">not</span> <span class="token boolean">None</span>
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_discriminator
</code></pre>
<h4 id="parent_value">parent_value</h4>
<p>Returns the membership query outcome (0 or 1) relative to the parent for TTT state distinction. The property raises an error if <code>parent</code> is <code>None</code>, returns <code>False</code> if the node is the left child (MQ=0), <code>True</code> if the right child (MQ=1), and handles invalid cases with an error, supporting signature computation.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">parent_value</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>parent <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">"{self} has no parent"</span><span class="token punctuation">)</span>
    <span class="token keyword">if</span> self <span class="token operator">==</span> self<span class="token punctuation">.</span>parent<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token boolean">False</span>
    <span class="token keyword">if</span> self <span class="token operator">==</span> self<span class="token punctuation">.</span>parent<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token boolean">True</span>
    <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">"{self} is not the child of its parent"</span><span class="token punctuation">)</span>
</code></pre>
<h4 id="signature">signature</h4>
<p>Returns the signature (discriminator-value pairs) for state distinction in the TTT algorithm. The property returns an empty list if there is no parent; otherwise, it creates a list with <code>(self.parent.discriminator, self.parent_value)</code> and prepends the parent’s signature recursively, identifying unique states in TTT.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">signature</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">list</span><span class="token punctuation">[</span><span class="token builtin">tuple</span><span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> <span class="token builtin">bool</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>parent <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">(</span>self<span class="token punctuation">.</span>parent<span class="token punctuation">.</span>discriminator<span class="token punctuation">,</span> self<span class="token punctuation">.</span>parent_value<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token operator">*</span>self<span class="token punctuation">.</span>parent<span class="token punctuation">.</span>signature<span class="token punctuation">]</span>
</code></pre>
<h4 id="incoming_non_tree">incoming_non_tree</h4>
<p>Returns the set of non-tree transitions to the node in the TTT hypothesis DFA. The property returns the <code>_incoming_non_tree</code> set, which stores <code>Transition</code> objects for edges outside the spanning tree, supporting complex DFA structures in TTT.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">incoming_non_tree</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">set</span><span class="token punctuation">[</span>Transition<span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_incoming_non_tree
</code></pre>
<h4 id="link">link</h4>
<p>Links a <code>State</code> object to the node, establishing a bidirectional connection in the TTT framework. The method sets the node’s <code>state</code> with the <code>state</code> parameter and sets <code>state.node</code> to <code>self</code>, creating a bidirectional link. Commented code for merging transitions is disabled, offering flexibility.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">link</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> state<span class="token punctuation">:</span> State<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
    self<span class="token punctuation">.</span>state <span class="token operator">=</span> state
    <span class="token keyword">if</span> state<span class="token punctuation">.</span>node<span class="token punctuation">:</span>
	    self<span class="token punctuation">.</span>incoming_non_tree <span class="token operator">|</span><span class="token operator">=</span> state<span class="token punctuation">.</span>node<span class="token punctuation">.</span>incoming_non_tree
	    self<span class="token punctuation">.</span>incoming_tree <span class="token operator">|</span><span class="token operator">=</span> state<span class="token punctuation">.</span>node<span class="token punctuation">.</span>incoming_tree
    state<span class="token punctuation">.</span>node <span class="token operator">=</span> self
</code></pre>
<h4 id="split_leaf">split_leaf</h4>
<p>Splits a leaf node into an inner node with two leaf children for TTT counterexample processing. The method asserts the node is a leaf, sets <code>is_leaf</code> to <code>False</code>, <code>_is_temporary</code> to <code>True</code>, clears <code>_state</code>, and sets <code>_discriminator</code>. It creates two leaf nodes, sets their <code>parent</code> to <code>self</code>, updates <code>_children</code>, and returns the children tuple, implementing TTT’s split for counterexamples.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">split_leaf</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> discriminator<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">tuple</span><span class="token punctuation">[</span>Node<span class="token punctuation">,</span> Node<span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">assert</span> self<span class="token punctuation">.</span>is_leaf
    self<span class="token punctuation">.</span>is_leaf <span class="token operator">=</span> <span class="token boolean">False</span>
    self<span class="token punctuation">.</span>_is_temporary <span class="token operator">=</span> <span class="token boolean">True</span>
    self<span class="token punctuation">.</span>_state <span class="token operator">=</span> <span class="token boolean">None</span>
    self<span class="token punctuation">.</span>_discriminator <span class="token operator">=</span> discriminator

    children <span class="token operator">=</span> <span class="token punctuation">(</span>Node<span class="token punctuation">.</span>make_leaf<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> Node<span class="token punctuation">.</span>make_leaf<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token keyword">for</span> child <span class="token keyword">in</span> children<span class="token punctuation">:</span>
        child<span class="token punctuation">.</span>parent <span class="token operator">=</span> self
    self<span class="token punctuation">.</span>_children <span class="token operator">=</span> children

    <span class="token keyword">return</span> children
</code></pre>
<h3 id="transition-class">Transition Class</h3>
<h4 id="transition.is_tree">Transition.is_tree</h4>
<p>Checks whether the transition is a tree transition, indicating it points to a finalized state. The method returns <code>True</code> if <code>_target_state</code> is not <code>None</code> (indicating a closed tree transition), and <code>False</code> otherwise (indicating an open non-tree transition). This distinction is central to TTT’s spanning-tree hypothesis.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">is_tree</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_target_state <span class="token keyword">is</span> <span class="token operator">not</span> <span class="token boolean">None</span>
</code></pre>
<h4 id="transition.target_node">Transition.target_node</h4>
<p>Returns the target node of the transition, adjusting based on whether it is a tree or non-tree transition. The property checks if <code>_target_state</code> exists; if true (tree transition), it returns the state’s <code>node</code>, otherwise it returns <code>_target_node</code>. This ensures the correct node is accessed depending on the transition type in the TTT tree.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">target_node</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Node<span class="token punctuation">:</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>_target_state<span class="token punctuation">:</span>
        <span class="token comment"># is tree transition</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>_target_state<span class="token punctuation">.</span>node
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>_target_node
</code></pre>
<h4 id="transition.target_state">Transition.target_state</h4>
<p>Returns the target state of the transition, falling back to the node’s state if not a tree transition. The property returns <code>_target_state</code> if set, otherwise it returns the <code>state</code> of <code>_target_node</code>, allowing access to the associated state when available in the TTT hypothesis.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">property</span>
<span class="token keyword">def</span> <span class="token function">target_state</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>_target_state <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>_target_node<span class="token punctuation">.</span>state
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>_target_state
</code></pre>
<h4 id="transition.make_tree">Transition.make_tree</h4>
<p>Converts a non-tree transition into a tree transition by adding a new state and linking it to the target node. The method checks if the transition is already a tree transition, raising a <code>ValueError</code> if true. Otherwise, it adds a new state with the transition’s <code>aseq</code> using <code>hypothesis.add_state</code>, removes the transition from <code>incoming_non_tree</code>, sets the target state, and makes the state final if the node’s signature includes <code>("", True)</code>. It then links the node to the state and returns the new state, finalizing the transition in TTT.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">make_tree</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> node<span class="token punctuation">:</span> Node<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> State<span class="token punctuation">:</span>
    <span class="token keyword">if</span> self<span class="token punctuation">.</span>is_tree<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">assert</span> self<span class="token punctuation">.</span>_target_state
        state <span class="token operator">=</span> self<span class="token punctuation">.</span>_target_state
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span><span class="token string">"transition is already a tree transition"</span><span class="token punctuation">)</span>
    <span class="token keyword">else</span><span class="token punctuation">:</span>
        state <span class="token operator">=</span> self<span class="token punctuation">.</span>hypothesis<span class="token punctuation">.</span>add_state<span class="token punctuation">(</span>self<span class="token punctuation">.</span>aseq<span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>target_node<span class="token punctuation">.</span>incoming_non_tree<span class="token punctuation">.</span>remove<span class="token punctuation">(</span>self<span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>target_state <span class="token operator">=</span> state
        <span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token string">""</span><span class="token punctuation">,</span> <span class="token boolean">True</span><span class="token punctuation">)</span> <span class="token keyword">in</span> node<span class="token punctuation">.</span>signature<span class="token punctuation">:</span>
            self<span class="token punctuation">.</span>hypothesis<span class="token punctuation">.</span>make_final<span class="token punctuation">(</span>state<span class="token punctuation">)</span>
    node<span class="token punctuation">.</span>link<span class="token punctuation">(</span>state<span class="token punctuation">)</span>
    <span class="token keyword">return</span> state
</code></pre>
<h2 id="the-ttt">THE TTT</h2>
<p>The full implementation for TTT algorithm can be found at <a href="https://replit.com/@ineerajrajeev/TTT-Algorithm">Replit</a></p>
<h3 id="step-1-initialization">Step 1: Initialization</h3>
<p>The algorithm begins with a single-state hypothesis and a minimal discrimination tree.</p>
<ul>
<li><strong>Discrimination Tree 1</strong>: Structure: A single node representing the empty string (epsilon) leads to state <code>q0</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Initial setup</span>
alphabet <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">"a"</span><span class="token punctuation">,</span> <span class="token string">"b"</span><span class="token punctuation">]</span>
teacher <span class="token operator">=</span> SimpleDFATeacher<span class="token punctuation">(</span>alphabet<span class="token punctuation">,</span> pattern<span class="token operator">=</span>r<span class="token string">"b*a(b*ab*ab*ab*)+b*"</span><span class="token punctuation">)</span>  <span class="token comment"># Regex for (4i + 3) 'a's</span>
hypothesis <span class="token operator">=</span> Hypothesis<span class="token punctuation">(</span>alphabet<span class="token punctuation">)</span>
root <span class="token operator">=</span> Node<span class="token punctuation">.</span>make_inner<span class="token punctuation">(</span><span class="token string">""</span><span class="token punctuation">,</span> children<span class="token operator">=</span><span class="token punctuation">(</span>Node<span class="token punctuation">.</span>make_leaf<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> Node<span class="token punctuation">.</span>make_leaf<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>  <span class="token comment"># Root with "" discriminator</span>
hypothesis<span class="token punctuation">.</span>root_node <span class="token operator">=</span> root
q0 <span class="token operator">=</span> hypothesis<span class="token punctuation">.</span>add_state<span class="token punctuation">(</span><span class="token string">""</span><span class="token punctuation">)</span>  <span class="token comment"># Initial state</span>
root<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>link<span class="token punctuation">(</span>q0<span class="token punctuation">)</span>  <span class="token comment"># Link q0 to t0</span>
</code></pre>
<pre class=" language-mermaid"><svg id="mermaid-svg-L4ln58QYY3XvkOUn" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="158" style="max-width: 206.734375px;" viewBox="0 0 206.734375 158"><style>#mermaid-svg-L4ln58QYY3XvkOUn{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-L4ln58QYY3XvkOUn .error-icon{fill:#552222;}#mermaid-svg-L4ln58QYY3XvkOUn .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-L4ln58QYY3XvkOUn .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-L4ln58QYY3XvkOUn .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-L4ln58QYY3XvkOUn .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-L4ln58QYY3XvkOUn .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-L4ln58QYY3XvkOUn .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-L4ln58QYY3XvkOUn .marker{fill:#666;stroke:#666;}#mermaid-svg-L4ln58QYY3XvkOUn .marker.cross{stroke:#666;}#mermaid-svg-L4ln58QYY3XvkOUn svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-L4ln58QYY3XvkOUn .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-L4ln58QYY3XvkOUn .cluster-label text{fill:#333;}#mermaid-svg-L4ln58QYY3XvkOUn .cluster-label span{color:#333;}#mermaid-svg-L4ln58QYY3XvkOUn .label text,#mermaid-svg-L4ln58QYY3XvkOUn span{fill:#000000;color:#000000;}#mermaid-svg-L4ln58QYY3XvkOUn .node rect,#mermaid-svg-L4ln58QYY3XvkOUn .node circle,#mermaid-svg-L4ln58QYY3XvkOUn .node ellipse,#mermaid-svg-L4ln58QYY3XvkOUn .node polygon,#mermaid-svg-L4ln58QYY3XvkOUn .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-L4ln58QYY3XvkOUn .node .label{text-align:center;}#mermaid-svg-L4ln58QYY3XvkOUn .node.clickable{cursor:pointer;}#mermaid-svg-L4ln58QYY3XvkOUn .arrowheadPath{fill:#333333;}#mermaid-svg-L4ln58QYY3XvkOUn .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-L4ln58QYY3XvkOUn .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-L4ln58QYY3XvkOUn .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-L4ln58QYY3XvkOUn .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-L4ln58QYY3XvkOUn .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-L4ln58QYY3XvkOUn .cluster text{fill:#333;}#mermaid-svg-L4ln58QYY3XvkOUn .cluster span{color:#333;}#mermaid-svg-L4ln58QYY3XvkOUn div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-L4ln58QYY3XvkOUn:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-L4ln58QYY3XvkOUn flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M103.3671875,54L103.3671875,79L103.3671875,104" marker-end="url(https://stackedit.io/app#arrowhead151)" style="fill:none"></path><defs><marker id="arrowhead151" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-602" transform="translate(103.3671875,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-36.609375" y="-23" width="73.21875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-26.609375,-13)"><foreignObject width="53.21875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-603" transform="translate(103.3671875,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-95.3671875" y="-23" width="190.734375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-85.3671875,-13)"><foreignObject width="170.734375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (t0, q0, non-final)</div></foreignObject></g></g></g></g></g></g></svg></pre>
<ul>
<li><strong>DFA 1</strong>:
<ul>
<li><strong>States</strong>: <code>{q0}</code></li>
<li><strong>Start State</strong>: <code>q0</code></li>
<li><strong>Final States</strong>: <code>{}</code></li>
<li><strong>Transitions</strong>:
<ul>
<li><code>q0 --a--&gt; q0</code></li>
<li><code>q0 --b--&gt; q0</code></li>
</ul>
</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-WaaXqKx3ja03TmYU" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="62" style="max-width: 111.515625px;" viewBox="0 0 111.515625 62"><style>#mermaid-svg-WaaXqKx3ja03TmYU{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-WaaXqKx3ja03TmYU .error-icon{fill:#552222;}#mermaid-svg-WaaXqKx3ja03TmYU .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-WaaXqKx3ja03TmYU .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-WaaXqKx3ja03TmYU .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-WaaXqKx3ja03TmYU .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-WaaXqKx3ja03TmYU .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-WaaXqKx3ja03TmYU .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-WaaXqKx3ja03TmYU .marker{fill:#666;stroke:#666;}#mermaid-svg-WaaXqKx3ja03TmYU .marker.cross{stroke:#666;}#mermaid-svg-WaaXqKx3ja03TmYU svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-WaaXqKx3ja03TmYU .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-WaaXqKx3ja03TmYU .cluster-label text{fill:#333;}#mermaid-svg-WaaXqKx3ja03TmYU .cluster-label span{color:#333;}#mermaid-svg-WaaXqKx3ja03TmYU .label text,#mermaid-svg-WaaXqKx3ja03TmYU span{fill:#000000;color:#000000;}#mermaid-svg-WaaXqKx3ja03TmYU .node rect,#mermaid-svg-WaaXqKx3ja03TmYU .node circle,#mermaid-svg-WaaXqKx3ja03TmYU .node ellipse,#mermaid-svg-WaaXqKx3ja03TmYU .node polygon,#mermaid-svg-WaaXqKx3ja03TmYU .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-WaaXqKx3ja03TmYU .node .label{text-align:center;}#mermaid-svg-WaaXqKx3ja03TmYU .node.clickable{cursor:pointer;}#mermaid-svg-WaaXqKx3ja03TmYU .arrowheadPath{fill:#333333;}#mermaid-svg-WaaXqKx3ja03TmYU .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-WaaXqKx3ja03TmYU .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-WaaXqKx3ja03TmYU .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-WaaXqKx3ja03TmYU .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-WaaXqKx3ja03TmYU .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-WaaXqKx3ja03TmYU .cluster text{fill:#333;}#mermaid-svg-WaaXqKx3ja03TmYU .cluster span{color:#333;}#mermaid-svg-WaaXqKx3ja03TmYU div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-WaaXqKx3ja03TmYU:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-WaaXqKx3ja03TmYU flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-A" id="L-A-A" style="opacity: 1;"><path class="path" d="M45.3125,22.37048287420132L76.38020833333333,8L84.14713541666666,8L91.9140625,31L84.14713541666666,54L76.38020833333333,54L45.3125,39.629517125798685" marker-end="url(https://stackedit.io/app#arrowhead152)" style="fill:none"></path><defs><marker id="arrowhead152" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="translate(91.9140625,31)" style="opacity: 1;"><g transform="translate(-11.6015625,-13)" class="label"><rect rx="0" ry="0" width="23.203125" height="26"></rect><foreignObject width="23.203125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-A" class="edgeLabel L-LS-A' L-LE-A">a,b</span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-606" transform="translate(26.65625,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q0</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
</ul>
<p>This initial DFA has <code>q0</code> self-looping on both <code>a</code> and <code>b</code>, rejecting all strings since there are no final states.</p>
<h3 id="step-2-first-refinement">Step 2: First Refinement</h3>
<p>The hypothesis is refined by introducing a new state and discriminator based on a counterexample.</p>
<ul>
<li><strong>Counterexample</strong>: <code>"aaa"</code> (3 'a’s, accepted by the target, rejected by DFA 1).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">process_counterexample</span><span class="token punctuation">(</span>hypothesis<span class="token punctuation">,</span> counterexample<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>counterexample<span class="token punctuation">)</span> <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        u <span class="token operator">=</span> counterexample<span class="token punctuation">[</span><span class="token punctuation">:</span>i<span class="token punctuation">]</span>
        a <span class="token operator">=</span> counterexample<span class="token punctuation">[</span>i<span class="token punctuation">]</span> <span class="token keyword">if</span> i <span class="token operator">&lt;</span> <span class="token builtin">len</span><span class="token punctuation">(</span>counterexample<span class="token punctuation">)</span> <span class="token keyword">else</span> <span class="token string">""</span>
        v <span class="token operator">=</span> counterexample<span class="token punctuation">[</span>i<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">:</span><span class="token punctuation">]</span> <span class="token keyword">if</span> i <span class="token operator">&lt;</span> <span class="token builtin">len</span><span class="token punctuation">(</span>counterexample<span class="token punctuation">)</span> <span class="token keyword">else</span> <span class="token string">""</span>
        current_state <span class="token operator">=</span> hypothesis<span class="token punctuation">.</span>run<span class="token punctuation">(</span>u<span class="token punctuation">)</span>
        <span class="token keyword">if</span> a <span class="token operator">and</span> v<span class="token punctuation">:</span>
            next_state <span class="token operator">=</span> current_state<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>a<span class="token punctuation">]</span><span class="token punctuation">.</span>target_state
            <span class="token keyword">if</span> next_state <span class="token operator">and</span> teacher<span class="token punctuation">.</span>is_member<span class="token punctuation">(</span>u <span class="token operator">+</span> a <span class="token operator">+</span> v<span class="token punctuation">)</span> <span class="token operator">!=</span> hypothesis<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>u <span class="token operator">+</span> a <span class="token operator">+</span> v<span class="token punctuation">)</span><span class="token punctuation">:</span>
                leaf_node <span class="token operator">=</span> next_state<span class="token punctuation">.</span>node
                leaf_node<span class="token punctuation">.</span>split_leaf<span class="token punctuation">(</span>discriminator<span class="token operator">=</span>v<span class="token punctuation">)</span>
                new_target <span class="token operator">=</span> hypothesis<span class="token punctuation">.</span>root_node<span class="token punctuation">.</span>sift<span class="token punctuation">(</span>u <span class="token operator">+</span> a<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>
                current_state<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>a<span class="token punctuation">]</span><span class="token punctuation">.</span>target_node <span class="token operator">=</span> new_target
                <span class="token keyword">if</span> teacher<span class="token punctuation">.</span>is_member<span class="token punctuation">(</span>u <span class="token operator">+</span> a <span class="token operator">+</span> v<span class="token punctuation">)</span><span class="token punctuation">:</span>
                    hypothesis<span class="token punctuation">.</span>make_final<span class="token punctuation">(</span>new_target<span class="token punctuation">.</span>state<span class="token punctuation">)</span>
                <span class="token keyword">break</span>
        <span class="token keyword">elif</span> <span class="token operator">not</span> a <span class="token operator">and</span> v <span class="token operator">==</span> <span class="token string">""</span><span class="token punctuation">:</span>
            <span class="token keyword">if</span> teacher<span class="token punctuation">.</span>is_member<span class="token punctuation">(</span>u<span class="token punctuation">)</span> <span class="token operator">!=</span> hypothesis<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>u<span class="token punctuation">)</span><span class="token punctuation">:</span>
                leaf_node <span class="token operator">=</span> current_state<span class="token punctuation">.</span>node
                leaf_node<span class="token punctuation">.</span>split_leaf<span class="token punctuation">(</span>discriminator<span class="token operator">=</span><span class="token string">""</span><span class="token punctuation">)</span>
                new_target <span class="token operator">=</span> hypothesis<span class="token punctuation">.</span>root_node<span class="token punctuation">.</span>sift<span class="token punctuation">(</span>u<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>
                <span class="token keyword">if</span> teacher<span class="token punctuation">.</span>is_member<span class="token punctuation">(</span>u<span class="token punctuation">)</span><span class="token punctuation">:</span>
                    hypothesis<span class="token punctuation">.</span>make_final<span class="token punctuation">(</span>new_target<span class="token punctuation">.</span>state<span class="token punctuation">)</span>
                <span class="token keyword">break</span>
    close_open_transitions<span class="token punctuation">(</span>hypothesis<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>

process_counterexample<span class="token punctuation">(</span>hypothesis<span class="token punctuation">,</span> <span class="token string">"aaa"</span><span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>
</code></pre>
<h4 id="hypothesis.evaluate">Hypothesis.evaluate</h4>
<p>Evaluates whether a string is accepted by the hypothesis DFA, based on reaching a final state. Using <code>run</code> with an optional <code>start</code> state (defaulting to <code>self.start</code>), it checks if the resulting state is in <code>final_states</code>, returning a boolean to indicate acceptance.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">evaluate</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> s<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span> start<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>State<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> start <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        start <span class="token operator">=</span> self<span class="token punctuation">.</span>start
    <span class="token keyword">return</span> self<span class="token punctuation">.</span>run<span class="token punctuation">(</span>s<span class="token punctuation">,</span> start<span class="token operator">=</span>start<span class="token punctuation">)</span> <span class="token keyword">in</span> self<span class="token punctuation">.</span>final_states
</code></pre>
<ul>
<li>
<p><strong>Equivalence Query</strong>:</p>
<ul>
<li><code>teacher.is_equivalent(hypothesis)</code> checks if the hypothesis matches the target language.</li>
<li>Current hypothesis (no final states) rejects all strings, but the target accepts <code>"aaa"</code> (3 'a’s).</li>
<li>Counterexample: <code>"aaa"</code> (accepted by target, rejected by hypothesis).</li>
</ul>
</li>
<li>
<p><strong>Process</strong>:</p>
<ul>
<li>Sift <code>"a"</code>: <code>teacher.is_member("a")</code> → <code>False</code> (1 ‘a’ ≠ 3 mod 4).</li>
</ul>
</li>
<li>
<p><strong>Sifting</strong>:<br>
Sifting refers to the process of traversing the discrimination tree to classify a given word or prefix into a corresponding leaf node, leveraging membership queries (MQs) to guide the traversal. This process is critical for mapping access sequences to states in the hypothesis DFA and refining the tree based on counterexamples.</p>
</li>
<li>
<p>Sift <code>""</code>: <code>teacher.is_member("")</code> returns <code>False</code> (0 'a’s, not <code>(4i + 3)</code>), so <code>q0</code> is non-accepting.</p>
</li>
<li>
<p>Transitions from <code>q0</code> for <code>a</code> and <code>b</code> are open, pointing to the root.</p>
</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Sifting functions extracted for reference</span>
<span class="token keyword">def</span>  <span class="token function">sift</span><span class="token punctuation">(</span>node<span class="token punctuation">:</span> Node<span class="token punctuation">,</span> s<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span> teacher<span class="token punctuation">:</span> Teacher<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Node<span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""
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
	"""</span>
	<span class="token keyword">if</span> node<span class="token punctuation">.</span>is_leaf<span class="token punctuation">:</span>
		<span class="token keyword">return</span> node <span class="token comment"># Return the leaf node if already at a leaf</span>
	<span class="token comment"># Compute the membership query outcome for the current discriminator</span>
	subtree <span class="token operator">=</span>  <span class="token builtin">int</span><span class="token punctuation">(</span>teacher<span class="token punctuation">.</span>is_member<span class="token punctuation">(</span>s <span class="token operator">+</span> node<span class="token punctuation">.</span>discriminator<span class="token punctuation">)</span><span class="token punctuation">)</span>
	child <span class="token operator">=</span> node<span class="token punctuation">.</span>children<span class="token punctuation">[</span>subtree<span class="token punctuation">]</span>
	<span class="token comment"># Assert the child exists to ensure tree integrity</span>
	<span class="token keyword">assert</span> child <span class="token keyword">is</span>  <span class="token operator">not</span> <span class="token boolean">None</span><span class="token punctuation">,</span>  <span class="token string">"Child node should not be None during sift"</span>
	<span class="token comment"># Recursively sift down the appropriate child</span>
	<span class="token keyword">return</span>  sift<span class="token punctuation">(</span>child<span class="token punctuation">,</span> s<span class="token punctuation">,</span> teacher<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>
<p>Add state <code>q1</code> with access sequence <code>"a"</code>.</p>
</li>
<li>
<p>Use discriminator <code>"aabbb"</code> to distinguish <code>q0</code> from <code>q1</code>:</p>
<ul>
<li><code>q0 · "aabbb" = "aabbb"</code> (2 'a’s) → <code>False</code>.</li>
<li><code>q1 · "aabbb" = "aaabbb"</code> (3 'a’s) → <code>True</code>.</li>
</ul>
</li>
<li>
<p><strong>Discrimination Tree 2</strong>:</p>
<ul>
<li>Root splits with discriminator <code>"aabbb"</code>.</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-aOOwwb36NN5YsA9O" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="158" style="max-width: 422.03125px;" viewBox="0 0 422.03125 158"><style>#mermaid-svg-aOOwwb36NN5YsA9O{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-aOOwwb36NN5YsA9O .error-icon{fill:#552222;}#mermaid-svg-aOOwwb36NN5YsA9O .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-aOOwwb36NN5YsA9O .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-aOOwwb36NN5YsA9O .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-aOOwwb36NN5YsA9O .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-aOOwwb36NN5YsA9O .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-aOOwwb36NN5YsA9O .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-aOOwwb36NN5YsA9O .marker{fill:#666;stroke:#666;}#mermaid-svg-aOOwwb36NN5YsA9O .marker.cross{stroke:#666;}#mermaid-svg-aOOwwb36NN5YsA9O svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-aOOwwb36NN5YsA9O .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-aOOwwb36NN5YsA9O .cluster-label text{fill:#333;}#mermaid-svg-aOOwwb36NN5YsA9O .cluster-label span{color:#333;}#mermaid-svg-aOOwwb36NN5YsA9O .label text,#mermaid-svg-aOOwwb36NN5YsA9O span{fill:#000000;color:#000000;}#mermaid-svg-aOOwwb36NN5YsA9O .node rect,#mermaid-svg-aOOwwb36NN5YsA9O .node circle,#mermaid-svg-aOOwwb36NN5YsA9O .node ellipse,#mermaid-svg-aOOwwb36NN5YsA9O .node polygon,#mermaid-svg-aOOwwb36NN5YsA9O .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-aOOwwb36NN5YsA9O .node .label{text-align:center;}#mermaid-svg-aOOwwb36NN5YsA9O .node.clickable{cursor:pointer;}#mermaid-svg-aOOwwb36NN5YsA9O .arrowheadPath{fill:#333333;}#mermaid-svg-aOOwwb36NN5YsA9O .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-aOOwwb36NN5YsA9O .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-aOOwwb36NN5YsA9O .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-aOOwwb36NN5YsA9O .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-aOOwwb36NN5YsA9O .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-aOOwwb36NN5YsA9O .cluster text{fill:#333;}#mermaid-svg-aOOwwb36NN5YsA9O .cluster span{color:#333;}#mermaid-svg-aOOwwb36NN5YsA9O div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-aOOwwb36NN5YsA9O:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-aOOwwb36NN5YsA9O flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M162.74625651041666,54L103.3671875,79L103.3671875,104" marker-end="url(https://stackedit.io/app#arrowhead153)" style="fill:none"></path><defs><marker id="arrowhead153" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-C" id="L-A-C" style="opacity: 1;"><path class="path" d="M272.0037434895833,54L331.3828125,79L331.3828125,104" marker-end="url(https://stackedit.io/app#arrowhead154)" style="fill:none"></path><defs><marker id="arrowhead154" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-C" class="edgeLabel L-LS-A' L-LE-C"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-612" transform="translate(217.375,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-58.390625" y="-23" width="116.78125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-48.390625,-13)"><foreignObject width="96.78125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;aabbb&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-613" transform="translate(103.3671875,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-95.3671875" y="-23" width="190.734375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-85.3671875,-13)"><foreignObject width="170.734375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (t0, q0, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-C-615" transform="translate(331.3828125,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q1, non-final)</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
</ul>
<h4 id="hypothesis.to_dfa">Hypothesis.to_dfa</h4>
<p>Converts the hypothesis into a formal DFA object for further analysis or export. It creates a <code>DFA</code> object, sets the start state ID, populates <code>states</code> and <code>final</code> sets with state IDs, and builds a transition dictionary mapping state IDs and symbols to target state IDs. It then closes the DFA with a sink state using the alphabet and returns the result.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">to_dfa</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> DFA<span class="token punctuation">:</span>
    dfa <span class="token operator">=</span> DFA<span class="token punctuation">(</span><span class="token punctuation">)</span>
    dfa<span class="token punctuation">.</span>start <span class="token operator">=</span> self<span class="token punctuation">.</span>start<span class="token punctuation">.</span><span class="token builtin">id</span>
    dfa<span class="token punctuation">.</span>states <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span><span class="token builtin">map</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> state<span class="token punctuation">:</span> state<span class="token punctuation">.</span><span class="token builtin">id</span><span class="token punctuation">,</span> self<span class="token punctuation">.</span>states<span class="token punctuation">)</span><span class="token punctuation">)</span>
    dfa<span class="token punctuation">.</span>final <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span><span class="token builtin">map</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> state<span class="token punctuation">:</span> state<span class="token punctuation">.</span><span class="token builtin">id</span><span class="token punctuation">,</span> self<span class="token punctuation">.</span>final_states<span class="token punctuation">)</span><span class="token punctuation">)</span>
    dfa<span class="token punctuation">.</span>next_state <span class="token operator">=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>dfa<span class="token punctuation">.</span>states<span class="token punctuation">)</span>
    <span class="token keyword">for</span> h_state <span class="token keyword">in</span> self<span class="token punctuation">.</span>states<span class="token punctuation">:</span>
        d_state <span class="token operator">=</span> h_state<span class="token punctuation">.</span><span class="token builtin">id</span>
        dfa<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>d_state<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span>
        <span class="token keyword">for</span> a<span class="token punctuation">,</span> transition <span class="token keyword">in</span> h_state<span class="token punctuation">.</span>transitions<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
            <span class="token keyword">assert</span> transition<span class="token punctuation">.</span>target_state  <span class="token comment"># otherwise not a DFA</span>
            dfa<span class="token punctuation">.</span>transitions<span class="token punctuation">[</span>d_state<span class="token punctuation">]</span><span class="token punctuation">[</span>a<span class="token punctuation">]</span> <span class="token operator">=</span> transition<span class="token punctuation">.</span>target_state<span class="token punctuation">.</span><span class="token builtin">id</span>
    dfa<span class="token punctuation">.</span>close_with_sink<span class="token punctuation">(</span><span class="token builtin">list</span><span class="token punctuation">(</span>self<span class="token punctuation">.</span>alphabet<span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> dfa
</code></pre>
<ul>
<li><strong>DFA 2</strong>:
<ul>
<li><strong>States</strong>: <code>{q0, q1}</code></li>
<li><strong>Start State</strong>: <code>q0</code></li>
<li><strong>Final States</strong>: <code>{}</code></li>
<li><strong>Transitions</strong>:
<ul>
<li><code>q0 --a--&gt; q1</code></li>
<li><code>q0 --b--&gt; q0</code></li>
<li><code>q1 --a--&gt; q0</code></li>
<li><code>q1 --b--&gt; q1</code></li>
</ul>
</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-UJd2uQ8DPEGhKg8k" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="184" style="max-width: 97.234375px;" viewBox="0 0 97.234375 184"><style>#mermaid-svg-UJd2uQ8DPEGhKg8k{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-UJd2uQ8DPEGhKg8k .error-icon{fill:#552222;}#mermaid-svg-UJd2uQ8DPEGhKg8k .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-UJd2uQ8DPEGhKg8k .marker{fill:#666;stroke:#666;}#mermaid-svg-UJd2uQ8DPEGhKg8k .marker.cross{stroke:#666;}#mermaid-svg-UJd2uQ8DPEGhKg8k svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-UJd2uQ8DPEGhKg8k .cluster-label text{fill:#333;}#mermaid-svg-UJd2uQ8DPEGhKg8k .cluster-label span{color:#333;}#mermaid-svg-UJd2uQ8DPEGhKg8k .label text,#mermaid-svg-UJd2uQ8DPEGhKg8k span{fill:#000000;color:#000000;}#mermaid-svg-UJd2uQ8DPEGhKg8k .node rect,#mermaid-svg-UJd2uQ8DPEGhKg8k .node circle,#mermaid-svg-UJd2uQ8DPEGhKg8k .node ellipse,#mermaid-svg-UJd2uQ8DPEGhKg8k .node polygon,#mermaid-svg-UJd2uQ8DPEGhKg8k .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .node .label{text-align:center;}#mermaid-svg-UJd2uQ8DPEGhKg8k .node.clickable{cursor:pointer;}#mermaid-svg-UJd2uQ8DPEGhKg8k .arrowheadPath{fill:#333333;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-UJd2uQ8DPEGhKg8k .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-UJd2uQ8DPEGhKg8k .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-UJd2uQ8DPEGhKg8k .cluster text{fill:#333;}#mermaid-svg-UJd2uQ8DPEGhKg8k .cluster span{color:#333;}#mermaid-svg-UJd2uQ8DPEGhKg8k div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-UJd2uQ8DPEGhKg8k:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-UJd2uQ8DPEGhKg8k flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M21.30097336065574,54L12.453125,92L21.30097336065574,130" marker-end="url(https://stackedit.io/app#arrowhead155)" style="fill:none"></path><defs><marker id="arrowhead155" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-A" id="L-A-A" style="opacity: 1;"><path class="path" d="M45.3125,21.456851615892507L71.61979166666667,8L78.19661458333334,8L84.7734375,31L78.19661458333334,54L71.61979166666667,54L45.3125,40.543148384107496" marker-end="url(https://stackedit.io/app#arrowhead156)" style="fill:none"></path><defs><marker id="arrowhead156" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-A" id="L-B-A" style="opacity: 1;"><path class="path" d="M32.01152663934426,130L40.859375,92L32.01152663934426,54" marker-end="url(https://stackedit.io/app#arrowhead157)" style="fill:none"></path><defs><marker id="arrowhead157" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-B" id="L-B-B" style="opacity: 1;"><path class="path" d="M45.3125,143.4568516158925L71.61979166666667,130L78.19661458333334,130L84.7734375,153L78.19661458333334,176L71.61979166666667,176L45.3125,162.5431483841075" marker-end="url(https://stackedit.io/app#arrowhead158)" style="fill:none"></path><defs><marker id="arrowhead158" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="translate(12.453125,92)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(84.7734375,31)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-A" class="edgeLabel L-LS-A' L-LE-A">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(40.859375,92)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-A" class="edgeLabel L-LS-B' L-LE-A">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(84.7734375,153)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-B" class="edgeLabel L-LS-B' L-LE-B">b</span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-624" transform="translate(26.65625,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q0</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-625" transform="translate(26.65625,153)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q1</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
</ul>
<h4 id="lca-classmethod">lca (classmethod)</h4>
<p>Finds the lowest common ancestor (LCA) of a list of nodes for merge points in TTT state equivalence checks. It calculates the minimum depth among <code>nodes</code> using a <code>map</code> function, collects nodes at this depth in a set by tracing up via <code>parent</code> (with error handling for null parents), and iteratively moves up until one node remains. It prints debug information and returns the LCA, supporting TTT’s tree refinement.</p>
<pre class=" language-python"><code class="prism  language-python">@<span class="token builtin">classmethod</span>
<span class="token keyword">def</span> <span class="token function">lca</span><span class="token punctuation">(</span>cls<span class="token punctuation">,</span> nodes<span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">[</span>Node<span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Node<span class="token punctuation">:</span>
    <span class="token keyword">if</span> <span class="token operator">not</span> nodes<span class="token punctuation">:</span>
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span><span class="token string">"Cannot compute LCA of an empty node list"</span><span class="token punctuation">)</span>
    min_depth <span class="token operator">=</span> <span class="token builtin">min</span><span class="token punctuation">(</span><span class="token builtin">map</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> node<span class="token punctuation">:</span> node<span class="token punctuation">.</span>depth<span class="token punctuation">,</span> nodes<span class="token punctuation">)</span><span class="token punctuation">)</span>
    nodes_in_layer<span class="token punctuation">:</span> <span class="token builtin">set</span><span class="token punctuation">[</span>Node<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
    <span class="token keyword">for</span> node <span class="token keyword">in</span> nodes<span class="token punctuation">:</span>
        current <span class="token operator">=</span> node
        <span class="token keyword">while</span> current<span class="token punctuation">.</span>depth <span class="token operator">&gt;</span> min_depth<span class="token punctuation">:</span>
            <span class="token keyword">if</span> current<span class="token punctuation">.</span>parent <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
                <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">"Node {current} has no parent at depth {current.depth}"</span><span class="token punctuation">)</span>
            current <span class="token operator">=</span> current<span class="token punctuation">.</span>parent
        nodes_in_layer<span class="token punctuation">.</span>add<span class="token punctuation">(</span>current<span class="token punctuation">)</span>
    <span class="token keyword">while</span> <span class="token builtin">len</span><span class="token punctuation">(</span>nodes_in_layer<span class="token punctuation">)</span> <span class="token operator">&gt;</span> <span class="token number">1</span><span class="token punctuation">:</span>
        nodes_in_layer <span class="token operator">=</span> <span class="token punctuation">{</span>node<span class="token punctuation">.</span>parent <span class="token keyword">for</span> node <span class="token keyword">in</span> nodes_in_layer <span class="token keyword">if</span> node<span class="token punctuation">.</span>parent <span class="token keyword">is</span> <span class="token operator">not</span> <span class="token boolean">None</span><span class="token punctuation">}</span>
        <span class="token keyword">if</span> <span class="token operator">not</span> nodes_in_layer<span class="token punctuation">:</span>
            <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">"No common ancestor found for nodes {nodes}"</span><span class="token punctuation">)</span>
    <span class="token keyword">if</span> <span class="token operator">not</span> nodes_in_layer<span class="token punctuation">:</span>
        <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">"LCA of {nodes} couldn't be computed"</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> nodes_in_layer<span class="token punctuation">.</span>pop<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<p>This DFA switches states on <code>a</code> and self-loops on <code>b</code>, but still rejects all strings, requiring further refinement.</p>
<h3 id="step-3-second-refinement">Step 3: Second Refinement</h3>
<p>A new counterexample drives the addition of states <code>q2</code> and <code>q3</code>.</p>
<ul>
<li><strong>Counterexample</strong>: <code>"aaa"</code> (reused, as DFA 2 rejects it: <code>q0 --a--&gt; q1 --a--&gt; q0 --a--&gt; q1</code>, non-final).</li>
<li><strong>RS Decomposition</strong>:<br>
The rs_eager_search function performs a <strong>binary search</strong> to find the index i in a counterexample string where the hypothesis DFA and the target DFA produce different outputs for a specific condition, defined by the function alpha. This index represents the point of divergence, which is critical for the RS decomposition process in the TTT algorithm. RS decomposition breaks a counterexample ww into parts uu, aa, and vv (where w=uavw=uav) to pinpoint where the hypothesis fails, allowing targeted refinement of the DFA or discrimination tree.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span>  <span class="token function">rs_eager_search</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> alpha<span class="token punctuation">:</span> Callable<span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">,</span>  <span class="token builtin">bool</span><span class="token punctuation">]</span><span class="token punctuation">,</span> high<span class="token punctuation">:</span> <span class="token builtin">int</span><span class="token punctuation">,</span> low<span class="token punctuation">:</span> <span class="token builtin">int</span>  <span class="token operator">=</span>  <span class="token number">0</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">int</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""
	Performs an eager Rivest-Schapire search to find the exact index i where
	alpha(i) != alpha(i+1), indicating the divergence point in the counterexample.
	Uses memoization to cache alpha results for efficiency.
	Args:
	alpha: A function that checks if the hypothesis and target agree at index i.
	high: The upper bound of the search.
	low: The lower bound of the search (default 0).
	Returns:
	The index i where alpha(i) != alpha(i+1).
	"""</span>
	<span class="token keyword">def</span>  <span class="token function">beta</span><span class="token punctuation">(</span>i<span class="token punctuation">:</span> <span class="token builtin">int</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">int</span><span class="token punctuation">:</span>
		<span class="token comment"># Check cache for alpha(i) and alpha(i+1)</span>
		<span class="token keyword">if</span>  i  <span class="token operator">not</span>  <span class="token keyword">in</span>  self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">:</span>
			self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">[</span>i<span class="token punctuation">]</span>  <span class="token operator">=</span>  alpha<span class="token punctuation">(</span>i<span class="token punctuation">)</span>
		<span class="token keyword">if</span>  i  <span class="token operator">+</span>  <span class="token number">1</span>  <span class="token operator">not</span>  <span class="token keyword">in</span>  self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">:</span>
			self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">[</span>i  <span class="token operator">+</span>  <span class="token number">1</span><span class="token punctuation">]</span>  <span class="token operator">=</span>  alpha<span class="token punctuation">(</span>i  <span class="token operator">+</span>  <span class="token number">1</span><span class="token punctuation">)</span>
		<span class="token keyword">return</span>  self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">[</span>i<span class="token punctuation">]</span>  <span class="token operator">+</span>  self<span class="token punctuation">.</span>alpha_cache<span class="token punctuation">[</span>i  <span class="token operator">+</span>  <span class="token number">1</span><span class="token punctuation">]</span>
	<span class="token keyword">while</span>  high  <span class="token operator">&gt;</span>  low<span class="token punctuation">:</span>
		mid  <span class="token operator">=</span>  <span class="token punctuation">(</span>low  <span class="token operator">+</span>  high<span class="token punctuation">)</span>  <span class="token operator">//</span>  <span class="token number">2</span>
		<span class="token keyword">if</span>  beta<span class="token punctuation">(</span>mid<span class="token punctuation">)</span>  <span class="token operator">==</span>  <span class="token number">1</span><span class="token punctuation">:</span> <span class="token comment"># alpha(mid) != alpha(mid+1)</span>
			<span class="token keyword">return</span>  mid
		<span class="token keyword">elif</span>  beta<span class="token punctuation">(</span>mid<span class="token punctuation">)</span>  <span class="token operator">==</span>  <span class="token number">0</span><span class="token punctuation">:</span> <span class="token comment"># beta(mid+1) &lt;= 1</span>
			low  <span class="token operator">=</span>  mid  <span class="token operator">+</span>  <span class="token number">1</span>
		<span class="token keyword">else</span><span class="token punctuation">:</span> <span class="token comment"># beta(mid - 1) &gt;= 1</span>
			high  <span class="token operator">=</span>  mid  <span class="token operator">-</span>  <span class="token number">1</span>
	<span class="token keyword">return</span>  low
</code></pre>
<ul>
<li>
<p>( w = “aaa” ).</p>
</li>
<li>
<p>( u = “aa” ), ( a = “a” ), ( v = “” ):</p>
<ul>
<li><code>q0 --a--&gt; q1 --a--&gt; q0</code>.</li>
<li><code>q0 · ""</code> → <code>False</code>, but <code>"aaa"</code> → <code>True</code>.</li>
</ul>
</li>
<li>
<p>Add <code>q3</code> (access sequence <code>"aaa"</code>, final) and <code>q2</code> (access sequence <code>"aa"</code>).</p>
</li>
<li>
<p>Discriminators <code>"aabbb"</code> and <code>"abbb"</code> refine the tree:</p>
<ul>
<li><code>q3 · "aabbb" = "aaaaabbb"</code> (5 'a’s) → <code>False</code>.</li>
<li><code>q1 · "aabbb" = "aaabbb"</code> (3 'a’s) → <code>True</code>.</li>
<li><code>q0 · "abbb" = "abbb"</code> (1 ‘a’) → <code>False</code>.</li>
<li><code>q2 · "abbb" = "aaabbb"</code> (3 'a’s) → <code>True</code>.</li>
</ul>
</li>
<li>
<p><strong>Discrimination Tree 3</strong>:</p>
<ul>
<li>Root splits to <code>q3</code>, then <code>"aabbb"</code> and <code>"abbb"</code> distinguish others.</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-vJp00lajTS4rnFA8" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="350" style="max-width: 475.78125px;" viewBox="0 0 475.78125 350"><style>#mermaid-svg-vJp00lajTS4rnFA8{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-vJp00lajTS4rnFA8 .error-icon{fill:#552222;}#mermaid-svg-vJp00lajTS4rnFA8 .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-vJp00lajTS4rnFA8 .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-vJp00lajTS4rnFA8 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-vJp00lajTS4rnFA8 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-vJp00lajTS4rnFA8 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-vJp00lajTS4rnFA8 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-vJp00lajTS4rnFA8 .marker{fill:#666;stroke:#666;}#mermaid-svg-vJp00lajTS4rnFA8 .marker.cross{stroke:#666;}#mermaid-svg-vJp00lajTS4rnFA8 svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-vJp00lajTS4rnFA8 .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-vJp00lajTS4rnFA8 .cluster-label text{fill:#333;}#mermaid-svg-vJp00lajTS4rnFA8 .cluster-label span{color:#333;}#mermaid-svg-vJp00lajTS4rnFA8 .label text,#mermaid-svg-vJp00lajTS4rnFA8 span{fill:#000000;color:#000000;}#mermaid-svg-vJp00lajTS4rnFA8 .node rect,#mermaid-svg-vJp00lajTS4rnFA8 .node circle,#mermaid-svg-vJp00lajTS4rnFA8 .node ellipse,#mermaid-svg-vJp00lajTS4rnFA8 .node polygon,#mermaid-svg-vJp00lajTS4rnFA8 .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-vJp00lajTS4rnFA8 .node .label{text-align:center;}#mermaid-svg-vJp00lajTS4rnFA8 .node.clickable{cursor:pointer;}#mermaid-svg-vJp00lajTS4rnFA8 .arrowheadPath{fill:#333333;}#mermaid-svg-vJp00lajTS4rnFA8 .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-vJp00lajTS4rnFA8 .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-vJp00lajTS4rnFA8 .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-vJp00lajTS4rnFA8 .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-vJp00lajTS4rnFA8 .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-vJp00lajTS4rnFA8 .cluster text{fill:#333;}#mermaid-svg-vJp00lajTS4rnFA8 .cluster span{color:#333;}#mermaid-svg-vJp00lajTS4rnFA8 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-vJp00lajTS4rnFA8:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-vJp00lajTS4rnFA8 flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M234.9921875,51.07479137846401L184.06640625,79L184.06640625,104" marker-end="url(https://stackedit.io/app#arrowhead159)" style="fill:none"></path><defs><marker id="arrowhead159" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-C" id="L-B-C" style="opacity: 1;"><path class="path" d="M139.30362955729166,150L90.6484375,175L90.6484375,200" marker-end="url(https://stackedit.io/app#arrowhead160)" style="fill:none"></path><defs><marker id="arrowhead160" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-D" id="L-B-D" style="opacity: 1;"><path class="path" d="M228.82918294270834,150L277.484375,175L277.484375,200" marker-end="url(https://stackedit.io/app#arrowhead161)" style="fill:none"></path><defs><marker id="arrowhead161" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-E" id="L-D-E" style="opacity: 1;"><path class="path" d="M225.90283203125,246L169.8359375,271L169.8359375,296" marker-end="url(https://stackedit.io/app#arrowhead162)" style="fill:none"></path><defs><marker id="arrowhead162" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-F" id="L-D-F" style="opacity: 1;"><path class="path" d="M329.06591796875,246L385.1328125,271L385.1328125,296" marker-end="url(https://stackedit.io/app#arrowhead163)" style="fill:none"></path><defs><marker id="arrowhead163" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-G" id="L-A-G" style="opacity: 1;"><path class="path" d="M308.2109375,51.07479137846401L359.13671875,79L359.13671875,104" marker-end="url(https://stackedit.io/app#arrowhead164)" style="fill:none"></path><defs><marker id="arrowhead164" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-C" class="edgeLabel L-LS-B' L-LE-C"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-D" class="edgeLabel L-LS-B' L-LE-D"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-E" class="edgeLabel L-LS-D' L-LE-E"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-F" class="edgeLabel L-LS-D' L-LE-F"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-G" class="edgeLabel L-LS-A' L-LE-G"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-644" transform="translate(271.6015625,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-36.609375" y="-23" width="73.21875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-26.609375,-13)"><foreignObject width="53.21875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-645" transform="translate(184.06640625,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-58.390625" y="-23" width="116.78125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-48.390625,-13)"><foreignObject width="96.78125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;aabbb&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-C-647" transform="translate(90.6484375,223)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q0, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-D-649" transform="translate(277.484375,223)" style="opacity: 1;"><rect rx="0" ry="0" x="-54.1875" y="-23" width="108.375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-44.1875,-13)"><foreignObject width="88.375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;abbb&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-E-651" transform="translate(169.8359375,319)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q1, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-F-653" transform="translate(385.1328125,319)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q2, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-G-655" transform="translate(359.13671875,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-66.6796875" y="-23" width="133.359375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-56.6796875,-13)"><foreignObject width="113.359375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q3, final)</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
<li>
<p><strong>DFA 3</strong>:</p>
<ul>
<li><strong>States</strong>: <code>{q0, q1, q2, q3}</code></li>
<li><strong>-transition</strong>: <code>q0</code></li>
<li><strong>Final States</strong>: <code>{q3}</code></li>
<li><strong>Transitions</strong>:
<ul>
<li><code>q0 --a--&gt; q1</code></li>
<li><code>q0 --b--&gt; q0</code></li>
<li><code>q1 --a--&gt; q2</code></li>
<li><code>q1 --b--&gt; q1</code></li>
<li><code>q2 --a--&gt; q3</code></li>
<li><code>q2 --b--&gt; q0</code></li>
<li><code>q3 --a--&gt; q0</code></li>
<li><code>q3 --b--&gt; q0</code></li>
</ul>
</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-OiVF1NcUcHvXyf0r" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="428" style="max-width: 208.60546875px;" viewBox="0 0 208.60546875 428"><style>#mermaid-svg-OiVF1NcUcHvXyf0r{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-OiVF1NcUcHvXyf0r .error-icon{fill:#552222;}#mermaid-svg-OiVF1NcUcHvXyf0r .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-OiVF1NcUcHvXyf0r .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-OiVF1NcUcHvXyf0r .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-OiVF1NcUcHvXyf0r .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-OiVF1NcUcHvXyf0r .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-OiVF1NcUcHvXyf0r .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-OiVF1NcUcHvXyf0r .marker{fill:#666;stroke:#666;}#mermaid-svg-OiVF1NcUcHvXyf0r .marker.cross{stroke:#666;}#mermaid-svg-OiVF1NcUcHvXyf0r svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-OiVF1NcUcHvXyf0r .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-OiVF1NcUcHvXyf0r .cluster-label text{fill:#333;}#mermaid-svg-OiVF1NcUcHvXyf0r .cluster-label span{color:#333;}#mermaid-svg-OiVF1NcUcHvXyf0r .label text,#mermaid-svg-OiVF1NcUcHvXyf0r span{fill:#000000;color:#000000;}#mermaid-svg-OiVF1NcUcHvXyf0r .node rect,#mermaid-svg-OiVF1NcUcHvXyf0r .node circle,#mermaid-svg-OiVF1NcUcHvXyf0r .node ellipse,#mermaid-svg-OiVF1NcUcHvXyf0r .node polygon,#mermaid-svg-OiVF1NcUcHvXyf0r .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-OiVF1NcUcHvXyf0r .node .label{text-align:center;}#mermaid-svg-OiVF1NcUcHvXyf0r .node.clickable{cursor:pointer;}#mermaid-svg-OiVF1NcUcHvXyf0r .arrowheadPath{fill:#333333;}#mermaid-svg-OiVF1NcUcHvXyf0r .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-OiVF1NcUcHvXyf0r .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-OiVF1NcUcHvXyf0r .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-OiVF1NcUcHvXyf0r .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-OiVF1NcUcHvXyf0r .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-OiVF1NcUcHvXyf0r .cluster text{fill:#333;}#mermaid-svg-OiVF1NcUcHvXyf0r .cluster span{color:#333;}#mermaid-svg-OiVF1NcUcHvXyf0r div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-OiVF1NcUcHvXyf0r:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-OiVF1NcUcHvXyf0r flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M119.37109375,41.21837185647645L26.65625,92L26.65625,130" marker-end="url(https://stackedit.io/app#arrowhead165)" style="fill:none"></path><defs><marker id="arrowhead165" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-A" id="L-A-A" style="opacity: 1;"><path class="path" d="M156.68359375,21.45685161589251L182.99088541666669,8L189.56770833333334,8L196.14453125,31L189.56770833333334,54L182.99088541666669,54L156.68359375,40.54314838410749" marker-end="url(https://stackedit.io/app#arrowhead166)" style="fill:none"></path><defs><marker id="arrowhead166" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-C" id="L-B-C" style="opacity: 1;"><path class="path" d="M26.65625,176L26.65625,214L44.923796106557376,252" marker-end="url(https://stackedit.io/app#arrowhead167)" style="fill:none"></path><defs><marker id="arrowhead167" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-B" id="L-B-B" style="opacity: 1;"><path class="path" d="M45.3125,143.4568516158925L71.61979166666667,130L78.19661458333334,130L84.7734375,153L78.19661458333334,176L71.61979166666667,176L45.3125,162.5431483841075" marker-end="url(https://stackedit.io/app#arrowhead168)" style="fill:none"></path><defs><marker id="arrowhead168" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-C LE-D" id="L-C-D" style="opacity: 1;"><path class="path" d="M55.98046875,298L55.98046875,336L119.37109375,383.1294991430204" marker-end="url(https://stackedit.io/app#arrowhead169)" style="fill:none"></path><defs><marker id="arrowhead169" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-C LE-A" id="L-C-A" style="opacity: 1;"><path class="path" d="M74.63671875,255.2818274111675L113.6953125,214L113.6953125,153L113.6953125,92L128.85297131147541,54" marker-end="url(https://stackedit.io/app#arrowhead170)" style="fill:none"></path><defs><marker id="arrowhead170" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-A" id="L-D-A" style="opacity: 1;"><path class="path" d="M143.57998206967213,374L152.75390625,336L152.75390625,275L152.75390625,214L152.75390625,153L152.75390625,92L143.57998206967213,54" marker-end="url(https://stackedit.io/app#arrowhead171)" style="fill:none"></path><defs><marker id="arrowhead171" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-A" id="L-D-A" style="opacity: 1;"><path class="path" d="M154.38774334016392,374L181.41796875,336L181.41796875,275L181.41796875,214L181.41796875,153L181.41796875,92L154.38774334016392,54" marker-end="url(https://stackedit.io/app#arrowhead172)" style="fill:none"></path><defs><marker id="arrowhead172" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="translate(26.65625,92)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(196.14453125,31)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-A" class="edgeLabel L-LS-A' L-LE-A">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(26.65625,214)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-C" class="edgeLabel L-LS-B' L-LE-C">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(84.7734375,153)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-B" class="edgeLabel L-LS-B' L-LE-B">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(55.98046875,336)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-C-D" class="edgeLabel L-LS-C' L-LE-D">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(113.6953125,153)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-C-A" class="edgeLabel L-LS-C' L-LE-A">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(152.75390625,214)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-A" class="edgeLabel L-LS-D' L-LE-A">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(181.41796875,214)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-A" class="edgeLabel L-LS-D' L-LE-A">b</span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-672" transform="translate(138.02734375,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q0</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-673" transform="translate(26.65625,153)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q1</div></foreignObject></g></g></g><g class="node default" id="flowchart-C-677" transform="translate(55.98046875,275)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q2</div></foreignObject></g></g></g><g class="node final" id="flowchart-D-681" transform="translate(138.02734375,397)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q3</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
</ul>
<p>This DFA now correctly handles strings up to 3 'a’s, with <code>q3</code> as the accepting state.</p>
<h3 id="step-4-final-refinement">Step 4: Final Refinement</h3>
<p>The hypothesis is tested and stabilized to match the target DFA.</p>
<ul>
<li><strong>Counterexample</strong>: <code>"aaaaaaa"</code> (7 'a’s, accepted, but needs verification).</li>
<li><strong>Validation</strong>:
<ul>
<li><code>"aaaaaaa"</code>: <code>q0 --a--&gt; q1 --a--&gt; q2 --a--&gt; q3 --a--&gt; q0 --a--&gt; q1 --a--&gt; q2 --a--&gt; q3</code> → <code>True</code> (correct).</li>
<li><code>"aaaaaa"</code>: <code>q0 --a--&gt; q1 --a--&gt; q2 --a--&gt; q3 --a--&gt; q0 --a--&gt; q1 --a--&gt; q2</code> → <code>False</code> (correct).</li>
</ul>
</li>
<li><strong>Adjustment</strong>:
<ul>
<li>Discriminator <code>"a"</code> ensures <code>q2</code> and <code>q3</code> are distinct from <code>q0</code> and <code>q1</code>.</li>
</ul>
</li>
<li><strong>Discrimination Tree 4</strong>:
<ul>
<li>Final tree refines states with <code>"a"</code> and <code>"aabbb"</code>.</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-0EoROxSfMs5rc6bZ" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="350" style="max-width: 505.421875px;" viewBox="0 0 505.421875 350"><style>#mermaid-svg-0EoROxSfMs5rc6bZ{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-0EoROxSfMs5rc6bZ .error-icon{fill:#552222;}#mermaid-svg-0EoROxSfMs5rc6bZ .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-0EoROxSfMs5rc6bZ .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-0EoROxSfMs5rc6bZ .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-0EoROxSfMs5rc6bZ .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-0EoROxSfMs5rc6bZ .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-0EoROxSfMs5rc6bZ .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-0EoROxSfMs5rc6bZ .marker{fill:#666;stroke:#666;}#mermaid-svg-0EoROxSfMs5rc6bZ .marker.cross{stroke:#666;}#mermaid-svg-0EoROxSfMs5rc6bZ svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-0EoROxSfMs5rc6bZ .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-0EoROxSfMs5rc6bZ .cluster-label text{fill:#333;}#mermaid-svg-0EoROxSfMs5rc6bZ .cluster-label span{color:#333;}#mermaid-svg-0EoROxSfMs5rc6bZ .label text,#mermaid-svg-0EoROxSfMs5rc6bZ span{fill:#000000;color:#000000;}#mermaid-svg-0EoROxSfMs5rc6bZ .node rect,#mermaid-svg-0EoROxSfMs5rc6bZ .node circle,#mermaid-svg-0EoROxSfMs5rc6bZ .node ellipse,#mermaid-svg-0EoROxSfMs5rc6bZ .node polygon,#mermaid-svg-0EoROxSfMs5rc6bZ .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-0EoROxSfMs5rc6bZ .node .label{text-align:center;}#mermaid-svg-0EoROxSfMs5rc6bZ .node.clickable{cursor:pointer;}#mermaid-svg-0EoROxSfMs5rc6bZ .arrowheadPath{fill:#333333;}#mermaid-svg-0EoROxSfMs5rc6bZ .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-0EoROxSfMs5rc6bZ .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-0EoROxSfMs5rc6bZ .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-0EoROxSfMs5rc6bZ .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-0EoROxSfMs5rc6bZ .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-0EoROxSfMs5rc6bZ .cluster text{fill:#333;}#mermaid-svg-0EoROxSfMs5rc6bZ .cluster span{color:#333;}#mermaid-svg-0EoROxSfMs5rc6bZ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-0EoROxSfMs5rc6bZ:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-0EoROxSfMs5rc6bZ flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M247.3828125,53.31539262860261L205.24609375,79L205.24609375,104" marker-end="url(https://stackedit.io/app#arrowhead173)" style="fill:none"></path><defs><marker id="arrowhead173" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-C" id="L-B-C" style="opacity: 1;"><path class="path" d="M164.43359375,146.22871055557687L103.3671875,175L103.3671875,200" marker-end="url(https://stackedit.io/app#arrowhead174)" style="fill:none"></path><defs><marker id="arrowhead174" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-D" id="L-B-D" style="opacity: 1;"><path class="path" d="M246.05859375,146.22871055557687L307.125,175L307.125,200" marker-end="url(https://stackedit.io/app#arrowhead175)" style="fill:none"></path><defs><marker id="arrowhead175" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-E" id="L-D-E" style="opacity: 1;"><path class="path" d="M255.54345703125,246L199.4765625,271L199.4765625,296" marker-end="url(https://stackedit.io/app#arrowhead176)" style="fill:none"></path><defs><marker id="arrowhead176" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-F" id="L-D-F" style="opacity: 1;"><path class="path" d="M358.70654296875,246L414.7734375,271L414.7734375,296" marker-end="url(https://stackedit.io/app#arrowhead177)" style="fill:none"></path><defs><marker id="arrowhead177" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-G" id="L-A-G" style="opacity: 1;"><path class="path" d="M320.6015625,53.31539262860261L362.73828125,79L362.73828125,104" marker-end="url(https://stackedit.io/app#arrowhead178)" style="fill:none"></path><defs><marker id="arrowhead178" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-C" class="edgeLabel L-LS-B' L-LE-C"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-D" class="edgeLabel L-LS-B' L-LE-D"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-E" class="edgeLabel L-LS-D' L-LE-E"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-F" class="edgeLabel L-LS-D' L-LE-F"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><rect rx="0" ry="0" width="0" height="0"></rect><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-G" class="edgeLabel L-LS-A' L-LE-G"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-700" transform="translate(283.9921875,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-36.609375" y="-23" width="73.21875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-26.609375,-13)"><foreignObject width="53.21875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-701" transform="translate(205.24609375,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-40.8125" y="-23" width="81.625" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-30.8125,-13)"><foreignObject width="61.625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;a&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-C-703" transform="translate(103.3671875,223)" style="opacity: 1;"><rect rx="0" ry="0" x="-95.3671875" y="-23" width="190.734375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-85.3671875,-13)"><foreignObject width="170.734375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (t0, q0, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-D-705" transform="translate(307.125,223)" style="opacity: 1;"><rect rx="0" ry="0" x="-58.390625" y="-23" width="116.78125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-48.390625,-13)"><foreignObject width="96.78125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node&lt;aabbb&gt;</div></foreignObject></g></g></g><g class="node default" id="flowchart-E-707" transform="translate(199.4765625,319)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q1, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-F-709" transform="translate(414.7734375,319)" style="opacity: 1;"><rect rx="0" ry="0" x="-82.6484375" y="-23" width="165.296875" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-72.6484375,-13)"><foreignObject width="145.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q2, non-final)</div></foreignObject></g></g></g><g class="node default" id="flowchart-G-711" transform="translate(362.73828125,127)" style="opacity: 1;"><rect rx="0" ry="0" x="-66.6796875" y="-23" width="133.359375" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-56.6796875,-13)"><foreignObject width="113.359375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Node (q3, final)</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
<li><strong>DFA 4</strong>:
<ul>
<li><strong>States</strong>: <code>{q0, q1, q2, q3}</code></li>
<li><strong>Start State</strong>: <code>q0</code></li>
<li><strong>Final States</strong>: <code>{q3}</code></li>
<li><strong>Transitions</strong>:
<ul>
<li><code>q0 --a--&gt; q1</code></li>
<li><code>q0 --b--&gt; q0</code></li>
<li><code>q1 --a--&gt; q2</code></li>
<li><code>q1 --b--&gt; q1</code></li>
<li><code>q2 --a--&gt; q3</code></li>
<li><code>q2 --b--&gt; q2</code></li>
<li><code>q3 --a--&gt; q0</code></li>
<li><code>q3 --b--&gt; q3</code></li>
</ul>
</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-ZbcTnI1bTZKsgIRp" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="428" style="max-width: 138.5234375px;" viewBox="0 0 138.5234375 428"><style>#mermaid-svg-ZbcTnI1bTZKsgIRp{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-ZbcTnI1bTZKsgIRp .error-icon{fill:#552222;}#mermaid-svg-ZbcTnI1bTZKsgIRp .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-ZbcTnI1bTZKsgIRp .marker{fill:#666;stroke:#666;}#mermaid-svg-ZbcTnI1bTZKsgIRp .marker.cross{stroke:#666;}#mermaid-svg-ZbcTnI1bTZKsgIRp svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-ZbcTnI1bTZKsgIRp .cluster-label text{fill:#333;}#mermaid-svg-ZbcTnI1bTZKsgIRp .cluster-label span{color:#333;}#mermaid-svg-ZbcTnI1bTZKsgIRp .label text,#mermaid-svg-ZbcTnI1bTZKsgIRp span{fill:#000000;color:#000000;}#mermaid-svg-ZbcTnI1bTZKsgIRp .node rect,#mermaid-svg-ZbcTnI1bTZKsgIRp .node circle,#mermaid-svg-ZbcTnI1bTZKsgIRp .node ellipse,#mermaid-svg-ZbcTnI1bTZKsgIRp .node polygon,#mermaid-svg-ZbcTnI1bTZKsgIRp .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .node .label{text-align:center;}#mermaid-svg-ZbcTnI1bTZKsgIRp .node.clickable{cursor:pointer;}#mermaid-svg-ZbcTnI1bTZKsgIRp .arrowheadPath{fill:#333333;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-ZbcTnI1bTZKsgIRp .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-ZbcTnI1bTZKsgIRp .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-ZbcTnI1bTZKsgIRp .cluster text{fill:#333;}#mermaid-svg-ZbcTnI1bTZKsgIRp .cluster span{color:#333;}#mermaid-svg-ZbcTnI1bTZKsgIRp div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-ZbcTnI1bTZKsgIRp:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-ZbcTnI1bTZKsgIRp flowchart{fill:apa;}</style><g><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath LS-A LE-B" id="L-A-B" style="opacity: 1;"><path class="path" d="M52.377305327868854,54L26.65625,92L26.65625,130" marker-end="url(https://stackedit.io/app#arrowhead179)" style="fill:none"></path><defs><marker id="arrowhead179" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-A LE-A" id="L-A-A" style="opacity: 1;"><path class="path" d="M86.6015625,21.456851615892507L112.90885416666667,8L119.48567708333334,8L126.0625,31L119.48567708333334,54L112.90885416666667,54L86.6015625,40.543148384107496" marker-end="url(https://stackedit.io/app#arrowhead180)" style="fill:none"></path><defs><marker id="arrowhead180" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-C" id="L-B-C" style="opacity: 1;"><path class="path" d="M26.65625,176L26.65625,214L26.65625,252" marker-end="url(https://stackedit.io/app#arrowhead181)" style="fill:none"></path><defs><marker id="arrowhead181" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-B LE-B" id="L-B-B" style="opacity: 1;"><path class="path" d="M45.3125,143.4568516158925L71.61979166666667,130L78.19661458333334,130L84.7734375,153L78.19661458333334,176L71.61979166666667,176L45.3125,162.5431483841075" marker-end="url(https://stackedit.io/app#arrowhead182)" style="fill:none"></path><defs><marker id="arrowhead182" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-C LE-D" id="L-C-D" style="opacity: 1;"><path class="path" d="M26.65625,298L26.65625,336L52.377305327868854,374" marker-end="url(https://stackedit.io/app#arrowhead183)" style="fill:none"></path><defs><marker id="arrowhead183" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-C LE-C" id="L-C-C" style="opacity: 1;"><path class="path" d="M45.3125,265.4568516158925L71.61979166666667,252L78.19661458333334,252L84.7734375,275L78.19661458333334,298L71.61979166666667,298L45.3125,284.5431483841075" marker-end="url(https://stackedit.io/app#arrowhead184)" style="fill:none"></path><defs><marker id="arrowhead184" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-A" id="L-D-A" style="opacity: 1;"><path class="path" d="M83.51331967213115,374L109.234375,336L109.234375,275L109.234375,214L109.234375,153L109.234375,92L83.51331967213115,54" marker-end="url(https://stackedit.io/app#arrowhead185)" style="fill:none"></path><defs><marker id="arrowhead185" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath LS-D LE-D" id="L-D-D" style="opacity: 1;"><path class="path" d="M86.6015625,387.4568516158925L112.90885416666667,374L119.48567708333334,374L126.0625,397L119.48567708333334,420L112.90885416666667,420L86.6015625,406.5431483841075" marker-end="url(https://stackedit.io/app#arrowhead186)" style="fill:none"></path><defs><marker id="arrowhead186" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="translate(26.65625,92)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-B" class="edgeLabel L-LS-A' L-LE-B">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(126.0625,31)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-A-A" class="edgeLabel L-LS-A' L-LE-A">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(26.65625,214)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-C" class="edgeLabel L-LS-B' L-LE-C">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(84.7734375,153)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-B-B" class="edgeLabel L-LS-B' L-LE-B">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(26.65625,336)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-C-D" class="edgeLabel L-LS-C' L-LE-D">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(84.7734375,275)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-C-C" class="edgeLabel L-LS-C' L-LE-C">b</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(109.234375,214)" style="opacity: 1;"><g transform="translate(-4.203125,-13)" class="label"><rect rx="0" ry="0" width="8.40625" height="26"></rect><foreignObject width="8.40625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-A" class="edgeLabel L-LS-D' L-LE-A">a</span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(126.0625,397)" style="opacity: 1;"><g transform="translate(-4.4609375,-13)" class="label"><rect rx="0" ry="0" width="8.921875" height="26"></rect><foreignObject width="8.921875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span id="L-L-D-D" class="edgeLabel L-LS-D' L-LE-D">b</span></div></foreignObject></g></g></g><g class="nodes"><g class="node default" id="flowchart-A-728" transform="translate(67.9453125,31)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q0</div></foreignObject></g></g></g><g class="node default" id="flowchart-B-729" transform="translate(26.65625,153)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q1</div></foreignObject></g></g></g><g class="node default" id="flowchart-C-733" transform="translate(26.65625,275)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q2</div></foreignObject></g></g></g><g class="node final" id="flowchart-D-737" transform="translate(67.9453125,397)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.65625" y="-23" width="37.3125" height="46" class="label-container"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.65625,-13)"><foreignObject width="17.3125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">q3</div></foreignObject></g></g></g></g></g></g></svg></pre>
</li>
</ul>
<p>This DFA accurately represents the target language, with states corresponding to remainders modulo 4: <code>q0</code> (0), <code>q1</code> (1), <code>q2</code> (2), <code>q3</code> (3, accepting).</p>
<h3 id="step-5-validation">Step 5: Validation</h3>
<ul>
<li>The final hypothesis correctly accepts strings with <code>(4i + 3)</code> 'a’s (e.g., <code>"aaa"</code>, <code>"aaaaaaa"</code>, <code>"bbbaaabbb"</code>) and rejects others (e.g., <code>"aaaa"</code>, <code>"aaaaaa"</code>).</li>
</ul>
<h2 id="summary">Summary</h2>
<p>The TTT algorithm represents a significant improvement over L* for learning regular languages:</p>
<h3 id="key-advantages-of-ttt">Key Advantages of TTT:</h3>
<ol>
<li><strong>Linear Space Complexity</strong>: TTT uses O(n) space compared to L*'s O(n²)</li>
<li><strong>No Redundancy</strong>: Each discriminator appears at most once in the tree</li>
<li><strong>Direct Counterexample Processing</strong>: RS decomposition directly identifies where to refine</li>
<li><strong>Efficient State Management</strong>: States are managed through tree leaves</li>
</ol>
<h3 id="when-to-use-ttt">When to Use TTT:</h3>
<ul>
<li>Learning regular languages from black-box systems</li>
<li>When space efficiency is important</li>
<li>When you need to minimize redundant queries</li>
<li>For protocol inference and verification tasks</li>
</ul>
<h4 id="references">References</h4>
<ul>
<li>Isberner, M., Howar, F., &amp; Steffen, B. (2014). <em>The TTT Algorithm: A Redundancy-Free Approach to Active Automata Learning</em>. Springer, Cham. <a href="https://doi.org/10.1007/978-3-319-11164-3_26">https://doi.org/10.1007/978-3-319-11164-3_26</a></li>
<li>Angluin, D. (1987). <em>Learning Regular Sets from Queries and Counterexamples</em>. Information and Computation, 75(2), 87–106.</li>
</ul>

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
eyJoaXN0b3J5IjpbMjY4NDIzNTY1XX0=
-->