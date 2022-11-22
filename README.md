---
title: Individual Study Report AM 2020 spring
author: Yan-Tong Lin
tag: 
---

# Individual Study Report AM 2020 spring

[online presentation](https://hackmd.io/@ytlin/kMaxVD-presentation)

---


# Maximum domination of k-vertex subset in trees

---

## Abstract
The minimum domination problem has been widely studied and has numerous applications in computer networks. Here we consider a variation of the minimum domination problem. Given a positive integer k, find the maximum number of vertices that can be dominated by a k-vertex subset. We present a dynamic programming algorithm to solve this problem for trees in $O(k^2|V|)$ time

---

## Introduction

----

### domination(list version)
- $G := (V, E)$
- for convinience
    - $n := |V|$
    - $m := |E|$
    - $v$ be any vertex $\in V$
- $N(v) :=$ the set of vertices adjecent to $v$
- $N[v] := N(v)\cup \{v\}$
- $\forall S \subseteq V$, $N(S) := \bigcup_{v \in S}N(v)$ and $N[S] := \bigcup_{v \in S}N[v]$
- define dominate:
    - a vertex $v$ is said to *dominate* all vertices in $N[v]$
- A subset $D \subseteq V$ is a *dominating set* of $G$ if every vertex in $V-D$ is dominated bt at least one vertex in $D$
- or equivalently $D \cup N(D) = V$ or $N[D] = V$
- A minimum dominating set (MDS) is a dominating set with the minimum cardinality
    - In what follows, we will represent a minimum dominating set as MDS
- The domination number $\gamma(G)$ of $G$ is the minimum cardinality of a dominating set of $G$

----

### domination(original)
Let $G=(V,E)$ be a graph.
$N(v)$ of a vertex $v$ is the set of all vertices adjacent to $v$;
the closed neighborhood $N[v]$ of $v$ is $N[v]=N(v) \cup \{v\}$.
$\forall S \subseteq V$, $N(S) := \bigcup_{v \in S}N(v)$ and $N[S] := \bigcup_{v \in S}N[v]$.
A vertex $v$ is said to dominate all vertices in $N[v]$.
A subset $D \subseteq V$ is a dominating set of $G$ if every vertex in $V-D$ is adjacent to at least one vertex in $D$.

----

### maximum domination of k-vertex subset
- Now we consider a variant of domination problem
- given a tree $T = (V, E)$, an integer $k$, find the maximum number of vertices that can be dominated by a k-vertex subset of $V$

----

### Previous Work(brief)
- there is a linear time algorithm for MDS on cactus graph
- due to submodularity property, straight-forward greedy algorithm can do $O(1-1/e)-approximation$

----

### Main Contribution(1, summer + explicit function, provide lower bound, cactus graph, etc.)
- the algorithm
- full binary/path explicit formula
- lowerbound of time
- spanning tree + this algo?

---

## Previous Work

----

###  MDS of tree
- MDS of tree has a linear time algorithm
- A greedy approach
    - Pick leaves' parent and remove dominated vertices
    - Recursively do it
- A brief proof of correctness
    - since each leaf has to be chosen or be dominated
    - each leaf or its parent has to be chosen 
    - $dominated(leaf) \subseteq dominated(parent)$
    - $T-subtree(leaf)\cup parent \supseteq T-subtree(parent)\cup parent's~parent$
    - picking parent is always better than picking leaf

----

### MDS of cactus
- a linear time algorithm proposed by reference paper
    - https://www.sciencedirect.com/science/article/pii/0166218X86900892
- use $(F,B,R)$ domination approach + induction on end block

----

### Maximum domination of k vertices, path
- $min(n, 3*k)$
- by greedy construction
    - pattern [-*-]
    - proof of correctness
        - one vertex at most dominate 3(max degree = 2)
        - by choosing the 2nd, 5th, ..., vertex counting from one end, we can avoid covering

----

### Maximum k-domination of general graph $O((|E|+k)log(|V|))$-time, $O(1-1/e)$-approximation 

- the problem have submodular property
- a greedy algorithm will give an $O(1 - 1/e)$-approximation 
- please refer to reference for proof
    - key: ratio factorization

----

###  Maximum k-domination of tree $O(|V|+k)$-time, $O(1-1/e)$-approximation 
- maximum domination problem has submodularity property
- greedy algorithm in trees can be done in $O(n)$
    - link vertex $v$ with $degree(v)$, $N(v)$
        - can finding maximumam degree with amortized $O(|V|)$
        - $O(1) * O(|E|)$ modification
    - will not cover detail here


---

## Proposed Algorithm
- Key 1
    - greedy algorithm fail when vertices with bigger degrees shares share vertices
    - Information need to be pass on when dealing with double counting
    - An exhausive search might be needed
- Key 2
    - we can use **subtree solution to construct root solution**
    - there are **subproblems** that will be **used many times**
    - so we can use remember the subtree information 
    - **dymamic programming** comes in!
    - $dp[u][k][case]$ = max(configuration of children)
- Key 3
    - there are too many configurations 
        - $H(|child[u]|,k)$ for distributing $k$ vertices among children
    - but observe that we do not have to enumerate the previous configuration again when we know the maximum
    - the knapsack dp technique can help

----

### Description of Algorithm
- latex algorithnm 

```=tex
\begin{algorithm}[h]
  \caption{$DFS(u, f, p, s)$}\label{DFS}
  \begin{algorithmic}[1]
    \State $p[u] \leftarrow f$;
    \State $child[p] \leftarrow child[p] \cup \{u\}$;
    \State push $u$ to $s$; 
    \For{each $v \in N(u) \setminus \{p[u]\} $}
      \State $DFS(v, u, p, s)$;
    \EndFor
  \end{algorithmic}
\end{algorithm}

% \begin{algorithm}[h]
\begin{center}
  \captionof{algorithm}{Maximum domination of k-vertex subset with tree DP}\label{ALGO}
%  \caption{Maximum domination of k-vertex subset with tree DP}\label{ALGO}
  \begin{algorithmic}[1]
    \Require
        the tree $T := (V, E)$, the number $k$;
    \Ensure
        the maximum number of vertices s.t. a k-vertex subset can dominate;
    \State $n \leftarrow |V|$;
    \State $p \leftarrow $ an array of size $n$ initialized with $-1$;
    \State $s \leftarrow $ an empty stack;
    \State $u \leftarrow $ the vertex with index $0$ in $T$;
    \State $child \leftarrow $ an array of set of size $n$ initialized with $\emptyset$;
    \State 
    \State {\bf DFS}$(u, -1, p, s)$;
    \State // after $DFS$ is performed, $s$ contains the desired ordering for dynamic programming
    
    \State $dp \leftarrow $  a 4-D integer array of size $n \times k \times 2 \times 2$ initialized with $-\infty$;
    \While {($s \neq \emptyset$)}
      \State $u \leftarrow s.pop()$;
      \If{($child(u) = \emptyset$)}
        \For{($i \leftarrow 0$ {\bf to} $k$)}
            \State $dp[u][i][0][0] \leftarrow 0$;
            \State $dp[u][i][0][1] \leftarrow -\infty$;
            \State $dp[u][i][1][1] \leftarrow (i >= 1) ? 1 : -\infty$;
        \EndFor
        \State continue;
      \EndIf
      \State $nc \leftarrow |child(u)|$
      \State $knapsack00 \leftarrow $  a 2-D integer array of size $|child(u)| \times  k$
      \State $knapsack11 \leftarrow $  a 2-D integer array of size $|child(u)| \times k$
      \State $knapsack01 \leftarrow $  a 3-D integer array of size $|child(u)| \times k \times 2$
      \State initialize each item in $knapsack00, knapsack01, knapsack11$ with value $-\infty$
      \State // now we see set $child[u]$ as a 0-indexed array and do  knapsack
      \State // for $dp[u][k][0][0]$
      \For{($i \leftarrow 0$ {\bf to} $nc-1$)}
        \State $v \leftarrow child[u][i]$
        \For{($ki \leftarrow 0$ {\bf to} $k$)}
            \For{($kj \leftarrow 0$ {\bf to} $k$)}
                \State $option \leftarrow (i>0 ? knapsack[i-1][ki-kj] : 0) + max(dp[v][kj][0][0], dp[v][kj][0][1])$;
                \State $knapsack[i][ki] \leftarrow max(knapsack[i][ki], option)$;
            \EndFor  
        \EndFor
      \EndFor
      
      \For{($ki \leftarrow 0$ {\bf to} $k$)}
        \State $dp[u][ki][0][0] \leftarrow knapsack[nc-1][ki]$;
      \EndFor
      
      \State // for $dp[u][k][0][1]$
      \For{($i \leftarrow 0$ {\bf to} $nc-1$)}
        \State $v \leftarrow child[u][i]$
        \For{($ki \leftarrow 0$ {\bf to} $k$)}
            \For{($kj \leftarrow 0$ {\bf to} $k$)}
                \State $val0 \leftarrow max(dp[v][kj][0][0], dp[v][kj][0][1])$
                \State $val1 \leftarrow dp[v][kj][1][1]$
                \State $option0 \leftarrow (i == 0 ? 0 : knapsack01[i-1][ki-kj][0]) + val0$;
                \State $option1 \leftarrow max((i == 0 ? 0 :knapsack01[i-1][ki-kj][0]) + val1,
                                    (i == 0 ? -\infty :knapsack01[i-1][ki-kj][1] + max(val0,val1)))$;
                \State $knapsack01[i][ki][0] \leftarrow max(knapsack01[i][ki][0], option0)$;
                \State $knapsack01[i][ki][1] \leftarrow max(knapsack01[i][ki][1], option1)$;
            \EndFor  
        \EndFor
      \EndFor
      
      \State $dp[u][0][0][1] \leftarrow -\infty$;
      \For{($ki \leftarrow 1$ {\bf to} $k$)}
        \State $dp[u][ki][0][1] \leftarrow knapsack01[nc-1][ki][1]+1$;
      \EndFor
      
      \State // for $dp[u][k][1][1]$
      \For{($i \leftarrow 0$ {\bf to} $nc-1$)}
        \State $v \leftarrow child[u][i]$
        \For{($ki \leftarrow 0$ {\bf to} $k$)}
            \For{($kj \leftarrow 0$ {\bf to} $k$)}
                \State $option \leftarrow (i > 0 ? knapsack[i-1][ki-kj] : 0) + max({dp[v][kj][0][0]+1, dp[v][kj][0][1], dp[v][kj][1][1]})$
                \State $knapsack[i][ki] \leftarrow max(knapsack[i][ki], option)$;
            \EndFor  
        \EndFor
      \EndFor
      
      \State $dp[u][0][1][1] \leftarrow -\infty$;
      \For{($ki \leftarrow 1$ {\bf to} $k$)}
        \State $dp[u][ki][1][1] \leftarrow knapsack[nc-1][ki-1] + 1$;
      \EndFor
    \EndWhile

  \end{algorithmic}
\end{center}
%\end{algorithm}

```

----

### Demonstration of solving a special testcase
- the star-like graph
    - n = 13, k = 4, solutiob should be 13
    - this graph shows when greedy algorithm fails
![](https://i.imgur.com/QIHwhjW.png =50%x)

#### execution result
-  the root is 0
-  show the result of $dp[0][4]$ is correct
![](https://i.imgur.com/vWJ2uwI.png =50%x)



----

### Correctness of Algorithm(brief)
- the algotirhm, in essence, is doing enumaration on all possiblity of allocating resources on subtrees and all possible state for each subtree. While maintaining the information of the root of the subtree to complete the global problem.
- By using dynamic programming technique, we memorize the result and avoid recalculation.
- so the proof is to show all circumstances is considered at some point of our dynamic progranming algorithm

----

### Complexity of Algorithm
- let $n = |V|, m = |E|$
- to discuss the conplexity of a dynamic programming algithm, we discuss the number of state and the time for each transition
- we have $O(kn)$ states
- for each k state of some same vertex, we use $O(k^2)$ to do knapsack on that vertex with the imformation of its children calculated
    - this amortized to $O(k)$ per state
- so the total running time is $O(k^2n)$

---

## Expiriment Results

### more examples
- use networkx for plotting graphs

#### Example 1
- a hand craft testcase for verification
    - should not choose the mid vertex with maximum degree
- $n = 13, k = 4$
![](https://i.imgur.com/QIHwhjW.png =50%x)
![](https://i.imgur.com/vWJ2uwI.png =50%x)

#### Example 2
- a random generated tree with 
- $n = 15$, $k = 3$
![](https://i.imgur.com/UsK9AQq.png =50%x)
![](https://i.imgur.com/c47OcLR.png =30%x)


### Time Complexity Testing
- how is experiment done
    - g++ compile c++ file
    - python generate testcase with networkx
    - python use subroutine to run c++ execution file
    - python plot result
    - macbook pro 19
- experinent setting
    - n range from 1 to 901
    - k range from 0 to n
- measurement
    - sum of clocks for each testcase's running time(exlude IO)

#### n-k graph
- the resulting graph shows that when $n$ is fixed, the execution time of our proposed algorithm grows quadratic with $k$
![](https://i.imgur.com/9UuFUai.png)

#### k-n graph
- the resulting graph shows that when $k$ is fixed, the execution time of our proposed algorithm is linear to $n$
- and the slope is related with $k$
![](https://i.imgur.com/K2otL3s.png)

#### discussion on graphs and time complexity
- the graphs generally show the complexity is $O(k^2n)$ as expected

---

## Concluding Remark
- In this paper, we consider the problem of maximum domination of k-vertex subset and proposed a $O(k^2|V)$ algorithnm.
- future work
    - One remainning work is to prove that any algorithm that solves this problem runs in $\Omega(k^2|V|)$ time or has a lower bound.
    - One believe that catus graph have a similar dp solution

---

## Reference
- tree MDS
    - A linear algorithm for the domination number of a tree
- cactus MDS
    - A linear algorithm for finding a minimum dominating set in a cactus
- submodularity paper?
    - ask MT

---

## Appedix - C++ Implementaion
``` cpp=
#include <bits/stdc++.h>
#define LOCAL
#define SHOW
using namespace std;

#define X first
#define Y second
#define pb push_back
#define mp make_pair
#define all(a) a.begin(), a.end()
#define rep(i, st, n) for (int i = (st); i < (n); ++i)
#define repinv(i, st, n) for (int i = ((n)-1); i >= (st); --i)
#define MEM(a, b) memset(a, b, sizeof(a));
#ifdef LOCAL
#define debug(x) std::cerr << #x << ": " << x << endl
#else
#define debug(x) 860111
#endif
#define fastIO() ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#define fileIO(in, out) freopen(in, "r", stdin); freopen(out, "w", stdout)

typedef long long ll;
typedef unsigned long long ull;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef long double ld;

//mt19937 mrand(random_device{}());
//mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const ll mod=1e9+7;
//int rnd(int x) { return mrand() % x;}
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}
pii operator+(const pii&x, const pii&y) { return mp(x.X+y.X, x.Y+y.Y);}
pii operator-(const pii&x, const pii&y) { return mp(x.X-y.X, x.Y-y.Y);}
//INT_MAX, ULLONG_MAX, LLONG_MAX or numeric_limits<int>::min()

inline ll read(){
	char ch=getchar();ll x=0,f=0;
	while(ch<'0' || ch>'9') f|=ch=='-',ch=getchar();
	while(ch>='0' && ch<='9') x=x*10+ch-'0',ch=getchar();
	return f?-x:x;
}

//------------------------------------------------------------------------//
int T;
const int maxn = 5e5+7;
int ROOT = 0;
const int INF = maxn;
int n, k;
vector<int> G[maxn];
//int dp[maxn];
int p[maxn];
int deg[maxn];

// over load << for vector<T>
template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
{
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
        os << " " << *ii;
    }
    os << " ]";
    return os;
}

void dfs_order(int u, int f, vector<int> &vs)
{
    p[u] = f;
    vs.push_back(u);
    deg[u] = G[u].size();
    // move p to last, should be ok swapping here
    rep(i, 0, deg[u]) if(G[u][i] == p[u]) {swap(G[u][i], G[u][deg[u]-1]); break;} 
    for(int v: G[u])if(v != f) dfs_order(v, u, vs);
    return;
}


//check T
void solve()
{
  //cin >> n; rep(i, 0, n) cin >> a[i];
  cin >> n >> k;
  int a, b;
  rep(i, 0, n-1)
  {
    cin >> a >> b;
    G[a].pb(b);
    G[b].pb(a);
  }
  // find rooted version, p vector, odering of dp
  ROOT = 0;
  vector<int> vs;
  dfs_order(ROOT, -1, vs);
  reverse(vs.begin(), vs.end());
  cout << vs << endl;
  //dp
  // dp[u][k][0][0];
  // knap[k][i];
  vector<vector<array<array<int, 2>, 2>>> dp;
  dp.resize(n, vector<array<array<int, 2>, 2>>(k+1, array<array<int, 2>, 2>({{{-INF,-INF},{-INF,-INF}}})));
  for(int u: vs)
  {
      //debug(u);
      int nc = G[u].size() - int(p[u]!=-1);
      //base case, leaf vertex
      if(nc == 0)
      {
          rep(i, 0, k+1)
          {
              dp[u][i][0][0] = 0;
              dp[u][i][0][1] = -INF;
              dp[u][i][1][1] = (i >= 1) ? 1 : -INF;
          }
          continue;
      }

      //not base case
      vector<vector<int>> knapsack;
      //dp[0][0]
      knapsack = vector<vector<int>>(k+1, vector<int>(nc, -INF));
      rep(i, 0, nc)//order of visit
      {
          int v = G[u][i];
          rep(ki, 0, k+1)//order of visit
          {
              rep(ki2, 0, ki+1) //now i use
              {
                  int option = (i > 0 ? knapsack[ki-ki2][i-1] : 0) + max(dp[v][ki2][0][0], dp[v][ki2][0][1]);
                  knapsack[ki][i] = max(knapsack[ki][i], option);
              }
          }
      }
      rep(ki, 0, k+1)
      {
          dp[u][ki][0][0] = knapsack[ki][nc-1]; //best option after seeing all child possibility
      }
      //dp[1][1]
      knapsack = vector<vector<int>>(k, vector<int>(nc, -INF));
      rep(i, 0, nc)//order of visit
      {
          int v = G[u][i];
          rep(ki, 0, k)//order of visit
          {
              rep(ki2, 0, ki+1) //now i use
              {
                  int option = (i > 0 ? knapsack[ki-ki2][i-1] : 0) + max({dp[v][ki2][0][0]+1, dp[v][ki2][0][1], dp[v][ki2][1][1]});
                  knapsack[ki][i] = max(knapsack[ki][i], option);
              }
          }
      }
      dp[u][0][1][1] = -INF;
      rep(ki, 1, k+1)
      {
          //have to use 1 for root u
          dp[u][ki][1][1] = knapsack[ki-1][nc-1]+1; //best option after seeing all child possibility
      }
      //dp[0][1]
      vector<vector<int>> knapsack01[2];
      knapsack01[0] = vector<vector<int>>(k+1, vector<int>(nc, -INF)); //-INF not 0
      knapsack01[1] = vector<vector<int>>(k+1, vector<int>(nc, -INF)); //-INF not 0
      rep(i, 0, nc)//order of visit
      {
          int v = G[u][i];
          rep(ki, 0, k+1)//order of visit
          {
              rep(ki2, 0, ki+1) //now i use
              {
                  int val0 = max(dp[v][ki2][0][0], dp[v][ki2][0][1]);
                  int val1 = dp[v][ki2][1][1];
                  int option0 = (i == 0 ? 0 : knapsack01[0][ki-ki2][i-1]) + val0;
                  int option1 = max((i == 0 ? 0 :knapsack01[0][ki-ki2][i-1]) + val1,
                                    (i == 0 ? -INF :knapsack01[1][ki-ki2][i-1] + max(val0,val1)));
                  knapsack01[0][ki][i] = max(knapsack01[0][ki][i], option0);
                  knapsack01[1][ki][i] = max(knapsack01[1][ki][i], option1);
              }
          }
      }
      
      dp[u][0][0][1] = -INF;
      rep(ki, 1, k+1)
      {
          //have to use 0 for root u, +1 for root u is dominated
          dp[u][ki][0][1] = knapsack01[1][ki][nc-1]+1; //best option after seeing all child possibility
      }

  }
#ifdef SHOW
  for(int i : vs)
  {
      cout << "subtree at vertex " << i << ":\n";
      rep(ki, 0, k+1)
      {
          cout << "if use " << ki << " vertices\n";
          debug(dp[i][ki][0][0]);
          debug(dp[i][ki][0][1]);
          debug(dp[i][ki][1][1]);
      }
      cout << endl;
  }
#endif
  int global_ans = max({dp[ROOT][k][0][0], dp[ROOT][k][0][1], dp[ROOT][k][1][1]});
  cout << global_ans << endl;

  return;
}


//g++ -o out -std=c++11 A.cpp

signed main()
{
  fastIO();
  T = 1;
  //cin >> T; //this
  rep(tc, 1, T+1)
  {
    //cout << "Case #" << tc << ": ";
    solve();
  }
  return 0;
}

/*
10 2
3 0 3 2 4 1 5 2 6 1 6 5 7 6 8 1 9 4
*/

/*
13 4
0 1 
0 2
0 3 
0 4 
1 5
1 6 
2 7
2 8
3 9
3 10
4 11
4 12
*/
```

## Appendix - Python Experiment
```python=
#!/usr/bin/env python
# coding: utf-8
# author: Yan-Tong Lin

import numpy as np
from collections import defaultdict
import networkx as nx
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT, run
import  matplotlib.pyplot as plt

exe_file_name = "exp_cpp"
exe_cmd = "./" + exe_file_name


def str_tc(n, k):
    T = nx.generators.trees.random_tree(n, int(time.time()))
    # print(T.edges)
    tc = ""
    tc += str(n) + " " + str(k)
    for a, b in T.edges:
        tc += " " + str(a) + " " + str(b)
    return tc


def run1(n, k):
    p = run([exe_cmd], stdout=PIPE, input=str_tc(n,k), encoding='ascii')
    ans, t= map(int,p.stdout.split())
    return ans, t

def exp(range_n=100, range_split=10, exp_time=10):
    dat = np.zeros((range_n//range_split+1, range_n//range_split+1, exp_time))
    for exp_n in range(1,range_n+1, range_split):
        for exp_k in range(1, exp_n+1, range_split):
            for ti in range(exp_time):
                ans, tt = run1(exp_n, exp_k)
                dat[exp_n//range_split][exp_k//range_split][ti] = tt
    return dat
    
dat2 = exp(1000,100,5)


def show_stat(dat, range_n, range_split):
    dats = dat.sum(axis=2)
    #print(dats)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i in range(range_n//range_split):
        ax1.set_title("n-k graph")
        ax1.plot(dats[i][:i], label=("n = " + str(i*range_split+1)))
        ax1.legend()
        
    for i in range(range_n//range_split):
        #print(i, dats.T[i][i:-1])
        ax2.set_title("k-n graph")
        ax2.plot((dats.T[i][i:-1]), label=("k = " + str(i*range_split+1)))
        ax2.legend()
    

show_stat(dat2, 1000, 100)

```
