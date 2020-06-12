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