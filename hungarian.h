//
// Created by samyr on 12/29/23.
//

#ifndef CONTROLLER_TRACKING_HUNGARIAN_H
#define CONTROLLER_TRACKING_HUNGARIAN_H

#include <vector>
#include <climits>

// code from emaxx.ru

struct solution{
    std::vector<int> idxs;
    int cost;
};

using std::vector;

inline solution hungarian(std::vector<std::vector<int>> a){
    int INF = 99999;
    int n = a.size()-1;
    int m = a[0].size()-1;
    vector<int> u (n+1), v (m+1), p (m+1), way (m+1);
    for (int i=1; i<=n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv (m+1, INF);
        vector<char> used (m+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0],  delta = INF,  j1;
            for (int j=1; j<=m; ++j)
                if (!used[j]) {
                    int cur = a[i0][j]-u[i0]-v[j];
                    if (cur < minv[j])
                        minv[j] = cur,  way[j] = j0;
                    if (minv[j] < delta)
                        delta = minv[j],  j1 = j;
                }
            for (int j=0; j<=m; ++j)
                if (used[j])
                    u[p[j]] += delta,  v[j] -= delta;
                else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> ans (n+1);
    for (int j=1; j<=m; ++j)
        ans[p[j]] = j;
    return solution(ans, -v[0]);
}
#endif //CONTROLLER_TRACKING_HUNGARIAN_H
