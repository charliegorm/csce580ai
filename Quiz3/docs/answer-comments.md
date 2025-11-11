# CSCE580 QUIZ 3 Answers
## Charlie Gorman

### Q1 Search and Heuristics 
- a. An admissable heuristic is a type of heuristic function that never overestimates the cost of reaching the goal / end state.
- b. Yes it is admissable, as it never goes over the actual cost. It does not overestimate. 
- c. This answer depends on the value of k, obviously. Similar to my previous answers, if k <= the actual cost of reaching the goal / end state, then yes it is admissable. If the goal / end state is reachable in less than k, though, it is not admissable. 
- d. The minimum of the three heuristic functions would be guaranteed to be admissable if at least one of the three is admissable. If two of the three are admissable, and you choose the min, you may be underestimating, but you are not overestimating (thus admissable). If only one of the three are admissable, then you select that one. With the maximum of the three functions, it is not guaranteed that the max of the three are not overestimating the cost of reaching goal / end state. 

### Q2 Using search for a practical problem
- Q2.1: Each state in the sample code is represented following this pattern: (left_missionaries, left_cannibals, right_missionaries, right_cannibals, boat_position). To begin, there are 3 missionaries and 3 cannibals on the left side of the bank of the river, with the boat on their same starting side, (3,3,0,0,"left"). To finish, the goal / end state is to get all of the missionaries and cannibals on the right side of the river, with the boat not needing to go back to the left side, because all travelers got over, (0,0,3,3,"right"). In the sample code, the search strategy that is used is Breadth-First Search (BFS), which guarantees that the solution that is found is the shortest solution (least amount of trips across the river). 

### Q3 Formulating a search problem
- a. CSP = {X, D, C} where X is the set of variables, D is the set of domains for these variables, and C is the set of constraints relating to these variables. 
- Variables (X): T, W, F, O, U, R, c1, c2, c3
- Domains (D): T,F can be any digit from 1-9 (leading digits / most significant, wont be 0), W,O,U can be any digit from 0-9, R is 0 (only adding 0s), c1 = 0 (carry will be 0 in ones column), c2,c3 can be either 0 or 1 (carry wont be larger than this)
- Constraints (C): All letters must represent dif unique digit - AllDiff(T, W, F, O, U, R), there cant be any leading 0s - T != 0, the equations for each column - 0 + 0 = R + 10 * c1, W + W + c1 = U + 10 * c2, T + T + c2 = O + 10 * c3, c3 = F (F != 0)
