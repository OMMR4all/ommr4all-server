import difflib


def solve_alignment(list_a, list_b):
    n = len(list_a)
    m = len(list_b)

    cost_matrix = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            similarity = difflib.SequenceMatcher(None, list_a[i], list_b[j]).ratio()
            cost_matrix[i][j] = 1.0 - similarity

    dp = [[float('inf')] * m for _ in range(n)]
    parent = [[-1] * m for _ in range(n)]

    for j in range(m):
        dp[0][j] = cost_matrix[0][j]

    for i in range(1, n):
        for j in range(m):

            best_prev_cost = float('inf')
            best_prev_k = -1
            for k in range(j + 1):
                if dp[i - 1][k] < best_prev_cost:
                    best_prev_cost = dp[i - 1][k]
                    best_prev_k = k

            if best_prev_k != -1:
                dp[i][j] = cost_matrix[i][j] + best_prev_cost
                parent[i][j] = best_prev_k

    min_total_cost = float('inf')
    last_b_index = -1

    for j in range(m):
        if dp[n - 1][j] < min_total_cost:
            min_total_cost = dp[n - 1][j]
            last_b_index = j

    assignments = [0] * n
    curr_b = last_b_index

    for i in range(n - 1, -1, -1):
        assignments[i] = curr_b
        curr_b = parent[i][curr_b]

    return assignments


# --- Example Usage ---
if __name__ == "__main__":
    list_a = [
        "The quick brown fox",
        "Jumps over the dog",
        "Jumps over the dog",
        "And runs away"
    ]

    list_b = [
        "Introduction to Animals",
        "The quick brown fox",
        "Jumps over the lazy dog",
        "And runs far away",
        "The End"
    ]

    results = solve_alignment(list_a, list_b)

    print(f"{'List A Index':<15} {'List A Content':<25} {'Matches B Index':<15} {'List B Content'}")
    print("-" * 80)

    for i, b_idx in enumerate(results):
        print(f"{i:<15} {list_a[i][:22]:<25} {b_idx:<15} {list_b[b_idx]}")