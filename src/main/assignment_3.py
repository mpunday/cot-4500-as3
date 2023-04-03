import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def function(t: float, w: float):
    return t - (w ** 2)


def euler(min_t, max_t, num_iterations, initial_w):
    # step size = h
    h = (max_t - min_t) / num_iterations

    # initialize w and t
    current_w = initial_w
    current_t = min_t

    for current_iteration in range(0, num_iterations):
        current_w = current_w + h * function(current_t, current_w)
        current_t = current_t + h

    return current_w


def runge_kutta(min_t, max_t, num_iterations, initial_w):
    # step size = h
    h = (max_t - min_t) / num_iterations

    # initialize w and t
    current_w = initial_w
    current_t = min_t

    for current_iteration in range(0, num_iterations):
        k_1 = h * function(current_t, current_w)
        k_2 = h * function(current_t + h / 2, current_w + k_1 / 2)
        k_3 = h * function(current_t + h / 2, current_w + k_2 / 2)
        k_4 = h * function(current_t + h, current_w + k_3)

        current_w = current_w + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
        current_t = current_t + h

    return current_w

def gauss_elimination(A, b):
    n = len(b)

    # Initialize solution matrix
    x = np.zeros(n)

    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # Perform elimination
    for i in range(n):
        # Check for zero along diagonal
        if Ab[i][i] == 0.0:
            return

        for j in range(i + 1, n):
            ratio = Ab[j][i] / Ab[i][i]
            Ab[j] = Ab[j] - (ratio * Ab[i])

    # Perform backward substitution
    x[n - 1] = Ab[n - 1][n] / Ab[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - (Ab[i][j] * x[j])
        x[i] = x[i] / Ab[i][i]

    return x

def LU_decomposition(A):
    n = len(A)

    U = A.copy()
    U = U.astype('double')
    L = np.eye(n, dtype=np.double)

    for i in range(n):
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k + 1]] = U[[k + 1, k]]

        # Use factors to create L and U
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return L, U

def determine_diagonally_dominate(A):
    n = len(A)

    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum = sum + abs(A[i][j])
        sum = sum - abs(A[i][i])

        # Compare diagonal to sum
        if (abs(A[i][i]) < sum):
            return False

    return True

def determine_positive_definite(A):
    if (np.array_equal(A, A.T)):
        return np.all(np.linalg.eigvals(A) > 0)

    return False

if __name__ == "__main__":
    # (1) Euler Method with the following details:
    min_t = 0
    max_t = 2
    num_iterations = 10
    initial_w = 1

    approximation = euler(min_t, max_t, num_iterations, initial_w)
    print("%.5f" % approximation)
    print()

    # (2) Runge-Kutta Method with the following details:
    min_t = 0
    max_t = 2
    num_iterations = 10
    initial_w = 1

    approximation = runge_kutta(min_t, max_t, num_iterations, initial_w)
    print("%.5f" % approximation)
    print()

    # (3) Use Gaussian elimination and backward substitution solve the following linear system:
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]])
    b = np.array([6, 0, -3])
    x = gauss_elimination(A, b)
    print(np.array(x, dtype=np.double))
    print()

    # (4) Implement LU Factorization for the following matrix:
    matrix_4 = np.array([[1, 1, 0, 3],
                         [2, 1, -1, 1],
                         [3, -1, -1, 2],
                         [-1, 2, 3, -1]])

    # (a) Print out the matrix determinant.
    det = np.linalg.det(matrix_4)
    print("%.5f" % det)
    print()

    # (b) Print out the L matrix.
    l_matrix, u_matrix = LU_decomposition(matrix_4)
    print(l_matrix)
    print()

    # (c) Print out the U matrix.
    print(u_matrix)
    print()

    # (5) Determine if the following matrix is diagonally dominate.
    matrix_5 = np.array([[9, 0, 5, 2, 1],
                         [3, 9, 1, 2, 1],
                         [0, 1, 7, 2, 3],
                         [4, 2, 3, 12, 2],
                         [3, 2, 4, 0, 8]])

    diagonally_dominate = determine_diagonally_dominate(matrix_5)

    print(diagonally_dominate)
    print()

    # (6) Determine if the matrix is a positive definite.
    matrix_6 = np.array([[2, 2, 1],
                         [2, 3, 0],
                         [1, 0, 2]])

    positive_definite = determine_positive_definite(matrix_6)

    print(positive_definite)