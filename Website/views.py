from flask import Blueprint, render_template, request, flash, redirect, url_for
from numpy.linalg import *
from scipy.linalg import lu, svd
from random import choice, randint
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')

# To know what files we currently have for plots
global plots
plots = []

views = Blueprint('views', __name__)

# This functions purpose is to setup the main page or home
# route of the website
@views.route('/', methods=["GET", "POST"])
def home():

    if request.method == "POST":

        #Remove old plot files if necessary
        for file in os.listdir("./Website/static/dataGraphs"):
            os.remove("./Website/static/dataGraphs/" + str(file))

        # Pulling the matrix information from the post request
        A = request.form.get("amatrix")
        B = request.form.get("bmatrix")
        D = request.form.get("data")
        print(A)  # Print A's string form
        print(B)  # Print B's string form
        print(D)  # Print D's string form

        # Getting the operation that needs to be done on the matrix
        for r in request.form:
            if r != "amatrix" and r != "bmatrix" and r != "m" and r != "n" and r != "data":
                operation = request.form.get(r)

				# To randomize our matrices
        if operation == "randA":
            m = request.form.get('m')
            n = request.form.get('n')
            r = randomize_matrix(m, n, 'A', B, D)

            return r

        elif operation == "randB":
            m = request.form.get('m')
            n = request.form.get('n')
            r = randomize_matrix(m, n, 'B', A, D)

            return r
					
				# Making sure that there was something given by the user
        if not A and not B and not D:
            flash(
                "Nothing entered for matrix A or B or data D. Cannot continue computation.",
                "error")
            return render_template("home.html",
                                   last_inputA=str(A),
                                   last_inputB=str(B),
                                   last_inputD=str(D),
                                   matrix="",
                                   output_type="",
                                   output="")

        # Building up the matrices A and B if they are not None
        if A:
            matrixA = build_matrix(A, False)

            if matrixA == False:
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")

            try:
                matrixA = np.matrix(matrixA)
            except:
                flash(
                    "Check that input for matrix A follows input criteria, spaces between each entry and press enter/return to start on a new row of the matrix. Also exponents are used by doing ** Ex: 2**2 = 4",
                    "error")
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")
            print("A = ", matrixA)
        else:
            A = ""

        if B:
            matrixB = build_matrix(B, False)

            if matrixB == False:
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")

            try:
                matrixB = np.matrix(matrixB)
            except:
                flash(
                    "Check that input for matrix B follows input criteria, spaces between each entry and press enter/return to start on a new row of the matrix. Also exponents are used by doing ** Ex: 2**2 = 4",
                    "error")
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")
            print("B = ", matrixB)
        else:
            B = ""

        # Building up the data if it is not None:
        if D:
            matrixD, b = build_matrix(str(D), True)

            if matrixD == False:
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")

            try:
                xData = matrixD
                yData = b
                matrixD = np.matrix(matrixD)
                vecB = np.matrix(b)
            except:
                flash(
                    "Check that input for Data follows input criteria, spaces between each entry and press enter/return to start on a new row of the matrix. There should only be two inputs which are you x and y points. Also exponents are used by doing ** Ex: 2**2 = 4",
                    "error")
                return render_template("home.html",
                                       last_inputA=str(A),
                                       last_inputB=str(B),
                                       last_inputD=str(D),
                                       matrix="",
                                       output_type="",
                                       output="")
            print("D = ", matrixD)
        else:
            D = ""

        # Handling the operation the user requested from the website
        if operation == "multiplyAB":
            r = check_matrices(A, B, D)
            if r: return r

            result = multiply_matrix(matrixA, matrixB)

            r = create_result(result, A, B, D, "", "multiply", "Successfully computed A * B, scroll down to see output.")
            return r

        elif operation == "A-B":
            r = check_matrices(A, B, D)
            if r: return r

            result = matrix_subtraction(matrixA, matrixB)

            r = create_result(result, A, B, D, "", "subtract", "Successfully computed A - B, scroll down to see output.")
            return r

        elif operation == "A+B":
            r = check_matrices(A, B, D)
            if r: return r

            result = matrix_addition(matrixA, matrixB)

            r = create_result(result, A, B, D, "", "add", "Successfully computed A + B, scroll down to see output.")
            return r

        elif operation == "luA":
            r = check_matrix(A, B, D, "A")
            if r: return r					

            result = lu_factor(matrixA)

            r = create_result(result, A, B, D, "A", "lu", "Successfully computed LU factorization of matrix A, scroll down to see output.")
            return r

        elif operation == "luB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = lu_factor(matrixB)

            r = create_result(result, A, B, D, "B", "lu", "Successfully computed LU factorization of matrix B, scroll down to see output.")
            return r

        elif operation == "qrA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = qr_factor(matrixA)

            r = create_result(result, A, B, D, "A", "qr", "Successfully computed QR factorization of matrix A, scroll down to see output.")
            return r

        elif operation == "qrB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = qr_factor(matrixB)

            r = create_result(result, A, B, D, "B", "qr", "Successfully computed QR factorization of matrix B, scroll down to see output.")
            return r

        elif operation == "svdA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = svd_factor(matrixA)

            r = create_result(result, A, B, D, "A", "svd", "Successfully computed SVD factorization of matrix A, scroll down to see output.")
            return r

        elif operation == "svdB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = svd_factor(matrixB)

            r = create_result(result, A, B, D, "B", "svd", "Successfully computed SVD factorization of matrix B, scroll down to see output.")
            return r

        elif operation == "eigValA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = eig_value(matrixA)

            r = create_result(result, A, B, D, "A", "eigvals", "Successfully computed eigenvalues of matrix A, scroll down to see output.")
            return r

        elif operation == "eigValB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = eig_value(matrixB)

            r = create_result(result, A, B, D, "B", "eigvals", "Successfully computed eigenvalues of matrix B, scroll down to see output.")
            return r

        elif operation == "invA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = inverse(matrixA)

            r = create_result(result, A, B, D, "A", "inverse", "Successfully computed the inverse of matrix A, scroll down to see output.")
            return r

        elif operation == "invB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = inverse(matrixB)

            r = create_result(result, A, B, D, "B", "inverse", "Successfully computed the inverse of matrix B, scroll down to see output.")
            return r

        elif operation == "transA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = transpose(matrixA)

            r = create_result(result, A, B, D, "A", "transpose", "Successfully computed the transpose of matrix A, scroll down to see output.")
            return r

        elif operation == "transB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = transpose(matrixB)

            r = create_result(result, A, B, D, "B", "transpose", "Successfully computed the transpose of matrix B, scroll down to see output.")
            return r

        elif operation == "rankA":
            r = check_matrix(A, B, D, "A")
            if r: return r	

            result = rank(matrixA)

            r = create_result(result, A, B, D, "A", "rank", "Successfully computed the rank of matrix A, scroll down to see output.")
            return r

        elif operation == "rankB":
            r = check_matrix(B, A, D, "B")
            if r: return r	

            result = rank(matrixB)

            r = create_result(result, A, B, D, "B", "rank", "Successfully computed the rank of matrix B, scroll down to see output.")
            return r

        elif operation == "dataIn":
            r = check_matrix(D, A, B, "D")
            if r: return r

            result = create_bestfit(matrixD, vecB, xData, yData)

            r = create_result(result, A, B, D, "D", "dataOut", "Successfully found the line of best fit for the given data, scroll down to see the outputed graph.")
            return r

        return render_template("home.html",
                               last_inputA=str(A),
                               last_inputB=str(B),
                               last_inputD=str(D),
                               matrix="",
                               output_type="",
                               output="")

    elif request.method == "GET":

        return render_template("home.html",
                               last_inputA="",
                               last_inputB="",
                               last_inputD="",
															 matrix="",
                               output_type="",
                               output="")


# The purpose of this function is to build the matrix from the string entered by the user on the website
def build_matrix(stringMatrix, isData):

    matrix = []  # Holds the entire matrix
    b = []  # What we are solving for when dealing with data input

    rowsOfMatrix = []  # Will hold the rows of the matrix
    itemB = []

    num = ""

    # Iterate through the items in a matrix
    for i in stringMatrix:

        # If i is not a space, return or newline character we keep building up that number.
        if i != " " and i != "\r" and i != "\n":

            num += i

        # Only if it is a space do we append the value we've been building to the row of our matrix.
        elif (i == " " or i == "\r") and num != "" and num != " ":

            detVal = determineValue(num)

            if detVal == False:
                return False
            elif len(rowsOfMatrix) == 2 and isData:
                itemB.append(detVal)
            else:
                rowsOfMatrix.append(detVal)

                # We append the 1 if it is data for the b or y-intercept of the line we are going to make.
            if len(rowsOfMatrix) == 1 and isData:
                rowsOfMatrix.append(1)

            num = ""

        # If there is a newline character then we are on a new row so we append our row to matrixA list
        if i == "\n":

            matrix.append(rowsOfMatrix)

            # Append the number to the b vector
            if isData:
                b.append(itemB)

                itemB = []

            rowsOfMatrix = []

    # Only append that last number if i is not a space or nothing in the string.
    if i != " " and i != "" and i != "\r" and i != "\n" and i != ")" and i != "(" and i != num[
            len(num) - 1]:

        # Append the last number to b vector because it will be apart of it
        if isData:
            b.append([determineValue(i)])
        else:
            rowsOfMatrix.append(determineValue(i))

    elif num != " " and num != "" and num != "\r" and num != "\n" and num != ")" and num != "(":
        # Append the last number to b vector because it will be apart of it
        if isData:
            b.append([determineValue(num)])
        else:
            rowsOfMatrix.append(determineValue(num))

    matrix.append(rowsOfMatrix)

    # If we are dealing with data we also return our b vector
    if isData:
        return matrix, b
    else:
        return matrix


# This functions purpose is to determine the value from the string of the matrix the user input on the website.
def determineValue(num):

    try:

        value = float(num)

    except ValueError:

        try:

            num = num.replace("^", "**")

            value = sp.parse_expr(num)

        except:

            flash(
                f"Could not parse the equation: {num} in your matrix, make sure to follow the input criteria and resend your matrix."
            )
            return False

    return value


# This functions purpose is to handle multiplying two matrices.
def multiply_matrix(matrixA, matrixB):

    try:

        if matrixA.shape[1] != matrixB.shape[0]:
            flash(
                "Cannot do matrix product if the matrix A's columns are not the same length as matrix B's rows.",
                "error")
            return "False"

        multiply_result = np.dot(matrixA, matrixB)

        print(multiply_result)

        return multiply_result

    except:
        flash(
            "Something went wrong while computing A * B, please check your input matrices and try again.",
            "error")
        return "False"


# This functions purpose is to subtract two matrices of equal size/shape
def matrix_subtraction(matrixA, matrixB):

    try:

        if matrixA.shape != matrixB.shape:
            flash(
                f"Cannot do matrix subtraction if the matrix A's shape is not the same as matrix B's shape. A = {matrixA.shape} B = {matrixB.shape}, A not equal to B's shape and vice versa.",
                "error")
            return "False"

        subtract_result = np.subtract(matrixA, matrixB)

        print(subtract_result)

        return subtract_result

    except:
        flash(
            "Something went wrong while computing A - B, please check your input matrices and try again.",
            "error")
        return "False"


# This functions purpose is to add two matrices of equal size/shape
def matrix_addition(matrixA, matrixB):

    try:

        if matrixA.shape != matrixB.shape:
            flash(
                f"Cannot do matrix addition if the matrix A's shape is not the same as matrix B's shape. A = {matrixA.shape} B = {matrixB.shape}, A not equal to B's shape and vice versa.",
                "error")
            return "False"

        addition_result = np.add(matrixA, matrixB)

        print(addition_result)

        return addition_result

    except:
        flash(
            "Something went wrong while computing A * B, please check your input matrices and try again.",
            "error")
        return "False"


# This functions purpose is to find the LU factorization of a square matrix seperating it into P, L, and U.
def lu_factor(matrix):

    try:
        matrix = matrix.astype("float64")

        if matrix.shape[0] != matrix.shape[1]:
            flash(
                "Matrix is not square cannot find the LU factorization of a non-square matrix.",
                "error")
            return "False"

        P, L, U = lu(matrix)

        print(P, "\n", L, "\n", U)

        return [P, L, U]

    except:
        flash(
            "Something went wrong while computing the LU factorization of the given matrix, please check your matrix input and try again.",
            "error")
        return "False"


# This functions purpose is to compute the QR factorization of a given matrix.
def qr_factor(matrix):

    try:
        matrix = matrix.astype("float64")

        if matrix.shape[0] < matrix.shape[1]:
            flash(
                "Matrix has less rows than columns, if a matrices rows m is < n columns then QR factorization is not possible.",
                "error")
            return "False"

        Q, R = np.linalg.qr(matrix)

        print(Q, "\n", R)

        return [Q, R]

    except:
        flash(
            "Something went wrong while computing the QR factorization of the matrix given, check your input and try again.",
            "error")
        return "False"


# This functions purpose is to compute the SVD factorization of a given matrix.
def svd_factor(matrix):

    try:
        matrix = matrix.astype("float64")

        S, s, D = svd(matrix)

        # We have to fix the size of s so that when multiplying S, V, and D we get the original matrix. We end up storing the fixed s inside of V variable.
        if matrix.shape[0] != matrix.shape[1]:
            V = np.zeros((matrix.shape[0], matrix.shape[1]))

            if matrix.shape[0] != 1:
                V[:matrix.shape[1], :matrix.shape[1]] = np.diag(s)

            else:
                V[0, 0] = np.diag(s)

        else:

            V = np.diag(s)

        print(S, "\n\n", V, "\n\n", D)

        return [S, V, D]

    except:
        flash(
            "Something went wrong while computing the SVD factorization of the matrix given, check your input and try again.",
            "error")
        return "False"


# This functions purpose is to find the eigen values of a matrix if they exist.
def eig_value(matrix):

    try:
        if matrix.shape[0] != matrix.shape[1]:
            flash(
                "Non-square matrices don't have eigen values. Use a square matrix to find the eigenvalues.",
                "error")
            return "False"

        eigenVals = np.linalg.eigvals(matrix)

        print("eigen values: ", eigenVals)

        return eigenVals

    except:
        flash(
            "Something went wrong while computing the eigen values of the matrix given, check your input and try again.",
            "EOFError")
        return "False"


# This functions purpose is to find the inverse of a given matrix if it meets the proper criteria.
def inverse(matrix):

    try:

        # Make sure the matrix is all the same type before calling to compute the inverse of the matrix
        matrix = matrix.astype("float64")

        if matrix.shape[0] != matrix.shape[1]:
            flash(
                "Matrix is not square cannot find the inverse of a non-square matrix.",
                "error")
            return "False"

        if np.linalg.det(matrix) == 0:
            flash(
                "Matrix is singular cannot compute the inverse of a singular matrix. This means the determinant of the matrix given was 0.",
                "error")
            return "False"

        inverse_matrix = np.linalg.inv(matrix)

        print(inverse_matrix)

        return inverse_matrix

    except:
        flash(
            "Something went wrong while computing the inverse of the matrix given, check your input matrix and try again.",
            "error")
        return "False"


# This functions purpose is to return the transpose of a given matrix
def transpose(matrix):
    try:

        transposed_matrix = np.transpose(matrix)

        print(transposed_matrix)

        return transposed_matrix

    except:

        return "False"


# This functions purpose is to return the rank of a given matrix
def rank(matrix):

    try:

        # Make sure the matrix is all the same type before calling to compute the rank
        matrix = matrix.astype("float64")

        rank = np.linalg.matrix_rank(matrix)

        print(rank)

        return rank

    except:

        return "False"


# This functions purpose is to find the lstsq solution to a set of finite data and create the bestfit line through the data returning a graph of the data to the website.
def create_bestfit(matrix, b, xData, yData):

    try:

        x = np.linalg.lstsq(matrix, b, rcond=None)[0]

        # Converting x that we solved for to a array so we can get the values for our slope and y-intercept
        newX = (np.asarray(x)).flatten()

        plotX = []
        plotY = []

        for i in xData:

            plotX.append(i[0])

        for i in yData:

            plotY.append(i[0])

        print("Slope: ", newX[0], "Y-Intercept: ", newX[1])

        # Creating the plot and saving it so the website can retrieve and show it in the output
        fig = plt.figure()
        sub1 = fig.add_subplot(111)
        sub1.plot(plotX, plotY, 'ro', label="Data Points")

        # Calculating the y values of the line
        lineY = []
        for i in plotX:
            lineY.append(newX[0] * i + newX[1])

        sub1.plot(plotX, lineY, 'b-', label="Best Fit Line")
        fig.legend()

        n = randint(1, 10000)
        fig.savefig(f"./Website/static/dataGraphs/plot{n}.jpg")

        plots.append(f"./Website/static/dataGraphs/plot{n}.jpg")

        return [newX[0], newX[1], f"/static/dataGraphs/plot{n}.jpg"]

    except:
        flash(
            "Something went wrong while computing the best fit line for your data, check your data input and try again.",
            "error")
        return "False"


# This functions purpose is to randomize a matrix of size that was given by the user.
def randomize_matrix(m, n, matrix, altMatrix, D):
		randMatrix = ""

		try:
				m = int(m)
				n = int(n)

		except ValueError:
				flash(
						"Improper m or n value given. Please enter two integer values between 1-100.",
						"error")
				return render_template("home.html",
															 last_inputA="",
															 last_inputB="",
															 last_inputD="",
															 matrix="",
															 output_type="",
															 output="")

		if (m < 1 or m > 100) or (n < 1 or n > 100):
				flash(
						"The m or n value given to randomize the matrix was out of the range 1-100, please keep the size of the matrix between 1-100.",
						"error")
				return render_template("home.html",
															 last_inputA="",
															 last_inputB="",
															 last_inputD="",
															 matrix="",
															 output_type="",
															 output="")

		if not m or not n:
				flash(
						f"Random matrix {matrix} cannot be created if m and n are not given as we do not know the size of random matrix wanted.",
						"error")
				return render_template("home.html",
															 last_inputA="",
															 last_inputB="",
															 last_inputD="",
															 matrix="",
															 output_type="",
															 output="")

		numbers = [i for i in range(-99, 101)]

		for i in range(1, m + 1):

				for j in range(1, n + 1):

						if j == n and i < m:
								randMatrix += str(choice(numbers)) + "\r\n"
						else:
								randMatrix += str(choice(numbers)) + " "

		flash(f"Randomized matrix {matrix} of size: {m}x{n}", "success")

		if matrix == 'A':
				return render_template("home.html",
															 last_inputA=str(randMatrix),
															 last_inputB=str(altMatrix),
															 last_inputD=str(D),
															 matrix="",
															 output_type="",
															 output="")
		else:
				return render_template("home.html",
															 last_inputA=str(altMatrix),
															 last_inputB=str(randMatrix),
															 last_inputD=str(D),
															 matrix="",
															 output_type="",
															 output="")


# This functions purpose is to check if the matrices given are real or not.
def check_matrices(A, B, D):

		if not A or not B:
				flash(
				"Cannot do matrix addition when there was no input for either A or B.",
				"error")
				return render_template("home.html",
															last_inputA=str(A),
															last_inputB=str(B),
															last_inputD=str(D),
															matrix="",
															output_type="",
															output="")

		return None


# This functions purpose is to check if a matrix is valid 
def check_matrix(A, alt1, alt2, matrix):

		if not A:
				flash(
				f"Cannot do operation when there was no input for matrix {matrix}.",
				"error")
				if matrix == "A":
						return render_template("home.html",
															last_inputA=str(A),
															last_inputB=str(alt1),
															last_inputD=str(alt2),
															matrix="",
															output_type="",
															output="")
				elif matrix == "B":
						return render_template("home.html",
															last_inputA=str(alt1),
															last_inputB=str(A),
															last_inputD=str(alt2),
															matrix="",
															output_type="",
															output="")

				else:
						return render_template("home.html",
															last_inputA=str(alt1),
															last_inputB=str(alt2),
															last_inputD=str(A),
															matrix="",
															output_type="",
															output="")

		return None
	
	
# This functions purpose is to create the return result and whether we are going to return the information back to the website to be displayed.
def create_result(result, A, B, D, matrixType, outputType, flashMSG):

		if result == "False":
				flash(f"There was an error when trying to compute {outputType} of matrix {matrixType}.", "error")
				return render_template("home.html",
															 last_inputA=str(A),
															 last_inputB=str(B),
															 last_inputD=str(D),
															 matrix="",
															 output_type="",
															 output="")
		else:
				flash(flashMSG, "success")
				return render_template("home.html",
															 last_inputA=str(A),
															 last_inputB=str(B),
															 last_inputD=str(D),
															 matrix=matrixType,
															 output_type=outputType,
															 output=result)

@views.route('/keeprunning')
def keep_running():
		return "<h4>Running...</h4>"