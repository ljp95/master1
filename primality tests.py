# COMPLEX PROJECT
# Created by ilyas Aroui and Jean-Paul Lam  on 19 / Nov / 2018

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from math import floor, sqrt
from matplotlib.legend_handler import HandlerLine2D


def my_gcd(a, b):
    """
    :param a: a  integer
    :param b: a  integer
    :return: the great common divisor between a and b
    """
    if b > a:
        a, b = b, a
    remainder = b
    while remainder != 0:
        remainder = a % b
        a = b
        b = remainder
    return a


def my_inverse(a, N):
    """ Find inverse of a modulo N.
    :param a: an integer
    :param N: an integer
    :return:
        u0: the modulo N inverse of a such that a*u0 is congurent to 1 modulo N
    """
    if a < N:
        u0 = 0
        u1 = 1
        b = a
        a = N
    else:
        u0 = 1
        u1 = 0
        b = N
    # after initializing properly a and b we apply extended Euclid algorithm on (a, b) and return u0 such that
    # a*u0 is cogurent to 1 modulo b
    remainder = b
    while remainder != 0:
        q = a // b
        remainder = a - q * b
        a = b
        b = remainder
        u2 = u0 - q * u1
        u0 = u1
        u1 = u2
    if a != 1:
        return "error"
    return u0


def my_expo_mod(g, n, N):
    """
    :param g: a positve integer
    :param n: a positive integer
    :param N: a positive integer
    :return:
        h : g^n modulo N
    """
    binary_representation = bin(n)[2:]
    h = 1
    for i in range(len(binary_representation)):
        h = (h**2) % N
        if binary_representation[i] == '1':
            h = (h*g) % N
    return h


def complexity_test():
    """ plot the complexity of my_gcd(a, N) and my_inverse(a, N) versus bitsize(a)* bitsize(N).
        Theoretically, the complexity of both functions is O( bitsize(a)* bitsize(N)).
    :return:
        the plot of the running time of both functions versus bitsize(a)* bitsize(N).
    """
    timing_gcd = []
    timing_inverse = []
    for i in range(2000):
        # we discretize the interval to assure that we sample from all the ranges of bits
        if i <= 500:
            b1 = random.randint(4, 16)
            b2 = random.randint(b1, 16) #  assure that N > a
        elif (i > 500) and (i <= 1000):
            b1 = random.randint(16, 32)
            b2 = random.randint(b1, 32)
        elif (i > 1000) and (i <= 1500):
            b1 = random.randint(32, 64)
            b2 = random.randint(b1, 64)
        else:
            b1 = random.randint(64, 128)
            b2 = random.randint(b1, 128)
        t_gcd_elapsed = 0
        t_inverse_elapsed = 0
        if i % 100 == 0:
            print(i)
        for j in range(3000):
            a = 0
            N = 0
            while a == 0:
                a = random.getrandbits(b1)

            while N == 0 or N == a:  # avoid these cases as the running time will be instantaneous
                N = random.getrandbits(b2)
            t_gcd = time.time()
            g = my_gcd(a, N)
            t_gcd_elapsed += time.time() - t_gcd

            t_inverse = time.time()
            g2 = my_inverse(a, N)
            t_inverse_elapsed += time.time() - t_inverse

        timing_gcd.append([(b1 * b2) / 32, t_gcd_elapsed / 3000])
        timing_inverse.append([(b1 * b2) / 32, t_inverse_elapsed / 3000])

    T = np.array(sorted([t for t in timing_gcd if t[1] > 0])) # as we sample randomly, so b1 and b2 may not be in order
    T2 = np.array(sorted([t for t in timing_inverse if t[1] > 0]))
    plt.figure()
    line2, = plt.plot(T[:, 0], T[:, 1], 'b', label='my_gcd(a, b)')
    line1, = plt.plot(T2[:, 0], T2[:, 1], 'r', label='my_inverse(a, b)')
    plt.title('Temps de calcul en fonction de taille binaire d'' entrée')
    plt.ylabel('Temps (secondes)')
    plt.xlabel('taille(a) * taille(b)')
    plt.legend(handler_map={line2: HandlerLine2D()})

    plt.show()


def first_test(n):
    """
    :param n: an integer
    :return:
        "prime" if it is a prime.
        composite if it is not a prime.
    """
    for k in range(2, int(floor(sqrt(n)) + 1)):
        if n % k == 0:
            return "composite"
    return "prime"


def carmichael_1(N):
    """ for all impair numbers n  less than N, if n is composite proceed to check if it passes the Fermat test.
    :param N: The maximal number to which we look for a carmichael number.
    :return:
        list_carmichael: list of all carmichael numbers less then N
    """
    list_carmichael = []
    for n in range(3, N, 2):
        if first_test(n) == "composite":
            a = 2
            ok = 1
            while a != n and ok == 1:
                if my_gcd(n, a) == 1:
                    if my_expo_mod(a, n, n) != a:
                        ok = 0
                a += 1
            if a == n and ok == 1:
                list_carmichael.append(n)
    return list_carmichael


def carmichael_2(N):
    """ for all impair numbers n  less than N, if n passes the Fermat test proceed to check if it is composite or not.
    :param N: The maximal number to which we look for a carmichael number.
    :return:
        list_carmichael: list of all carmichael numbers less then N
    """
    list_carmichael = []
    for n in range(3, N, 2):
        pseudo_prime = True
        for a in range(2, n):
            if my_gcd(a, n) == 1:
                if my_expo_mod(a, n-1, n) != 1:
                    pseudo_prime = False
                    break
        if pseudo_prime is True:
            if first_test(n) == "composite":
                list_carmichael.append(n)
    return list_carmichael


def carmichael_3(N, primes):
    """for all n in range from 3 to N such that: n is impair and not prime, check if n passes the fermat test.
    :param N: The maximal number to which we look for a carmichael number.
    :param primes: list of all primes less than N
    :return:
        list_carmichael : list of all carmichael numbers less than N
    """
    # file = open("primes1.txt", "r")
    # lines = file.readlines()
    # primes = np.array([int(prime) for l in lines for prime in l.split() if int(prime) < N])
    list_carmichael = []
    possible_n = list(set(list(range(3, N, 2))) - set(primes))

    for n in possible_n:
        a = 2
        while a < n:
            if my_gcd(n, a) == 1:
                if my_expo_mod(a, n, n) != a:
                    break
            a += 1
        if a == n:
            list_carmichael.append(n)

    return list_carmichael


def carmichael_complexity():
    """ plot the complexity of carmichael_1(N) and carmichael_2(N) versusN.
    :return:
        the plot of the running time of both functions versus N.
    """
    N= [2000, 3000, 4000,5000, 8000, 10000, 30000, 50000, 80000, 100000]
    T2 = []
    T1 = []
    for n in N:
        print('...', n)
        t2 = time.time()
        l2 = carmichael_2(n)
        T2.append(time.time() - t2)
        t1 = time.time()
        l1 = carmichael_1(n)
        T1.append(time.time() - t1)
        if len(l1) != len(l2):
            print("Error")
            return
    line1,  = plt.plot(N, T1, 'r--', label= 'Testant d''abord si le nombre est composé.')
    line2,  = plt.plot(N, T2, 'b--', label = 'Testant d''abord si le nombre valide fermat test.')
    plt.legend(handler_map={line1: HandlerLine2D()})
    plt.show()


def gen_carmichael_1(n):
    """ first find all primes less then n. Then, find the list of all carmichael less then n using carmichael_1.
    lastly, select all carmichael numbers with 3 factors from the total list of carmichael numbers.
    :param n: The maximal number to which we look for a carmichael numbers with 3 prime factors.
    :return:
        list_carmichael_3_factors : list of all carmichael numbers with 3 prime factors.
    """
    list_carmichael_3_factors =[]
    count_primes = 0
    primes = []
    for i in range(2, n):
        if first_test(i) == "prime":
            count_primes += 1
            primes.append(i)
    list_carmichael = carmichael_1(n)
    for carmichael_number in list_carmichael:
        counter = 0
        temp_number = carmichael_number
        for p in primes:
            if temp_number % p == 0:
                temp_number = temp_number//p
                counter += 1
            if p > temp_number or counter == 3:
                break

        if counter == 3 and temp_number == 1:
            list_carmichael_3_factors.append(carmichael_number)
    return list_carmichael_3_factors


def gen_carmichael_2(n):
    """ first find all primes less then n. Then, find the list of all carmichael less then n using carmichael_2.
    lastly, select all carmichael numbers with 3 factors from the total list of carmichael numbers.
    :param n: The maximal number to which we look for a carmichael numbers with 3 prime factors.
    :return:
        list_carmichael_3_factors : list of all carmichael numbers with 3 prime factors.
    """
    list_carmichael_3_factors =[]
    count_primes = 0
    primes = []
    for i in range(2, n):
        if first_test(i) == "prime":
            count_primes += 1
            primes.append(i)
    list_carmichael = carmichael_3(n, primes)
    for carmichael_number in list_carmichael:
        counter = 0
        temp_number = carmichael_number
        for p in primes:
            if temp_number % p == 0:
                temp_number = temp_number//p
                counter += 1
            if p > temp_number or counter == 3:
                break

        if counter == 3 and temp_number == 1:
            list_carmichael_3_factors.append(carmichael_number)
    return list_carmichael_3_factors


def gen_carmichael_complexity():
    """ plot the complexity of gen_carmichael_1(N) and gen_carmichael_2(N) versusN.
    :return:
        the plot of the running time of both functions versus N.
    """
    N= [6000, 7000, 8000, 10000, 15000, 20000, 25000, 70000]
    T2 = []
    T1 = []
    for n in N:
        print('...', n)
        t2 = time.time()
        l2 = gen_carmichael_2(n)
        T2.append(time.time() - t2)
        t1 = time.time()
        l1 = gen_carmichael_1(n)
        T1.append(time.time() - t1)
        if len(l1) != len(l2):
            print("Error")
            return
    line1,  = plt.plot(N, T1, 'r--', label= 'En utilisant carmichael_l(n)')
    line2,  = plt.plot(N, T2, 'b--', label = 'En utilisant carmichael_3(n, primes)')
    plt.legend(handler_map={line1: HandlerLine2D()})
    plt.show()


def gen_carmichael_p(p):
    """ first find all primes less then n. Then, find the list of all carmichael less then n using carmichael_1.
    lastly, select all carmichael numbers with 3 factors from the total list of carmichael numbers.
    :param p: the first factor of all carmichael numbers with 3 factors written as p*q*r
    :return:
        list_p_carmichael : list of all carmichael numbers with 3 prime factors p*q*r.
    """
    list_p_carmichael = []
    prime_max = p * (2*p *(p - 1) + 1)
    carmichael_max = prime_max ** 2
    primes = []
    for i in range(p + 2, prime_max):
        if first_test(i) == "prime":
            primes.append(i)

    list_carmichael = carmichael_1(carmichael_max)

    """do prime factorizations for all carmichael numbers found and take those with 3 factors only
    """
    for carmichael_number in list_carmichael:
        if carmichael_number % p == 0:
            counter = 0
            temp_number = carmichael_number
            temp_number //= p
            for pm in primes:
                if temp_number % pm == 0:
                    temp_number = temp_number // pm
                    counter += 1
                if p > temp_number or counter == 2:
                    break

            if counter == 2 and temp_number == 1:
                list_p_carmichael.append(carmichael_number)
    return list_p_carmichael


def fermat_test(n, k):
    """test whether n is probably prime or not using k iterations to choose the base of fermat test.

    :param n: an integer
    :param k: number of iteration to apply the fermat test.
    :return:
        "composite": if it fails one test.
        "probably prime": if it passes all the k tests.
    """
    for i in range(1, k):
        a = random.randrange(2, n-2)
        if my_expo_mod(a, n, n) != a:
            return n, "composite"

    return n, "probably prime"


def tester_fermat_test():
    """
    Test the fermat_test(n, k) on different instances
    carmichael numbers, composite numbers and random numbers and see how it performs
    :return:
        print to the screen the results
    """
    c_list = carmichael_1(100000)
    r_list = [random.randrange(2, 1.e8) for i in range(1000)]
    com_list = [n for n in r_list if first_test(n) == 'composite']
    for c in c_list:
        print(fermat_test(c, 100))
    for r in r_list:
        print(fermat_test(r, 100))
    for com in com_list:
        print(fermat_test(com, 100))


def fermat_test_success_probability(max_number):
    """ calculate the error probability of different values of k.
    :param max_number: the biggest prime number tested
    :return:
        "probably prime": if it passes all the test in k iterations.
        "composite": if it fails one test.
    """
    list_proba = []
    for k in range(2, 21):
        total_count = 0
        failure_count = 0
        print('...k=', k)
        for n in range(5, max_number):
            number, test = fermat_test(n, k)
            if test == "probably prime":
                total_count += 1
                if first_test(number) == 'composite':
                    failure_count += 1
        list_proba.append(failure_count / total_count)

    plt.plot(range(2, 21), list_proba)
    plt.show()


def miller_rabin_test(n, k):
    """test whether n is probably prime or not using k iterations to choose the base of miller test.

    :param n: an integer
    :param k: number of iteration to apply the miller test.
    :return:
        "composite": if it fails a test.
        "probably prime": if it passes all the k tests.
    """
    h, m = 0, n - 1
    while m % 2 == 0:
        h += 1
        m //= 2
    for i in range(k):
        a = random.randrange(2, n - 1)
        b = my_expo_mod(a, m, n)
        if b != 1 and b != n - 1:
            j = 1
            while (j < h) and (b != n - 1):
                if my_gcd(b * 2, n) == 1:
                    return n, 'composite'
                b = my_gcd(b * 2, n)
                j += 1
            if b != n - 1:
                return n, 'composite'
    return n, 'probably prime'


def tester_millier_test():
    """
    Test the miller_rabin_test(n, k) on different instances
    carmichael numbers, composite numbers and random numbers and see how it performs
    :return:
        print to the screen the results
    """
    c_list = carmichael_1(100000)
    r_list = [random.randrange(2, 1.e8) for i in range(1000)]
    com_list = [n for n in r_list if first_test(n) == 'composite']
    for c in c_list:
        print(miller_rabin_test(c, 1))
    # for r in r_list:
    #     print(miller_rabin_test(c, 1))
    # for com in com_list:
    #     print(miller_rabin_test(c, 1))


def miller_rabin_test_success_probability(max_number):
    """ calculate the error probability of different values of k.
    :param max_number: the biggest prime number tested
    :return:
        "probably prime": if it passes all the test in k iterations.
        "composite": if it fails one test.
    """
    list_proba = []
    for k in range(2, 21):
        total_count = 0
        failure_count = 0
        print('...k=', k)
        for n in com_list:
            number, test = miller_rabin_test(n, k)
            if test == "probably prime":
                total_count += 1
                if first_test(number) == 'composite':
                    failure_count += 1
        list_proba.append(failure_count / total_count)

    plt.plot(range(2, 21), list_proba)
    plt.show()


def gen_rsa(t):
    """
    1. pick randomly p and q in range(2**(t-1), 2**t) and perform miller_test on them.
    2. if they are composite repeat 1, else return p*q
    :param t: a positive integer
    :return: return p*q
    """
    p, q = random.randrange(2 ** (t - 1), 2 ** t), random.randrange(2 ** (t - 1), 2 ** t)
    a, b = miller_rabin_test(p, 1)
    while b == 'composite':
        p = random.randrange(2 ** (t - 1), 2 ** t)
        a, b = miller_rabin_test(p, 1)
    a, b = miller_rabin_test(q, 1)
    while b == 'composite':
        q = random.randrange(2 ** (t - 1), 2 ** t)
        a, b = miller_rabin_test(p, 1)
    return p * q


