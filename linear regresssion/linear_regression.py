import numpy as np

def compute_total_error(points,initial_m,initial_b):
    n=len(points)
    total_error=0
    for g in range(n):
        x=points[g,0]
        y=points[g,1]
        total_error=total_error+(y-(initial_m*x+initial_b))**2
    return total_error/n

def gradient_descent(points,initial_m,initial_b,number_of_iterations,learning_rate):
    n=len(points)
    m,b=initial_m,initial_b
    for g in range(number_of_iterations):
        m_gradient=0
        b_gradient=0
        for j in range(n):
            x=points[j,0]
            y=points[j,1]
            m_gradient+=-(2/n)*(y-(m*x+b))*x
            b_gradient+=-(2/n)*(y-(m*x+b))
        m=m-(learning_rate*m_gradient)
        b=b-(learning_rate*b_gradient)
    return m,b

if __name__=='__main__':
    points=np.genfromtxt("data.csv",delimiter=",")
    learning_rate=0.0001
    initial_m=0
    initial_b=0
    number_of_iterations=1000
    print("Total error in the beginning->"+str(compute_total_error(points,initial_m,initial_b)))
    m,b=gradient_descent(points,initial_m,initial_b,number_of_iterations,learning_rate)
    print("Total error in the end->"+str(compute_total_error(points,m,b)))
    print("m->"+str(m))
    print("b->"+str(b))
