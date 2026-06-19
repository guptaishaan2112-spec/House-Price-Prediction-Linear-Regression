#coding the cost function and gradient descent for single and multiple regression from scratch without using sklearn
import numpy as np
x=np.array([
    [2104,5,1],
    [1416,3,4]
    ])
w=np.array([200,0.1,0.1])
y=np.array([460,232])
#cost func for one parameter
def cost_funct(x,y,w,b):
    x=x[:,0] 
    e_sq_sum=0
    for i in range(x.shape[0]):
        error=w*x[i]+b-y[i]
        e_sq=error**2
        e_sq_sum+=e_sq
    func=e_sq_sum/(2*(x.shape[0]))
    return func
def grad_desc(w,b,x,y,value):
    x=x[:,0]
    deriv_for_w=0
    deriv_for_b=0
    for i in range(x.shape[0]):
        a=(w*x[i]+b-y[i])*x[i]
        b1=(w*x[i]+b-y[i])
        deriv_for_w=deriv_for_w+a
        deriv_for_b=deriv_for_b+b1
    w_best=w-0.1*deriv_for_w/(x.shape[0])
    b_best=b-0.1*deriv_for_b/(x.shape[0])
    pred=w_best*value+b_best
    return [w_best,b_best], pred
def cost_funct_2(x,y,w,b):
    t_sum=0
    for i in range(x.shape[0]):
        sum=(np.dot(w,x[i]) + b - y[i])**2
        t_sum=t_sum + sum
    j=t_sum/(2*(x.shape[0]))
    return j
def grad_desc_2(x,y,b,w,iter):#multiple parameters
    sum=0
    for i in range(iter):
        err= np.dot(w,x) + b - y
        dw= (x.T @ err)/x.shape[0]
        dj_db = (1/x.shape[0]) * np.sum(err)
        w = w - 0.01 * dw
        b = b - 0.01 * dj_db
    return w,b
        
#rough idea of what is happening behind the model


