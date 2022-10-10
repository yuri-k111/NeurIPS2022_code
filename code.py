import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma,psi
from scipy.spatial.distance import cdist
sns.set()
%matplotlib inline

#parameters
theta1=0
theta2=1
sigmax=2**0.5
sigma1=10**0.5
sigma2=1
n=10000

#sample points from p(x|\theta)
def sample_points(theta1,theta2,sigmax,n):
    x=[]
    for i in range (n):
        p=np.random.random()
       # print(p)
        if p<0.5:
            x.append(np.random.normal(theta1, sigmax))
        else:
            x.append(np.random.normal(theta1+theta2, sigmax))
    return x

x=sample_points(theta1,theta2,sigmax,n)
np.savetxt('x.txt', x, delimiter=',')

#\nabla f_i
def nablafi(theta,x,i,n):
    nablogp=np.array([-theta[0]/sigma1**2,-theta[1]/sigma2**2])
    c=0.5*gaus(x[i],theta[0],2**0.5)+0.5*gaus(x[i],theta[0]+theta[1],2**0.5)
    p1=(-0.5*gaus(x[i],theta[0],2**0.5)*(theta[0]-x[i])/2-0.5*gaus(x[i],theta[0]+theta[1],2**0.5)*(theta[0]+theta[1]-x[i])/2)/c
    p2=(-0.5*gaus(x[i],theta[0]+theta[1],2**0.5)*(theta[0]+theta[1]-x[i])/2)/c
    return -(nablogp+n*np.array([p1,p2]))

#SGLD
def SGLD(B,tmx,eta):
    theta_list_sgld=[[] for i in range (tmx+1)]
    for sample_num in range (0,1000):
        print(sample_num)
        theta=np.array([np.random.normal(0,1),np.random.normal(0,1)])
        theta_list_sgld[0].append(theta)
        for t in range (0,tmx):
            v=0
            e1=np.random.normal(0,1)
            e2=np.random.normal(0,1)
            i=np.random.choice(n)
            Ik=np.random.choice(I,B,replace=False)
            for k in Ik:
                v+=nablafi(theta,x,k,n)/B
            theta=theta-eta*v+(2*eta)**0.5*np.array([e1,e2])
            theta_list_sgld[int(t+1)].append(theta)
    theta_list_sgld=np.array(theta_list_sgld)
    with open('theta_list_sgld'+f'{B}'+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(theta_list_sgld.shape))
        for data_slice in theta_list_sgld:
            np.savetxt(outfile, data_slice,delimiter=',')
            outfile.write('# New slice\n')
    #plot last iteration
    theta1_list_sgld=[]
    theta2_list_sgld=[]
    for i in range (0,1000):
        theta1_list_sgld.append(theta_list_sgld[-1][i][0])
        theta2_list_sgld.append(theta_list_sgld[-1][i][1])
    plt.scatter(theta1_list_sgld,theta2_list_sgld)
    plt.show()

#SVRG-LD
B_list=[10,1]
def SVRGLD(B,tmx,eta):
    m=int(n/B)
    I= np.array(range(0,n))
    theta_list_svrgld=[[] for i in range (tmx*m+1)]
    for sample_num in range (0,1000):
        print(sample_num)
        theta=np.array([np.random.normal(0,1),np.random.normal(0,1)])
        theta_list_svrgld[0].append(theta)
        for t in range (0,tmx):
            v=0
            for i in range (n):
                v+=nablafi(theta,x,i,n)/n
            G=v
            e1=np.random.normal(0,1)
            e2=np.random.normal(0,1)
            thetab=theta
            theta=theta-eta*G+(2*eta)**0.5*np.array([e1,e2])
            theta_list_svrgld[int(t*m+1)].append(theta)
            for j in range (1,m):
                Ik=np.random.choice(I,B,replace=False)
                v=0
                e1=np.random.normal(0,1)
                e2=np.random.normal(0,1)
                for k in Ik:
                    v+=(nablafi(theta,x,k,n)-nablafi(thetab,x,k,n)+G)/B
                theta=theta-eta*v+(2*eta)**0.5*np.array([e1,e2])
                theta_list_svrgld[int(t*m+j+1)].append(theta)
    theta_list_svrgld=np.array(theta_list_svrgld)
    with open('theta_list_svrgld'+f'{B}'+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(theta_list_svrgld.shape))
        for data_slice in theta_list_svrgld:
            np.savetxt(outfile, data_slice,delimiter=',')
            outfile.write('# New slice\n')
    #plot last iteration
    theta1_list_svrgld=[]
    theta2_list_svrgld=[]
    for i in range (0,1000):
        theta1_list_svrgld.append(theta_list_svrgld[-1][i][0])
        theta2_list_svrgld.append(theta_list_svrgld[-1][i][1])
    plt.scatter(theta1_list_svrgld,theta2_list_svrgld)
    plt.show()

#SARAH-LD
def SARAHLD(B,tmx,eta):
    m=int(n/B)
    I= np.array(range(0,n))
    theta_list_sarahld=[[] for i in range (tmx*m+1)]
    for sample_num in range (0,1000):
        print(sample_num)
        theta=np.array([np.random.normal(0,1),np.random.normal(0,1)])
        theta_list_sarahld[0].append(theta)
        for t in range (0,tmx):
            v=0
            for i in range (n):
                v+=nablafi(theta,x,i,n)/n
            G=v
            e1=np.random.normal(0,1)
            e2=np.random.normal(0,1)
            thetab=theta
            theta=theta-eta*G+(2*eta)**0.5*np.array([e1,e2])
            theta_list_sarahld[int(t*m+1)].append(theta)
            for j in range (1,m):
                Ik=np.random.choice(I,B,replace=False)
                v=0
                e1=np.random.normal(0,1)
                e2=np.random.normal(0,1)
                for k in Ik:
                    v+=(nablafi(theta,x,k,n)-nablafi(thetab,x,k,n)+G)/B
                thetab=theta
                theta=theta-eta*v+(2*eta)**0.5*np.array([e1,e2])
                G=v
                theta_list_sarahld[int(t*m+j+1)].append(theta)
    theta_list_sarahld=np.array(theta_list_sarahld)
    with open('theta_list_sarahld'+f'{B}'+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(theta_list_sarahld.shape))
        for data_slice in theta_list_sarahld:
            np.savetxt(outfile, data_slice,delimiter=',')
            outfile.write('# New slice\n')
    theta1_list_svrgld=[]
    theta2_list_svrgld=[]
    for i in range (0,1000):
        theta1_list_svrgld.append(theta_list_svrgld[-1][i][0])
        theta2_list_svrgld.append(theta_list_svrgld[-1][i][1])
    plt.scatter(theta1_list_svrgld,theta2_list_svrgld)
    plt.show()

#MALA: Estimation of the true posterior
eta=0.0001
theta_list=[]

def q_tilde_divided_by_q(x,y,nablogx,nablogy):
    c=np.linalg.norm(x-y+eta*nablogy)**2-np.linalg.norm(y-x+eta*nablogx)**2
    return np.exp(-c/(4*eta))
def pi_tilde_divided_by_pi(theta,theta_tilde,x,sigma1,sigma2):
    ratio=1
    for i in range (0,len(x)):
        c=(0.5*gaus(x[i],theta_tilde[0],2**0.5)+0.5*gaus(x[i],theta_tilde[0]+theta_tilde[1],2**0.5))/(0.5*gaus(x[i],theta[0],2**0.5)+0.5*gaus(x[i],theta[0]+theta[1],2**0.5))
        ratio=ratio*c
    ratio=ratio*gaus(theta_tilde[0],0,sigma1)*gaus(theta_tilde[1],0,sigma2)/(gaus(theta[0],0,sigma1)*gaus(theta[1],0,sigma2))
    #print(ratio)
    return ratio
def MALA(eta,tmx):
    for sample_num in range (0,1000):
        print(sample_num)
        theta=np.array([np.random.normal(0,1),np.random.normal(0,1)])
        v=0
        for i in range (n):
            v+=nablafi(theta,x,i,n)/n
        for t in range (0,tmx):
            e1=np.random.normal(0,1)
            e2=np.random.normal(0,1)
            theta_tilde=theta-eta*v+(2*eta)**0.5*np.array([e1,e2])
            v_tilde=0
            for i in range (n):
                v_tilde+=nablafi(theta_tilde,x,i,n)/n
            p=pi_tilde_divided_by_pi(theta,theta_tilde,x,sigma1,sigma2)
            q=q_tilde_divided_by_q(theta,theta_tilde,v,v_tilde)
            product=p*q
            alpha=min(1, product)
            u=np.random.random()
            if u<=alpha:
                theta=theta_tilde
                v=v_tilde
        theta_list.append(theta)
    theta1_list=[]
    theta2_list=[]
    for i in range (0,len(theta_list)):
        theta1_list.append(theta_list[i][0])
        theta2_list.append(theta_list[i][1])
    np.savetxt('theta1_list_mala.txt', theta1_list, delimiter=',')
    np.savetxt('theta2_list_mala.txt', theta2_list, delimiter=',')
    plt.scatter(theta1_list,theta2_list)

#Function to estimate KL-divergence
#From https://github.com/blakeaw/Python-knn-entropy/blob/master/knn_entropy.py
# And F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures for Continuous Random Variables. Advances in Neural Information
# Processing Systems 21 (NIPS). Vancouver (Canada), December.
# https://papers.nips.cc/paper/3417-estimation-of-information-theoretic-measures-for-continuous-random-variables.pdf

def kth_nearest_neighbor_distances_self(X, k=1):

    #length
    nX = len(X)
    #make sure X has the right shape for the cdist function
    X = np.reshape(X, (nX,-1))
    dists_arr = cdist(X, X)
    #sorts each row
    dists_arr.sort()
    return [dists_arr[i][k] for i in range(nX)]

def kth_nearest_neighbor_distances(X,Y, k=1):

    #length
    nX = len(X)
    nY=len(Y)
    #make sure X has the right shape for the cdist function
    X = np.reshape(X, (nX,-1))
    Y = np.reshape(Y, (nY,-1))
    dists_arr = cdist(X, Y)
    #sorts each row
    dists_arr.sort()
    return [dists_arr[i][k-1] for i in range(nX)]


def KL(X,Y, k=1):
    r_k = np.array(kth_nearest_neighbor_distances_self(X, k=k))
    s_k= np.array(kth_nearest_neighbor_distances(X,Y, k=k))
    n = len(X)
    m=len(Y)
    d = 1
    if len(X.shape) == 2:
        d = X.shape[1]
    #volume of the unit ball
    v_unit_ball = np.pi**(0.5*d)/gamma(0.5*d + 1.0)
    # probability estimator using knn distances
    # log probability
    log_ratio = np.log(s_k/r_k)
    #entropy estimator
    h_k_hat = d*log_ratio.sum() / (1.0*n)+np.log(m/(n-1))
    return h_k_hat
B_list=[100,1000,10000]
eta=0.00001
tmx=20
for B in B_list:
    SVRGLD(B,tmx,eta)
    
B=100
eta=0.00001
tmx=20
SARAHLD(B,tmx,eta)

B=100
eta=0.00001
tmx=4000
SGLD(B,tmx,eta)

eta=0.0001
tmx=400
MALA(eta,tmx)

#plot result
marker_list={100:"o",1000:"^",10000:"s"}
X=np.loadtxt("theta_list_mala.txt",delimiter=",")
B_list=[100,1000,10000]
t=np.arange(21)*20000
for B in B_list:
    print(B)
    tm=int(10000/B)
    Y_hist=np.loadtxt("theta_list_svrgld"+f"{B}"+".txt",delimiter=",")
    Y_hist=Y_hist.reshape((int(200000/B+1),1000,2))
    KL_list=[]
    Y=Y_hist[0]
    KL_list.append(KL(X,Y,k=1))
    for i in range (20):
        Y=Y_hist[int((i+1)*tm)]
        KL_list.append(KL(X,Y,k=1))
    plt.plot(t,KL_list,label="SVRG-LD, B="+f"{B}",marker=marker_list[B], linestyle='--')

B=100
Y_hist=np.loadtxt("theta_list_sgld100.txt",delimiter=",")
Y_hist=Y_hist.reshape((int(4000+1),1000,2))
KL_list=[]
Y=Y_hist[0]
KL_list.append(KL(X,Y,k=1))
for i in range (20):
    Y=Y_hist[int((i+1)*200)]
    KL_list.append(KL(X,Y,k=1))
plt.plot(t,KL_list,label="SGLD, B="+f"{B}",marker='o', linestyle='--')

plt.legend()
plt.xlabel("Gradient Computation")
plt.ylabel("KL-divergence")
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig("KL-divergence.png")
plt.show()
