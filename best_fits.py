import numpy as np
from numba import jit,njit

a=0.018
delta=0.7

@njit
def set_seed(value):
    np.random.seed(value)

@njit('float64(float64, float64, float64)')
def pow_samp(min1, max1, p):
    '''
    Generate a random from a truncated power law PDF with power law index p. 
    min1 and max1

    '''
    r=np.random.uniform(0, 1)
    if p==1:
        return min1*np.exp(r*np.log(max1/min1))
    else:
        return (r*(max1**(1.-p)-min1**(1.-p))+min1**(1.-p))**(1./(1-p))

@njit('float64(float64, float64)')
def eta(m,p):
    # eta1=0.6-0.7/(np.log10(p)-0.5)
    # if np.log10(p) < 0.7:
    #     eta1=-0.1
    # eta2=0.9-0.2/(np.log10(p)-0.5)
    # if np.log10(p) < 0.7:
    #     eta2=-2.9

    eta1=0.6-0.7/(np.log10(p)-0.5)
    eta2=0.9-0.2/(np.log10(p)-0.5)
    eta_interp=np.interp(m, [3,7], [eta1, eta2])

    if m<3:
        return eta1
    elif m>7:
        return eta2
    else:
        return eta_interp

@njit('float64(float64, float64, float64)')
def cdf_inv_ecc(x, eta, emax):
    """
    Inverse of cdf for eccentricity distribution with linear turnover...
    """
    e1 = 0.8 * emax
    e2 = emax
    
    norm = 1.0 / (e1 / (eta + 1) + e2 - 0.5 * (e2**2. - e1**2.) / (e2 - e1))
    tp = e1 / (eta + 1) * norm

    if x < tp:
        return (x * (eta + 1) / (norm) * e1**eta)**(1.0 / (eta + 1))
    else:
        n1 = norm   
        return         (e2*(n1 - np.sqrt((n1*(e1*(-1 + eta)*n1 - (1 + eta)*(e2*n1 - 2*x)))/\
                                         ((e1 - e2)*(1 + eta)))) + \
                        e1*np.sqrt((n1*(e1*(-1 + eta)*n1 - (1 + eta)*(e2*n1 - 2*x)))/\
                                  ((e1 - e2)*(1 + eta))))/n1

@njit('float64(float64, float64)')
def gen_e(m, p):
    emax=(1.-(p/2.)**(-2./3.))
    eta_=eta(m,p)

    if eta_<-1 or np.log10(p)<0.5:
        return 0
    else:
        x = np.random.uniform(0, 1)
        return cdf_inv_ecc(x, eta_, emax)

@njit('float64(float64)')
def gamma_large_q_solar(p):
    lp=np.log10(p)
    if  lp<5.0:
        return -0.5
    else:
        return -0.5-0.3*(np.log10(p)-5.0)

@njit('float64(float64)')
def gamma_large_q_B(p):
    lp=np.log10(p)
    if lp<1.0:
        return -.5
    elif lp<4.5:
        return -0.5-0.2*(lp-1.0)
    elif lp<6.5:
        return -1.2-0.4*(lp-4.5)
    else:
        return -2.0

@njit('float64(float64)')
def gamma_large_q_O(p):
    lp=np.log10(p)
    if  lp<1.:
        return -.5
    elif lp<2:
        return -0.5-0.9*(lp-1.0)
    elif lp<4:
        return -1.4-0.3*(lp-2.0)
    else:
        return -2.0

@njit('float64(float64, float64)')
def gamma_large_q(m, p):
    g1=gamma_large_q_solar(p)
    g2=gamma_large_q_B(p)
    g3=gamma_large_q_O(p)
    if m<1.2:
        return g1
    elif m<3.5:
        return np.interp(m, [1.2, 3.5], [g1, g2])
    elif m<6:
        return np.interp(m, [3.5, 6.0], [g2, g3])
    else:
        return g3

@njit('float64(float64)')
def gamma_small_q_solar(p):
    return 0.3

@njit('float64(float64)')
def gamma_small_q_B(p):
    lp=np.log10(p)
    if lp<2.5:
        return .2
    elif lp<5.5:
        return 0.2-0.3*(lp-2.5)
    else:
        return -0.7-0.2*(lp-5.5)

@njit('float64(float64)')
def gamma_small_q_O(p):
    lp=np.log10(p)
    if lp<1:
        return .1
    elif lp<3:
        return 0.1-0.15*(lp-1.0)
    elif lp<5.6:
        return -0.2-0.5*(lp-3.0)
    else:
        return -1.5

@njit('float64(float64, float64)')
def gamma_small_q(m, p):
    g1=gamma_small_q_solar(p)
    g2=gamma_small_q_B(p)
    g3=gamma_small_q_O(p)
    if m<1.2:
        return g1
    elif m<3.5:
        return np.interp(m, [1.2, 3.5], [g1, g2])
    elif m<6:
        return np.interp(m, [3.5, 6.0], [g2, g3])
    else:
        return g3

@njit('float64(float64, float64, float64)')
def pdf_q(q, gsq, glq):
    if q<0.3:
        return (q/0.3)**gsq
    else:
        return (q/0.3)**glq

@njit('float64(float64)')
def f_twin_short(m):
    return 0.3-0.15*np.log10(m)

@njit('float64(float64)')
def lp_twin_max(m):
    if m<6.5:
        return 8.0-m
    else:
        return 1.5

@njit('float64(float64, float64)')
def f_twin(m, p):
    lp=np.log10(p)
    lp_twin_max1=lp_twin_max(m)
    f_twin_short1=f_twin_short(m)
    if lp<1:
        return f_twin_short1
    elif lp<lp_twin_max1:
        return f_twin_short1*(1-(lp-1)/(lp_twin_max1-1))
    else:
        return 0.

@njit('float64(float64, float64, float64)')
def get_corr(m, p, qmin):
    """
    Ratio between the total number of binaries and those with mass ratios >= 0.3
    """
    gsq=gamma_small_q(m, p)
    glq=gamma_large_q(m, p)
    twin1=f_twin(m, p)
    x=(1+gsq)/(1+glq)*((1./0.3)**(1+glq)-1.)/(1-(qmin/0.3)**(1.+gsq))

    # return (1.-(1-twin1)/(1+x))**-1.
    return 1. + (1. - twin1) / x

@njit('float64(float64, float64, float64)')
def gen_q(m, p, qmin):
    gsq=gamma_small_q(m, p)
    glq=gamma_large_q(m, p)
    ##Twin fraction is defined with respect to the high mass ratio binaries.
    twin1=f_twin(m, p) / get_corr(m, p, qmin)
    if np.random.uniform(0,1)<twin1:
        return np.random.uniform(0.95, 1)

    accept=False
    while not accept:
        trial=np.random.uniform(qmin, 1)
        max_pdf=max(pdf_q(qmin, gsq, glq), pdf_q(0.3, gsq, glq), pdf_q(1.0, gsq, glq))

        if np.random.uniform(0,max_pdf)<=pdf_q(trial,gsq,glq):
            accept=True
    return trial

@njit('float64(float64)')
def f1(m):
    return 0.02+0.04*np.log10(m)+0.07*np.log10(m)**2.

@njit('float64(float64)')
def f2(m):
    return 0.039+0.07*np.log10(m)+0.01*np.log10(m)**3.

@njit('float64(float64)')
def f3(m):
    return 0.078-0.05*np.log10(m)+0.04*np.log10(m)**2.

@njit('float64(float64, float64, float64)')
def pdf_lp(m, lp, qmin):
    """
    PDF of log period of binaries with mass ratios down to qmin

    :param float m: Primary mass
    :param float lp: Log period
    :param float qmin: Minimum mass ratio

    :return: pdf
    :rtype: float
    """
    ##Correction for low mass ratio binaries
    c1=get_corr(m, 10.**lp, qmin)
    c2 = 1.0 - 0.11 * (lp - 1.5) ** 1.43 * (m / 10.0) ** 0.56
    if lp <= 1.5:
        c2 = 1.0
    if c2 < 0:
        c2 = 0
    c1 = c1 * c2

    if (lp<1) & (lp>=0.2):
        return c1*f1(m)
    elif (lp>=1) & (lp<(2.7-delta)):
        return c1*(f1(m)+(lp-1.)/(1.7-delta)*(f2(m)-f1(m)-a*delta))
    elif (lp>=(2.7-delta)) & (lp<(2.7+delta)):
        return c1*(f2(m)+a*(lp-2.7))
    elif (lp>=2.7+delta) & (lp<5.5):
        return c1*(f2(m)+a*delta+(lp-2.7-delta)/(2.8-delta)*(f3(m)-f2(m)-a*delta))
    elif (lp>=5.5) & (lp<8.0):
        return c1*(f3(m)*np.exp(-0.3*(lp-5.5)))
    else:
        return 0.

@njit('float64(float64, float64, float64)')
def gen_period(m, pmax, qmin):
    """
    :param float m: Mass of primary
    :param float pmax: Upper limit for period
    :param float qmin: Minimum mass ratio
    :return:
    """
    accept=False
    ##Compute pdf across grid of periods to get safer estimate of the maximum
    lpers=np.arange(0.2, np.log10(pmax), 0.005)
    ords=[pdf_lp(m, xx, qmin) for xx in lpers]
    ## of the pdf...
    while not accept:
        trial=np.random.uniform(0.2, np.log10(pmax))
        if (np.random.uniform(0, 1.1 * max(ords))<=pdf_lp(m, trial, qmin)):
                accept=True
    return 10.**trial

@njit('float64(float64, float64, float64)')
def r_pcomp1(m, pmax, qmin):
    """
    Radius where the number of companions goes to 1

    :param m: Primary mass
    :param qmin: Minimum mass ratio

    :return: Radius where the number of companions goes to 1
    :rtype: float
    """
    lpers = np.arange(0.2, np.log10(pmax), 0.005)
    ords = [pdf_lp(m, xx, qmin) for xx in lpers]
    integ = np.zeros(len(lpers))

    for ii in range(3, len(lpers)):
        integ[ii] = np.trapz(ords[:ii], lpers[:ii].copy())
    if max(integ) < 1:
        return pmax
    return 10.**np.interp(1.0, integ[3:], lpers[3:])

@njit('float64(float64, float64, float64)')
def get_norm(m, pmax, qmin):
    """
    Total multiplicity fraction

    :param m: Primary mass
    :param pmax: Maximum period
    :param qmin: Minimum mass ratio
    :return:
    """
    lpers=np.arange(0.2, np.log10(pmax), 0.005)
    ords=[pdf_lp(m, xx, qmin) for xx in lpers]
    return np.trapz(ords, lpers)


@njit('float64(float64, float64)')
def pdf_lp_noc(m, lp):
    """
    PDF of log period of binaries for q>=0.3

    :param float m: Primary mass
    :param float lp: Log period
    :param float qmin: Minimum mass ratio

    :return: pdf
    :rtype: float
    """
    ##Correction for low mass ratio binaries
    c1=1.0
    if (lp<1) & (lp>=0.2):
        return c1*f1(m)
    elif (lp>=1) & (lp<(2.7-delta)):
        return c1*(f1(m)+(lp-1.)/(1.7-delta)*(f2(m)-f1(m)-a*delta))
    elif (lp>=(2.7-delta)) & (lp<(2.7+delta)):
        return c1*(f2(m)+a*(lp-2.7))
    elif (lp>=2.7+delta) & (lp<5.5):
        return c1*(f2(m)+a*delta+(lp-2.7-delta)/(2.8-delta)*(f3(m)-f2(m)-a*delta))
    elif (lp>=5.5) & (lp<8.0):
        return c1*(f3(m)*np.exp(-0.3*(lp-5.5)))
    else:
        return 0.

@njit('float64(float64, float64)')
def gen_period_noc(m, pmax):
        accept=False
        while not accept:
                trial=np.random.uniform(0.2, np.log10(pmax))
                if (np.random.uniform(0,1)<=pdf_lp_noc(m, trial)/0.5):
                        accept=True
        return 10.**trial

@njit('float64(float64, float64)')
def get_norm_noc(m, pmax):
    lpers=np.arange(0.2, np.log10(pmax), 0.005)
    ords=[pdf_lp_noc(m, xx) for xx in lpers]
    return np.trapz(ords, lpers)

@njit('float64[:](float64, float64, float64)')
def gen_bin_noc(m, pmax, qmin):
    p1=gen_period_noc(m, pmax)
    e1=gen_e(m, p1)
    q1=gen_q(m, p1, qmin)
    m2=m*q1
    mbin=m*(1+q1)
    sma1=(p1/365.25*(mbin)**.5)**(2./3.)

    return np.array([p1, sma1, e1, q1, m2])


@njit('float64[:](float64, float64, float64)')
def gen_bin(m, pmax, qmin):
    """
    Generate binary according to distributions in Moe&di Stefano 2017

    :param m: Primary mass
    :param pmax: Maximum period
    :param qmin: Minimum mass ratio
    :return:
    """
    p1=gen_period(m, pmax, qmin)
    e1=gen_e(m, p1)
    q1=gen_q(m, p1, qmin)
    m2=m*q1
    mbin=m*(1+q1)
    sma1=(p1/365.25*(mbin)**.5)**(2./3.)

    return np.array([p1, sma1, e1, q1, m2])

def gen_mult(m, pmax, qmin, ncomp):
    dlp = 0.1
    bins_lp = np.arange(0, 8.1, dlp)
    
    mults = []
    for lp in bins_lp:
        pcomp = pdf_lp(mass1, lp, 0.1) * dlp
        ##Does it actually make sense to do this 
        ptest = np.random.uniform(0, 1)
        if ptest < pcomp: 
            p1 = 10.**np.random.uniform(lp, lp + dp)
            e1 = gen_e(m, p1)
            q1 = gen_q(m, p1, qmin)
            m2 = m*q1
            mbin = m*(1+q1)
            sma1 = (p1/365.25*(mbin)**.5)**(2./3.)

            mults.append([p1, sma1, e1, q1, m2])

    return mults




