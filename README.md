Stochastically generate multiple systems following the prescriptions in Moe and Di Stefano 2017, using the function 
"gen_mult" 

This function takes: Primary mass [solar masses], Maximum period [days], Minimum mass ratio (0.1 recommended), Maximum number of companions to generate.
It returns a list of companion properties. Each row contains period (days), sma (au), eccentricity, mass ratio, and companion mass (solar masses). If no companions are generate then return a list of -1's

Package requirements: numpy and numba 

Example: 

```
from best_fits import gen_mult

gen_mult(10.0, 1e8, 0.1, 2)
array([[9.90200198e+03, 2.08422997e+01, 1.99034686e-01, 2.31888337e-01,
        2.31888337e+00]])

```
