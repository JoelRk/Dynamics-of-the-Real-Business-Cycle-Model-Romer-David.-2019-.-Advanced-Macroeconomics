# Dynamics-of-the-Real-Business-Cycle-Model-Romer-David.-2019-.-Advanced-Macroeconomics
In this Project, I try compute and plot impulse responses for shocks to technology and government spending according to a real-business cycle model. For the theory behind this notebook, see Romer (2019), chapter 5, Real-Business-Cycle Theory. The key equations can be found below. Source of this code and methodology: Romer, David. (2019). Advanced Macroeconomics. New York: McGraw-Hill/Irwin.](https://www.mheducation.com/highered/product/advanced-macroeconomics-romer/M9781260185218.html

Key equations for the log-deviations from trend

Consumption and employment (eq. (5.51) and (5.52)): 𝐶̃𝑡+1≃𝑎𝐶𝐾𝐾̃𝑡+1+𝑎𝐶𝐴𝐴̃𝑡+1+𝑎𝐶𝐺𝐺̃𝑡+1; 𝐿̃𝑡+1≃𝑎𝐿𝐾𝐾̃𝑡+1+𝑎𝐿𝐴𝐴̃𝑡+1+𝑎𝐿𝐺𝐺̃𝑡+1 

Capital (eq. (5.53)): 𝐾̃𝑡+1≃𝑏𝐾𝐾𝐾̃𝑡+𝑏𝐾𝐴𝐴̃𝑡+𝑏𝐾𝐺𝐺̃𝑡 
Technology and government spending (eq. (5.9) and (5.11)): 𝐴̃𝑡+1≃𝜌𝐴𝐴̃𝑡; 𝐺̃𝑡+1≃𝜌𝐺𝐺̃𝑡 

Output (eq. (5.54)): 𝑌̃𝑡=𝛼𝐾̃𝑡+(1−𝛼)(𝐿̃𝑡+𝐴̃𝑡) 

Computing the impulse responses

With these key equations, we discuss the solution algorithm. Notice that we have two types of equations:
Equations for the evolution of the state variables, in which time subscripts on the right hand side are smaller than those on the left-hand side ( 𝐴̃𝑡+1 ,  𝐺̃𝑡+1 , and  𝐾̃𝑡+1 )
Equations for the controls and thus output, in which time subscripts of both sides of the equation are equal ( 𝐶̃𝑡+1 ,  𝐿̃𝑡+1 , and  𝑌̃𝑡 )
Using starting values (either the ones on the balanced growth path, or those implied by setting the initial value of the specific shocks, e.g.  𝐴̃0=1  to retrieve the impact of a  1%  technology shock), we can iterate over the time periods  𝑡=1,2,…,𝑇  and compute the value of each variable in the above equations based on the value we computed in the last iteration (or based on the the initial values in the first iteration). Note that it is crucial to start by computing the log-deviations of the state variables  𝐴̃𝑡 ,  𝐺̃𝑡 , and  𝐾̃𝑡  because these values are needed in the same time iteration for the computation of the log-deviations of the control variables  𝐶̃𝑡  and  𝐿̃𝑡 , and thus  𝑌̃𝑡 
Note on the units: because the variables are log-deviations from the balanced growth path, a deviation of  1  denotes approximately a deviation of  1% .
