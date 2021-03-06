# Dynamics-of-the-Real-Business-Cycle-Model-Romer-David.-2019-.-Advanced-Macroeconomics
In this Project, I try compute and plot impulse responses for shocks to technology and government spending according to a real-business cycle model. For the theory behind this notebook, see Romer (2019), chapter 5, Real-Business-Cycle Theory. The key equations can be found below. Source of this code and methodology: Romer, David. (2019). Advanced Macroeconomics. New York: McGraw-Hill/Irwin.](https://www.mheducation.com/highered/product/advanced-macroeconomics-romer/M9781260185218.html

Key equations for the log-deviations from trend

Consumption and employment (eq. (5.51) and (5.52)): πΆΜπ‘+1βππΆπΎπΎΜπ‘+1+ππΆπ΄π΄Μπ‘+1+ππΆπΊπΊΜπ‘+1; πΏΜπ‘+1βππΏπΎπΎΜπ‘+1+ππΏπ΄π΄Μπ‘+1+ππΏπΊπΊΜπ‘+1 

Capital (eq. (5.53)): πΎΜπ‘+1βππΎπΎπΎΜπ‘+ππΎπ΄π΄Μπ‘+ππΎπΊπΊΜπ‘ 
Technology and government spending (eq. (5.9) and (5.11)): π΄Μπ‘+1βππ΄π΄Μπ‘; πΊΜπ‘+1βππΊπΊΜπ‘ 

Output (eq. (5.54)): πΜπ‘=πΌπΎΜπ‘+(1βπΌ)(πΏΜπ‘+π΄Μπ‘) 

Computing the impulse responses

With these key equations, we discuss the solution algorithm. Notice that we have two types of equations:
Equations for the evolution of the state variables, in which time subscripts on the right hand side are smaller than those on the left-hand side ( π΄Μπ‘+1 ,  πΊΜπ‘+1 , and  πΎΜπ‘+1 )
Equations for the controls and thus output, in which time subscripts of both sides of the equation are equal ( πΆΜπ‘+1 ,  πΏΜπ‘+1 , and  πΜπ‘ )
Using starting values (either the ones on the balanced growth path, or those implied by setting the initial value of the specific shocks, e.g.  π΄Μ0=1  to retrieve the impact of a  1%  technology shock), we can iterate over the time periods  π‘=1,2,β¦,π  and compute the value of each variable in the above equations based on the value we computed in the last iteration (or based on the the initial values in the first iteration). Note that it is crucial to start by computing the log-deviations of the state variables  π΄Μπ‘ ,  πΊΜπ‘ , and  πΎΜπ‘  because these values are needed in the same time iteration for the computation of the log-deviations of the control variables  πΆΜπ‘  and  πΏΜπ‘ , and thus  πΜπ‘ 
Note on the units: because the variables are log-deviations from the balanced growth path, a deviation of  1  denotes approximately a deviation of  1% .
