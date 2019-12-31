#include "vdt/vdtMath.h"
#include <cmath>
double fast_pow(double base, double exponent){
   if (base==0. && exponent>0.) return 0.;
   else if (base>0.) return vdt::fast_exp(exponent*vdt::fast_log(base));
   else return std::nan("");
}

