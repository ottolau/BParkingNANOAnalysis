#include "TMath.h"
#include "models.h"

double DoubleCBFast(double x, double mean, double width, double alpha1, double n1, double alpha2, double n2){
   double t = (x-mean)*vdt::fast_inv(width);
   double val = -99.;
   if(t>-alpha1 && t<alpha2){
     val = vdt::fast_exp(-0.5*t*t);
   }else if(t<=-alpha1){
     double alpha1invn1 = alpha1*vdt::fast_inv(n1);
     val = vdt::fast_exp(-0.5*alpha1*alpha1)*fast_pow(1. - alpha1invn1*(alpha1+t), -n1);
     
   }else if(t>=alpha2){
     double alpha2invn2 = alpha2*vdt::fast_inv(n2);
     val = vdt::fast_exp(-0.5*alpha2*alpha2)*fast_pow(1. - alpha2invn2*(alpha2-t), -n2);     
     
   }
     
   if (!std::isnormal(val)) {
     printf("bad val: x = %5f, t = %5f, mean = %5f, sigma = %5f, alpha1 = %5f, n1 = %5f, alpha2 = %5f, n2 = %5f\n",double(x), t, double(mean),double(width),double(alpha1),double(n1),double(alpha2), double(n2));
     printf("val = %5f\n",val);
   }
     
   return val; 

}
