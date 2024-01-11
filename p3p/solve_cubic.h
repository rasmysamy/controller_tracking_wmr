#pragma once


#include <math.h>





namespace cvl{



// template<class T> T cubick_new(T c2, T c1, T c0) {
//     T root;
//     T a = c1 - c2 * c2 / 3.0;
//     T b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
//     T c = b * b / 4.0 + a * a * a / 27.0;
//     if (c >= 0.0 || std::abs(c) < 1.0e-26) {
//         c = std::sqrt(std::abs(c));
//         b *= -0.5;
//         root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
//     } else {
//         c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
//         root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
//     }


//     return root;
// }



    template<class T> T cubick_new(T c2, T c1, T c0) {
        T root;
        T a = c1 - c2 * c2 / 3.0;
        T b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
        T c = b * b / 4.0 + a * a * a / 27.0;
        if (c > 1.0e-26) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
        } else if (c < -1.0e-26) {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
        else {
            root = 3.0*b/a - c2 / 3.0;
        }


        return root;
    }

    template<class T> inline bool root2real(T b, T c,T& r1, T& r2){



        T v=b*b -4.0*c;
        if(v<-1.0e-12)
        {
            r1=r2=-0.5*b;
            return v>=0;
            // return true;
        }
        if(v>-1.0e-12 && v<0.0)
        {
            r1=-0.5*b;
            r2=-2;
            // return v>=0;
            return true;
        }

        T y=std::sqrt(v);
        if(b<0){
            r1= 0.5*(-b+y);
            r2= 0.5*(-b-y);
        }else{
            r1= 2.0*c/(-b+y);
            r2= 2.0*c/(-b-y);
        }
        return true;
    }


}