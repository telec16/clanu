// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#if defined(_OPENMP)
    #include <omp.h>
#endif

#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"



using namespace std;

int main(int argc, char *argv[])
{
    // Test some compialtion options
    #if defined(_OPENMP)
        cout << " OPENMP is activated  : great! " << endl;
    #else
        cout << " OPENMP is not activated  (good for debug)" << endl;
    #endif

    #ifdef __FAST_MATH__
        cout << " fast-math is activated : great! " << endl;
    #else
        cout << " fast-math is strangely not activated " << endl;
    #endif


        ///////////////////////////////////////////////////////////
        //           START YOUR MODIFICATIONS HERE               //
        ///////////////////////////////////////////////////////////
















cout << " end." << endl;
return 0;
}
