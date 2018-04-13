// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#include <cmath>
#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"

using namespace std;

int main(int argc, char *argv[])
{
if( argc < 3)
    {
        cerr << " Usage : " << argv[0] << "Theta_filename Test_filename.csv" << endl;
        return -1;
    }

// Summarizing options
cout << " ** summarize options : " << endl;
cout << " \t Theta file    : " << argv[1] << endl;
cout << " \t Testing  file : " << argv[2] << endl;

///////////////////////////////////////////////////////////
//           START YOUR MODIFICATIONS HERE               //
///////////////////////////////////////////////////////////












cout << " end." << endl;
return 0;
}
